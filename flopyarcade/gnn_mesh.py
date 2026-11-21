"""Graph neural network on MODFLOW mesh (ROADMAP R12).

Nodes = grid cells, edges = 4-connected adjacency.
Message-passing GNN predicts hydraulic heads from boundary conditions.
Agent reads node-level features for observation.

Pure numpy; no external ML dependencies.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field


@dataclass
class MeshNode:
    """A single grid cell node in the MODFLOW mesh graph.

    Attributes:
        row: Row index in the grid (0-based).
        col: Column index in the grid (0-based).
        head: Hydraulic head value at this node.
        k_value: Hydraulic conductivity of the cell.
        neighbors: List of neighbor node indices (4-connected).
    """

    row: int
    col: int
    head: float = 0.0
    k_value: float = 1.0
    neighbors: list[int] = field(default_factory=list)


class MeshGraph:
    """4-connected grid graph representing a MODFLOW mesh.

    Node index = row * cols + col.
    Edges connect each cell to its up to 4 orthogonal neighbors.
    """

    def __init__(self, rows: int, cols: int) -> None:
        """Create a rows x cols 4-connected grid graph.

        Args:
            rows: Number of grid rows.
            cols: Number of grid columns.
        """
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        self.rows = rows
        self.cols = cols
        self._nodes: list[MeshNode] = []
        self._build()

    def _build(self) -> None:
        """Construct nodes and 4-connected adjacency."""
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                neighbors: list[int] = []
                # up
                if r > 0:
                    neighbors.append((r - 1) * self.cols + c)
                # down
                if r < self.rows - 1:
                    neighbors.append((r + 1) * self.cols + c)
                # left
                if c > 0:
                    neighbors.append(r * self.cols + (c - 1))
                # right
                if c < self.cols - 1:
                    neighbors.append(r * self.cols + (c + 1))
                self._nodes.append(MeshNode(row=r, col=c, neighbors=neighbors))

    def nodes(self) -> list[MeshNode]:
        """Return all mesh nodes."""
        return self._nodes

    def n_nodes(self) -> int:
        """Return total number of nodes."""
        return len(self._nodes)

    def n_edges(self) -> int:
        """Return number of undirected edges (each counted once)."""
        total = sum(len(n.neighbors) for n in self._nodes)
        return total // 2

    def adjacency(self) -> list[list[int]]:
        """Return adjacency list for all nodes."""
        return [n.neighbors for n in self._nodes]

    def set_head(self, idx: int, head: float) -> None:
        """Set hydraulic head at a specific node.

        Args:
            idx: Node index (row * cols + col).
            head: Head value to set.
        """
        if not 0 <= idx < len(self._nodes):
            raise IndexError(f"Node index {idx} out of range [0, {len(self._nodes)})")
        self._nodes[idx].head = head

    def set_all_heads(self, heads: list[float]) -> None:
        """Set heads for all nodes.

        Args:
            heads: List of head values, one per node.
        """
        if len(heads) != len(self._nodes):
            raise ValueError(f"Expected {len(self._nodes)} heads, got {len(heads)}")
        for i, h in enumerate(heads):
            self._nodes[i].head = h

    def get_boundary_nodes(self) -> list[int]:
        """Return indices of boundary nodes (fewer than 4 neighbors)."""
        return [i for i, n in enumerate(self._nodes) if len(n.neighbors) < 4]

    def get_interior_nodes(self) -> list[int]:
        """Return indices of interior nodes (exactly 4 neighbors)."""
        return [i for i, n in enumerate(self._nodes) if len(n.neighbors) == 4]


class MessagePassingLayer:
    """Single message-passing layer with pure numpy operations.

    For each node: aggregate mean of neighbor features,
    transform with weight matrix + bias, apply ReLU.
    """

    def __init__(self, in_dim: int = 1, out_dim: int = 8, seed: int | None = None) -> None:
        """Initialize layer dimensions and optionally seed RNG.

        Args:
            in_dim: Input feature dimension per node.
            out_dim: Output feature dimension per node.
            seed: Optional random seed for reproducibility.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._seed = seed
        self._rng = random.Random(seed)
        # Weight: out_dim x in_dim, Bias: out_dim (Glorot-scaled init)
        std_w = 1.0 / math.sqrt(max(1, in_dim))
        std_b = 0.01
        self._weight = [[self._rng.gauss(0, std_w) for _ in range(in_dim)] for _ in range(out_dim)]
        self._bias = [self._rng.gauss(0, std_b) for _ in range(out_dim)]

    def random_init(self) -> None:
        """Re-initialize weights and bias with random values (Glorot-scaled)."""
        std_w = 1.0 / math.sqrt(max(1, self.in_dim))
        self._weight = [
            [self._rng.gauss(0, std_w) for _ in range(self.in_dim)]
            for _ in range(self.out_dim)
        ]
        self._bias = [self._rng.gauss(0, 0.01) for _ in range(self.out_dim)]

    def _relu(self, x: float) -> float:
        return max(0.0, x)

    def forward(self, node_features: list[float], adjacency: list[list[int]]) -> list[float]:
        """Single message-passing step.

        Args:
            node_features: Flat list of length n_nodes * in_dim.
            adjacency: Adjacency list (list of neighbor index lists).

        Returns:
            Flat list of length n_nodes * out_dim.
        """
        n_nodes = len(adjacency)
        out: list[float] = []

        for i in range(n_nodes):
            # Extract own features
            own = node_features[i * self.in_dim : (i + 1) * self.in_dim]
            # Aggregate mean of neighbor features
            nbrs = adjacency[i]
            if nbrs:
                agg = [0.0] * self.in_dim
                for nb in nbrs:
                    nb_feat = node_features[nb * self.in_dim : (nb + 1) * self.in_dim]
                    for d in range(self.in_dim):
                        agg[d] += nb_feat[d]
                inv = 1.0 / len(nbrs)
                agg = [a * inv for a in agg]
            else:
                agg = own[:]  # no neighbors, use own features

            # Combine own + aggregated (concatenate then project)
            # Simple approach: use aggregated as input to linear layer
            combined = [own[d] + agg[d] for d in range(self.in_dim)]

            # Linear transform + ReLU
            for o in range(self.out_dim):
                val = self._bias[o]
                for d in range(self.in_dim):
                    val += self._weight[o][d] * combined[d]
                out.append(self._relu(val))

        return out

    def forward_multi(
        self,
        features: list[float],
        mesh_graph: MeshGraph,
        n_layers: int = 2,
    ) -> list[float]:
        """Apply the message-passing layer multiple times.

        After the first layer, input dimension becomes out_dim.
        Subsequent layers use out_dim as both in and out.

        Args:
            features: Initial flat feature list (n_nodes * in_dim).
            mesh_graph: The mesh graph providing adjacency.
            n_layers: Number of layers to stack.

        Returns:
            Flat feature list after n_layers message-passing steps.
        """
        adj = mesh_graph.adjacency()
        n_nodes = mesh_graph.n_nodes()
        current = features
        for layer_i in range(n_layers):
            if layer_i == 0:
                current = self.forward(current, adj)
            else:
                # Now in_dim = out_dim of previous layer
                cur_in_dim = self.out_dim
                # Use same weights but with adjusted in_dim
                out: list[float] = []
                for i in range(n_nodes):
                    own = current[i * cur_in_dim : (i + 1) * cur_in_dim]
                    nbrs = adj[i]
                    if nbrs:
                        agg = [0.0] * cur_in_dim
                        for nb in nbrs:
                            nb_feat = current[nb * cur_in_dim : (nb + 1) * cur_in_dim]
                            for d in range(cur_in_dim):
                                agg[d] += nb_feat[d]
                        inv = 1.0 / len(nbrs)
                        agg = [a * inv for a in agg]
                    else:
                        agg = own[:]
                    combined = [own[d] + agg[d] for d in range(cur_in_dim)]
                    for o in range(self.out_dim):
                        val = self._bias[o]
                        # Use available weight columns (min of cur_in_dim and in_dim)
                        for d in range(min(cur_in_dim, self.in_dim)):
                            val += self._weight[o][d] * combined[d]
                        out.append(max(0.0, val))
                current = out
        return current

    def save_weights(self, path: str) -> None:
        """Save weights and bias to a JSON file.

        Args:
            path: File path for JSON output.
        """
        data = {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "weight": self._weight,
            "bias": self._bias,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_weights(cls, path: str) -> MessagePassingLayer:
        """Load a MessagePassingLayer from a JSON file.

        Args:
            path: File path to read JSON from.

        Returns:
            A new MessagePassingLayer with loaded parameters.
        """
        with open(path, "r") as f:
            data = json.load(f)
        layer = cls(in_dim=data["in_dim"], out_dim=data["out_dim"])
        layer._weight = data["weight"]
        layer._bias = data["bias"]
        return layer


class HeadPredictor:
    """Predict interior hydraulic heads from boundary conditions using GNN.

    Uses 2 stacked message-passing layers + linear readout.
    """

    def __init__(self, mesh: MeshGraph, n_hidden: int = 8, seed: int | None = None) -> None:
        """Initialize the head predictor.

        Args:
            mesh: The mesh graph.
            n_hidden: Hidden dimension for GNN layers.
            seed: Optional random seed.
        """
        self.mesh = mesh
        self.n_hidden = n_hidden
        self._seed = seed
        self._rng = random.Random(seed)

        n = mesh.n_nodes()
        n_int = len(mesh.get_interior_nodes())

        # Layer 1: in_dim=1 -> n_hidden
        self._layer1 = MessagePassingLayer(in_dim=1, out_dim=n_hidden, seed=seed)
        # Layer 2: in_dim=n_hidden -> n_hidden
        self._layer2_in = n_hidden
        self._layer2_out = n_hidden
        std_w2 = 1.0 / math.sqrt(max(1, n_hidden))
        self._w2 = [[self._rng.gauss(0, std_w2) for _ in range(n_hidden)] for _ in range(n_hidden)]
        self._b2 = [self._rng.gauss(0, 0.01) for _ in range(n_hidden)]

        # Readout: n_hidden -> 1 (for interior nodes only)
        std_ro = 1.0 / math.sqrt(max(1, n_hidden))
        self._w_readout = [self._rng.gauss(0, std_ro) for _ in range(n_hidden)]
        self._b_readout = self._rng.gauss(0, 0.01)

        self._interior_indices = mesh.get_interior_nodes()

    def _forward_gnn(self, boundary_heads: dict[int, float]) -> list[float]:
        """Run 2-layer GNN forward pass.

        Args:
            boundary_heads: dict mapping boundary node index -> head value.

        Returns:
            Flat list of n_nodes * n_hidden features after 2 layers.
        """
        n = self.mesh.n_nodes()
        adj = self.mesh.adjacency()

        # Input features: head value for boundary nodes, 0 for interior
        features = [0.0] * n
        for idx, h in boundary_heads.items():
            features[idx] = h

        # Layer 1: in_dim=1, out_dim=n_hidden
        h1 = self._layer1.forward(features, adj)

        # Layer 2: in_dim=n_hidden, out_dim=n_hidden
        h2: list[float] = []
        for i in range(n):
            own = h1[i * self.n_hidden : (i + 1) * self.n_hidden]
            nbrs = adj[i]
            if nbrs:
                agg = [0.0] * self.n_hidden
                for nb in nbrs:
                    nb_feat = h1[nb * self.n_hidden : (nb + 1) * self.n_hidden]
                    for d in range(self.n_hidden):
                        agg[d] += nb_feat[d]
                inv = 1.0 / len(nbrs)
                agg = [a * inv for a in agg]
            else:
                agg = own[:]

            combined = [own[d] + agg[d] for d in range(self.n_hidden)]

            for o in range(self.n_hidden):
                val = self._b2[o]
                for d in range(self.n_hidden):
                    val += self._w2[o][d] * combined[d]
                h2.append(max(0.0, val))

        return h2

    def _readout(self, gnn_features: list[float]) -> list[float]:
        """Linear readout from GNN features to predicted heads for interior nodes.

        Args:
            gnn_features: Flat list of n_nodes * n_hidden.

        Returns:
            List of predicted heads for interior nodes.
        """
        predictions = []
        for idx in self._interior_indices:
            feat = gnn_features[idx * self.n_hidden : (idx + 1) * self.n_hidden]
            val = self._b_readout
            for d in range(self.n_hidden):
                val += self._w_readout[d] * feat[d]
            predictions.append(val)
        return predictions

    def predict(self, boundary_heads: dict[int, float]) -> list[float]:
        """Predict interior heads from boundary conditions.

        Args:
            boundary_heads: dict mapping boundary node index -> head value.

        Returns:
            List of predicted head values for interior nodes.
        """
        h2 = self._forward_gnn(boundary_heads)
        return self._readout(h2)

    def train(
        self,
        boundary_heads: dict[int, float],
        true_interior: list[float],
        lr: float = 0.01,
        n_steps: int = 100,
    ) -> list[float]:
        """Train via simple gradient descent on the readout layer.

        Only the readout weights are updated (GNN layers fixed).

        Args:
            boundary_heads: Boundary condition dict.
            true_interior: True interior head values.
            lr: Learning rate.
            n_steps: Number of training steps.

        Returns:
            List of loss values per step.
        """
        loss_history: list[float] = []
        h2 = self._forward_gnn(boundary_heads)
        n_int = len(self._interior_indices)

        for step in range(n_steps):
            # Forward
            preds = self._readout(h2)
            # MSE loss
            loss = sum((p - t) ** 2 for p, t in zip(preds, true_interior)) / max(1, n_int)
            loss_history.append(loss)

            # Gradient of MSE w.r.t. readout params
            grad_w = [0.0] * self.n_hidden
            grad_b = 0.0
            for i, idx in enumerate(self._interior_indices):
                feat = h2[idx * self.n_hidden : (idx + 1) * self.n_hidden]
                err = preds[i] - true_interior[i]
                grad_b += 2.0 * err / n_int
                for d in range(self.n_hidden):
                    grad_w[d] += 2.0 * err * feat[d] / n_int

            # Update
            for d in range(self.n_hidden):
                self._w_readout[d] -= lr * grad_w[d]
            self._b_readout -= lr * grad_b

        return loss_history

    def agent_features(self, node_idx: int, boundary_heads: dict[int, float]) -> list[float]:
        """Extract GNN feature vector at a specific node (for agent observation).

        Args:
            node_idx: Node index in the mesh.
            boundary_heads: Boundary condition dict.

        Returns:
            Feature vector of length n_hidden for that node.
        """
        h2 = self._forward_gnn(boundary_heads)
        return h2[node_idx * self.n_hidden : (node_idx + 1) * self.n_hidden]

    def compare_predictions(
        self,
        boundary_heads: dict[int, float],
        true_interior: list[float],
    ) -> dict:
        """Compare predicted vs true interior heads.

        Args:
            boundary_heads: Boundary condition dict.
            true_interior: True interior head values.

        Returns:
            dict with keys: mae, rmse, max_error, n_interior.
        """
        preds = self.predict(boundary_heads)
        n = len(true_interior)
        if n == 0:
            return {"mae": 0.0, "rmse": 0.0, "max_error": 0.0, "n_interior": 0}

        errors = [abs(p - t) for p, t in zip(preds, true_interior)]
        mae = sum(errors) / n
        rmse = math.sqrt(sum(e * e for e in errors) / n)
        max_error = max(errors)

        return {
            "mae": mae,
            "rmse": rmse,
            "max_error": max_error,
            "n_interior": n,
        }


def grid_to_graph(rows: int, cols: int) -> MeshGraph:
    """Convenience: create a MeshGraph from grid dimensions.

    Args:
        rows: Number of grid rows.
        cols: Number of grid columns.

    Returns:
        A MeshGraph instance.
    """
    return MeshGraph(rows, cols)


def format_gnn_report(
    mesh: MeshGraph,
    predictor: HeadPredictor,
    metrics: dict,
) -> str:
    """Format a human-readable GNN prediction report.

    Args:
        mesh: The mesh graph.
        predictor: The head predictor.
        metrics: Dict with mae, rmse, max_error, n_interior.

    Returns:
        Multi-line formatted report string.
    """
    lines = [
        "=" * 50,
        "  GNN Mesh Head Prediction Report",
        "=" * 50,
        f"  Grid: {mesh.rows} x {mesh.cols}",
        f"  Nodes: {mesh.n_nodes()}",
        f"  Edges: {mesh.n_edges()}",
        f"  Boundary nodes: {len(mesh.get_boundary_nodes())}",
        f"  Interior nodes: {len(mesh.get_interior_nodes())}",
        "-" * 50,
        f"  MAE:       {metrics.get('mae', 0.0):.6f}",
        f"  RMSE:      {metrics.get('rmse', 0.0):.6f}",
        f"  Max Error: {metrics.get('max_error', 0.0):.6f}",
        f"  N Interior: {metrics.get('n_interior', 0)}",
        "=" * 50,
    ]
    return "\n".join(lines)
