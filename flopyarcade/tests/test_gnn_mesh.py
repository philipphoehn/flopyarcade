"""Tests for GNN on MODFLOW mesh (ROADMAP R12)."""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from flopyarcade.gnn_mesh import (
    HeadPredictor,
    MeshGraph,
    MeshNode,
    MessagePassingLayer,
    format_gnn_report,
    grid_to_graph,
)


class TestMeshGraph:
    """Tests for MeshGraph creation and topology."""

    def test_3x3_creation(self):
        """3x3 grid has 9 nodes."""
        g = MeshGraph(3, 3)
        assert g.n_nodes() == 9
        assert len(g.nodes()) == 9

    def test_4_connected_neighbors(self):
        """Corners have 2, edges have 3, interior has 4 neighbors."""
        g = MeshGraph(3, 3)
        adj = g.adjacency()
        # Corners: (0,0),(0,2),(2,0),(2,2) -> indices 0,2,6,8
        assert len(adj[0]) == 2
        assert len(adj[2]) == 2
        assert len(adj[6]) == 2
        assert len(adj[8]) == 2
        # Edges (non-corner boundary): (0,1),(1,0),(1,2),(2,1) -> indices 1,3,5,7
        assert len(adj[1]) == 3
        assert len(adj[3]) == 3
        assert len(adj[5]) == 3
        assert len(adj[7]) == 3
        # Interior: (1,1) -> index 4
        assert len(adj[4]) == 4

    def test_n_edges_3x3(self):
        """3x3 grid: each edge counted once. Total directed = 2*3 + 4*3 = 24, undirected = 12."""
        g = MeshGraph(3, 3)
        # Manual count: horizontal edges: 3 rows * 2 cols = 6
        # vertical edges: 3 cols * 2 rows = 6
        # total = 12
        assert g.n_edges() == 12

    def test_boundary_interior_split(self):
        """3x3: 8 boundary, 1 interior."""
        g = MeshGraph(3, 3)
        boundary = g.get_boundary_nodes()
        interior = g.get_interior_nodes()
        assert len(boundary) == 8
        assert len(interior) == 1
        assert interior == [4]
        assert set(boundary) == {0, 1, 2, 3, 5, 6, 7, 8}


class TestMessagePassingLayer:
    """Tests for the message-passing layer."""

    def test_forward_shape(self):
        """Forward output has n_nodes * out_dim elements."""
        layer = MessagePassingLayer(in_dim=1, out_dim=8, seed=42)
        g = MeshGraph(3, 3)
        adj = g.adjacency()
        features = [1.0] * 9  # n_nodes * in_dim
        out = layer.forward(features, adj)
        assert len(out) == 9 * 8

    def test_relu_non_negative(self):
        """All outputs must be >= 0 due to ReLU."""
        layer = MessagePassingLayer(in_dim=1, out_dim=4, seed=7)
        g = MeshGraph(2, 3)
        adj = g.adjacency()
        features = [-1.0, 2.0, -3.0, 0.5, 1.0, -0.5]
        out = layer.forward(features, adj)
        assert all(v >= 0.0 for v in out)

    def test_multi_layer(self):
        """forward_multi applies layer twice, output dim = n_nodes * out_dim."""
        layer = MessagePassingLayer(in_dim=1, out_dim=4, seed=3)
        g = MeshGraph(2, 2)
        features = [0.5, 1.0, -0.5, 2.0]
        out = layer.forward_multi(features, g, n_layers=2)
        # After first layer: n_nodes * out_dim = 4 * 4 = 16
        # After second layer: the in_dim is still 4 (out_dim of first), out_dim=4
        # So output = 4 * 4 = 16
        assert len(out) == 4 * 4


class TestHeadPredictor:
    """Tests for the head predictor."""

    def test_predict_shape(self):
        """predict returns one value per interior node."""
        g = MeshGraph(4, 4)
        predictor = HeadPredictor(g, n_hidden=8, seed=42)
        boundary = g.get_boundary_nodes()
        boundary_heads = {i: 1.0 for i in boundary}
        preds = predictor.predict(boundary_heads)
        n_interior = len(g.get_interior_nodes())
        assert len(preds) == n_interior

    def test_train_reduces_loss(self):
        """Training should reduce loss over steps."""
        g = MeshGraph(3, 3)
        predictor = HeadPredictor(g, n_hidden=8, seed=42)
        boundary = g.get_boundary_nodes()
        boundary_heads = {i: 1.5 for i in boundary}
        true_interior = [1.2]
        loss_history = predictor.train(boundary_heads, true_interior, lr=0.1, n_steps=200)
        assert loss_history[0] > loss_history[-1]

    def test_agent_features_length(self):
        """agent_features returns n_hidden values."""
        g = MeshGraph(3, 3)
        predictor = HeadPredictor(g, n_hidden=8, seed=42)
        boundary = g.get_boundary_nodes()
        boundary_heads = {i: 1.0 for i in boundary}
        feats = predictor.agent_features(4, boundary_heads)
        assert len(feats) == 8

    def test_compare_predictions_structure(self):
        """compare_predictions returns required keys with correct types."""
        g = MeshGraph(3, 3)
        predictor = HeadPredictor(g, n_hidden=8, seed=42)
        boundary = g.get_boundary_nodes()
        boundary_heads = {i: 2.0 for i in boundary}
        true_interior = [1.8]
        result = predictor.compare_predictions(boundary_heads, true_interior)
        assert "mae" in result
        assert "rmse" in result
        assert "max_error" in result
        assert "n_interior" in result
        assert isinstance(result["mae"], float)
        assert isinstance(result["rmse"], float)
        assert isinstance(result["max_error"], float)
        assert result["n_interior"] == 1

    def test_constant_boundary_uniform_interior(self):
        """With constant boundary heads, interior predictions should be approximately uniform.

        After training on a constant target, the spread of predictions should be small.
        """
        g = MeshGraph(4, 4)
        predictor = HeadPredictor(g, n_hidden=8, seed=42)
        boundary = g.get_boundary_nodes()
        boundary_heads = {i: 2.0 for i in boundary}
        interior_nodes = g.get_interior_nodes()
        true_interior = [2.0] * len(interior_nodes)
        predictor.train(boundary_heads, true_interior, lr=0.1, n_steps=500)
        preds = predictor.predict(boundary_heads)
        # Check that predictions are approximately uniform (low spread)
        mean_p = sum(preds) / len(preds)
        spread = max(abs(p - mean_p) for p in preds)
        assert spread < 0.5, f"Predictions not uniform: spread={spread}"


class TestUtilities:
    """Tests for utility functions."""

    def test_grid_to_graph_5x5(self):
        """grid_to_graph creates a 5x5 MeshGraph."""
        g = grid_to_graph(5, 5)
        assert isinstance(g, MeshGraph)
        assert g.rows == 5
        assert g.cols == 5
        assert g.n_nodes() == 25
        # 5x5: horizontal edges = 5*4=20, vertical = 4*5=20, total=40
        assert g.n_edges() == 40

    def test_save_load_weights_roundtrip(self):
        """save_weights then load_weights produces identical forward output."""
        layer = MessagePassingLayer(in_dim=2, out_dim=6, seed=99)
        g = MeshGraph(3, 3)
        adj = g.adjacency()
        # 9 nodes * in_dim 2 = 18 feature values
        features = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9,
                    0.2, -0.3, 0.4, -0.1, 0.5, -0.4, 0.3, -0.6, 0.7]
        expected = layer.forward(features, adj)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            layer.save_weights(path)
            loaded = MessagePassingLayer.load_weights(path)
            result = loaded.forward(features, adj)
            assert len(result) == len(expected)
            for a, b in zip(result, expected):
                assert abs(a - b) < 1e-10
        finally:
            os.unlink(path)

    def test_format_gnn_report(self):
        """format_gnn_report returns a meaningful string."""
        g = MeshGraph(3, 3)
        predictor = HeadPredictor(g, n_hidden=8, seed=42)
        metrics = {"mae": 0.123, "rmse": 0.156, "max_error": 0.2, "n_interior": 1}
        report = format_gnn_report(g, predictor, metrics)
        assert "GNN" in report
        assert "3 x 3" in report
        assert "Nodes: 9" in report
        assert "MAE" in report
        assert "0.123" in report

    def test_2x2_minimum_grid(self):
        """2x2 grid is the minimum valid grid; all nodes are boundary."""
        g = MeshGraph(2, 2)
        assert g.n_nodes() == 4
        # Each node has 2 neighbors
        adj = g.adjacency()
        for nbrs in adj:
            assert len(nbrs) == 2
        # No interior nodes in 2x2
        assert g.get_interior_nodes() == []
        # All 4 are boundary
        assert len(g.get_boundary_nodes()) == 4
        # Edges: 4 (2 horizontal + 2 vertical)
        assert g.n_edges() == 4
        # Predictor should work (empty interior means predict returns [])
        predictor = HeadPredictor(g, n_hidden=4, seed=1)
        boundary_heads = {i: 1.0 for i in range(4)}
        preds = predictor.predict(boundary_heads)
        assert preds == []
