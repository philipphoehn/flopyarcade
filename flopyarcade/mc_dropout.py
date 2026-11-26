"""R11: MC-Dropout Ensemble Uncertainty Quantification.

Implements a 5-head DQN ensemble with Monte-Carlo dropout for
uncertainty estimation, exploration bonus, and intrinsic motivation.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DQNHead:
    """A single DQN head with flat weight and bias vectors."""

    index: int
    weights: List[float]
    biases: List[float]


class MCEnsemble:
    """MC-Dropout ensemble of DQN heads for uncertainty quantification.

    Architecture per head: linear(input_dim -> hidden_dim) + ReLU + linear(hidden_dim -> output_dim)
    where hidden_dim == input_dim by default.
    """

    def __init__(
        self,
        n_heads: int = 5,
        input_dim: int = 64,
        output_dim: int = 5,
        dropout_rate: float = 0.3,
        seed: int | None = None,
    ):
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.hidden_dim = input_dim
        self.heads: List[DQNHead] = []
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def random_init(self) -> None:
        """Initialise all heads with random weights (deterministic with seed)."""
        self.heads = []
        for i in range(self.n_heads):
            w1 = [self._rng.gauss(0, 0.1) for _ in range(self.input_dim * self.hidden_dim)]
            b1 = [0.0] * self.hidden_dim
            w2 = [self._rng.gauss(0, 0.1) for _ in range(self.hidden_dim * self.output_dim)]
            b2 = [0.0] * self.output_dim
            self.heads.append(
                DQNHead(index=i, weights=w1 + w2, biases=b1 + b2)
            )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _split(self, head: DQNHead):
        """Split flat vectors into (W1, b1, W2, b2)."""
        n_w1 = self.input_dim * self.hidden_dim
        n_b1 = self.hidden_dim
        W1 = head.weights[:n_w1]
        W2 = head.weights[n_w1:]
        b1 = head.biases[:n_b1]
        b2 = head.biases[n_b1:]
        return W1, b1, W2, b2

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_single(self, head: DQNHead, x: List[float]) -> List[float]:
        """Deterministic forward: linear + ReLU + linear."""
        W1, b1, W2, b2 = self._split(head)
        # Layer 1
        h = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            val = b1[j]
            for i in range(self.input_dim):
                val += x[i] * W1[i * self.hidden_dim + j]
            h[j] = max(0.0, val)
        # Layer 2
        out = [0.0] * self.output_dim
        for j in range(self.output_dim):
            val = b2[j]
            for i in range(self.hidden_dim):
                val += h[i] * W2[i * self.output_dim + j]
            out[j] = val
        return out

    def _forward_dropout(self, head: DQNHead, x: List[float], mask: List[float]) -> List[float]:
        """Forward with a pre-computed dropout mask on hidden units."""
        W1, b1, W2, b2 = self._split(head)
        h = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            val = b1[j]
            for i in range(self.input_dim):
                val += x[i] * W1[i * self.hidden_dim + j]
            h[j] = max(0.0, val) * mask[j]
        out = [0.0] * self.output_dim
        for j in range(self.output_dim):
            val = b2[j]
            for i in range(self.hidden_dim):
                val += h[i] * W2[i * self.output_dim + j]
            out[j] = val
        return out

    def forward_ensemble(self, x: List[float], n_samples: int = 10) -> List[List[float]]:
        """Run every head *n_samples* times with random dropout masks."""
        keep = 1.0 - self.dropout_rate
        results: List[List[float]] = []
        for head in self.heads:
            for _ in range(n_samples):
                mask = [1.0 if self._rng.random() < keep else 0.0 for _ in range(self.hidden_dim)]
                results.append(self._forward_dropout(head, x, mask))
        return results

    # ------------------------------------------------------------------
    # Uncertainty metrics
    # ------------------------------------------------------------------

    def mean_std(self, x: List[float], n_samples: int = 10) -> Tuple[List[float], List[float]]:
        """Return (mean_Q, std_Q) across all head samples."""
        all_q = self.forward_ensemble(x, n_samples)
        n = len(all_q)
        if n == 0:
            return [0.0] * self.output_dim, [0.0] * self.output_dim
        mean = [sum(q[i] for q in all_q) / n for i in range(self.output_dim)]
        var = [sum((q[i] - mean[i]) ** 2 for q in all_q) / n for i in range(self.output_dim)]
        std = [math.sqrt(v) for v in var]
        return mean, std

    def uncertainty_bonus(self, x: List[float], n_samples: int = 10, beta: float = 1.0) -> List[float]:
        """beta * std_Q per action – the exploration bonus vector."""
        _, std_q = self.mean_std(x, n_samples)
        return [beta * s for s in std_q]

    def intrinsic_reward(self, x: List[float], n_samples: int = 10) -> float:
        """max(std_Q) – intrinsic motivation signal."""
        _, std_q = self.mean_std(x, n_samples)
        return max(std_q)

    def confidence(self, x: List[float], n_samples: int = 10) -> float:
        """1 - max_std/(max_mean+1e-8), clamped to [0, 1]."""
        mean_q, std_q = self.mean_std(x, n_samples)
        conf = 1.0 - max(std_q) / (max(mean_q) + 1e-8)
        return max(0.0, min(1.0, conf))

    def action_with_bonus(self, x: List[float], n_samples: int = 10, beta: float = 1.0) -> int:
        """argmax(mean_Q + beta * std_Q)."""
        mean_q, std_q = self.mean_std(x, n_samples)
        best = 0
        best_score = mean_q[0] + beta * std_q[0]
        for i in range(1, self.output_dim):
            s = mean_q[i] + beta * std_q[i]
            if s > best_score:
                best_score = s
                best = i
        return best

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, x: List[float], target: List[float], lr: float = 0.01) -> float:
        """One gradient-descent step on all heads toward *target*. Returns MSE loss."""
        head_outs = [self.forward_single(h, x) for h in self.heads]
        nh = len(head_outs)
        mean_out = [sum(ho[i] for ho in head_outs) / nh for i in range(self.output_dim)]
        loss = sum((mean_out[i] - target[i]) ** 2 for i in range(self.output_dim)) / self.output_dim

        # dL/d(head_output) – each head contributes 1/nh to the ensemble mean
        d_out = [2.0 * (mean_out[i] - target[i]) / (nh * self.output_dim) for i in range(self.output_dim)]

        for head in self.heads:
            W1, b1, W2, b2 = self._split(head)
            # Forward (recompute hidden for gradient)
            h = [0.0] * self.hidden_dim
            for j in range(self.hidden_dim):
                val = b1[j]
                for i in range(self.input_dim):
                    val += x[i] * W1[i * self.hidden_dim + j]
                h[j] = max(0.0, val)

            # dh (through ReLU gate)
            dh = [0.0] * self.hidden_dim
            for i in range(self.hidden_dim):
                if h[i] > 0.0:
                    s = 0.0
                    for j in range(self.output_dim):
                        s += W2[i * self.output_dim + j] * d_out[j]
                    dh[i] = s

            new_w = head.weights[:]
            new_b = head.biases[:]
            # W2 update
            for i in range(self.hidden_dim):
                for j in range(self.output_dim):
                    idx = self.input_dim * self.hidden_dim + i * self.output_dim + j
                    new_w[idx] -= lr * h[i] * d_out[j]
            # W1 update
            for i in range(self.input_dim):
                for j in range(self.hidden_dim):
                    idx = i * self.hidden_dim + j
                    new_w[idx] -= lr * x[i] * dh[j]
            # b2 update
            for j in range(self.output_dim):
                new_b[self.hidden_dim + j] -= lr * d_out[j]
            # b1 update
            for j in range(self.hidden_dim):
                new_b[j] -= lr * dh[j]

            head.weights = new_w
            head.biases = new_b

        return loss

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save ensemble to a JSON file."""
        state = {
            "n_heads": self.n_heads,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "seed": self.seed,
            "heads": [
                {"index": h.index, "weights": h.weights, "biases": h.biases}
                for h in self.heads
            ],
        }
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load_state(cls, path: str) -> "MCEnsemble":
        """Load ensemble from a JSON file."""
        with open(path) as f:
            state = json.load(f)
        ens = cls(
            n_heads=state["n_heads"],
            input_dim=state["input_dim"],
            output_dim=state["output_dim"],
            dropout_rate=state["dropout_rate"],
            seed=state.get("seed"),
        )
        ens.heads = [
            DQNHead(index=h["index"], weights=h["weights"], biases=h["biases"])
            for h in state["heads"]
        ]
        return ens


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def compare_uncertainty(ensemble_a: MCEnsemble, ensemble_b: MCEnsemble, x: List[float]) -> dict:
    """Compare epistemic uncertainty of two ensembles on input *x*."""
    _, std_a = ensemble_a.mean_std(x)
    _, std_b = ensemble_b.mean_std(x)
    max_a, max_b = max(std_a), max(std_b)
    return {
        "std_a": std_a,
        "std_b": std_b,
        "more_certain": "A" if max_a <= max_b else "B",
        "difference": max_b - max_a,
    }


def format_ensemble_report(ensemble: MCEnsemble, x: List[float]) -> str:
    """Human-readable summary of ensemble state on input *x*."""
    mean_q, std_q = ensemble.mean_std(x)
    bonus = ensemble.uncertainty_bonus(x)
    intrinsic = ensemble.intrinsic_reward(x)
    conf = ensemble.confidence(x)
    action = ensemble.action_with_bonus(x)
    lines = [
        "MC-Dropout Ensemble Report",
        f"  Heads            : {ensemble.n_heads}",
        f"  Input dim        : {ensemble.input_dim}",
        f"  Output dim       : {ensemble.output_dim}",
        f"  Dropout rate     : {ensemble.dropout_rate}",
        "",
        f"  Mean Q           : {[f'{v:.6f}' for v in mean_q]}",
        f"  Std  Q           : {[f'{v:.6f}' for v in std_q]}",
        f"  Bonus (beta=1)   : {[f'{v:.6f}' for v in bonus]}",
        f"  Intrinsic reward : {intrinsic:.6f}",
        f"  Confidence       : {conf:.6f}",
        f"  Best action      : {action}",
    ]
    return "\n".join(lines)
