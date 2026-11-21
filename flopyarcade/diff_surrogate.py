"""R5: Differentiable MODFLOW surrogate.

Pure-numpy linear diffusion operator with gradient-based MPC.
Model: h_{t+1} = h_t + alpha * laplacian(h_t) + W @ q_wells
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class SurrogateConfig:
    grid_size: Tuple[int, int] = (8, 8)
    n_channels: int = 1
    learning_rate: float = 1e-3
    n_epochs: int = 50
    batch_size: int = 8
    dt: float = 0.1  # time step for diffusion
    well_influence_radius: int = 2


@dataclass
class MPCStepResult:
    steps: list = field(default_factory=list)
    q_sequence: list = field(default_factory=list)
    predicted_heads: list = field(default_factory=list)
    total_reward: float = 0.0


class HydraulicSurrogate:
    """Linearized hydraulic surrogate with analytical gradients.

    The forward model is:
        h_{t+1}[i,j] = h_t[i,j] + alpha * L(h_t)[i,j] + sum_k w[k, i,j] * q_wells[k]

    where L is the 5-point Laplacian and w is the learned well-influence matrix.
    """

    def __init__(self, config: SurrogateConfig | None = None):
        self.config = config or SurrogateConfig()
        rows, cols = self.config.grid_size
        n_cells = rows * cols
        self.alpha = 0.5  # diffusion coefficient (learnable)
        # Well influence: shape (n_wells_max, n_cells)
        # We support up to 8 wells by default
        self.n_wells_max = 8
        self.well_weights = np.zeros((self.n_wells_max, n_cells))
        self.n_wells_active = 4  # number of wells actually used
        self._train_loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Laplacian (5-point, zero-flux boundary)
    # ------------------------------------------------------------------
    def _laplacian(self, h: np.ndarray) -> np.ndarray:
        """5-point Laplacian with zero-flux (Neumann) boundaries."""
        lap = np.zeros_like(h)
        # interior
        lap[1:-1, 1:-1] = (
            h[2:, 1:-1] + h[:-2, 1:-1] + h[1:-1, 2:] + h[1:-1, :-2] - 4.0 * h[1:-1, 1:-1]
        )
        # boundaries (zero-flux: reflect)
        # top row
        lap[0, 1:-1] = h[1, 1:-1] + h[0, 1:-1] - 2.0 * h[0, 1:-1] + h[0, 2:] - h[0, :-2] * 0 + (h[0, :-2] - h[0, 2:]) * 0
        # Simpler: just use the interior formula with clamped indices
        lap = np.zeros_like(h)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                up = h[max(i - 1, 0), j]
                down = h[min(i + 1, h.shape[0] - 1), j]
                left = h[i, max(j - 1, 0)]
                right = h[i, min(j + 1, h.shape[1] - 1)]
                lap[i, j] = up + down + left + right - 4.0 * h[i, j]
        return lap

    # ------------------------------------------------------------------
    # Training data generation
    # ------------------------------------------------------------------
    def generate_training_data(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic (h_t, q_wells, h_t1) triples via diffusion."""
        rows, cols = self.config.grid_size
        dt = self.config.dt
        h_t_batch = np.zeros((n_samples, rows, cols))
        q_batch = np.zeros((n_samples, self.n_wells_active))
        h_t1_batch = np.zeros((n_samples, rows, cols))

        rng = np.random.default_rng(42)
        for i in range(n_samples):
            # Random initial head field (smooth-ish)
            h_t_batch[i] = rng.normal(1.0, 0.1, (rows, cols))
            # Random well rates
            q_batch[i] = rng.uniform(-0.5, 0.5, self.n_wells_active)
            # Forward step: h_t1 = h_t + alpha_true * lap(h_t) + W_true @ q
            # Use ground-truth alpha=1.0 and simple well placement
            lap = self._laplacian(h_t_batch[i])
            well_source = np.zeros((rows, cols))
            for k in range(self.n_wells_active):
                wr = (i + k) % rows
                wc = (i + k * 2) % cols
                well_source[wr, wc] += q_batch[i, k]
            h_t1_batch[i] = h_t_batch[i] + 1.0 * dt * lap + well_source

        return h_t_batch, q_batch, h_t1_batch

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def predict(self, h_t: np.ndarray, q_wells: np.ndarray) -> np.ndarray:
        """Apply learned operator: h_t1 = h_t + alpha*lap(h_t) + W@q."""
        lap = self._laplacian(h_t)
        well_source = self.well_weights[:len(q_wells), :].T @ q_wells
        well_source = well_source.reshape(self.config.grid_size)
        return h_t + self.alpha * self.config.dt * lap + well_source

    # ------------------------------------------------------------------
    # Training (gradient descent on alpha and well_weights)
    # ------------------------------------------------------------------
    def train(
        self,
        h_t: np.ndarray | None = None,
        q_wells: np.ndarray | None = None,
        h_t1: np.ndarray | None = None,
    ) -> list[float]:
        """Train via gradient descent. Returns loss history."""
        if h_t is None:
            h_t, q_wells, h_t1 = self.generate_training_data(100)

        lr = self.config.learning_rate
        n_epochs = self.config.n_epochs
        loss_history = []

        for epoch in range(n_epochs):
            # Compute predictions
            preds = np.zeros_like(h_t)
            for i in range(len(h_t)):
                preds[i] = self.predict(h_t[i], q_wells[i])

            # Loss: MSE
            diff = preds - h_t1
            loss = float(np.mean(diff ** 2))
            loss_history.append(loss)

            # Gradient descent (analytical for linear model)
            # dLoss/dalpha = 2 * mean(diff * dt * lap) per sample
            d_alpha = 0.0
            d_well = np.zeros_like(self.well_weights[:len(q_wells[0]), :])
            for i in range(len(h_t)):
                lap = self._laplacian(h_t[i])
                d_alpha += 2.0 * self.config.dt * np.mean(diff[i] * lap)
                # well gradient
                for k in range(len(q_wells[i])):
                    if q_wells[i, k] != 0:
                        d_well[k, :] += 2.0 * diff[i].flatten() * q_wells[i, k]

            n = len(h_t)
            self.alpha -= lr * d_alpha / n
            self.well_weights[:len(q_wells[0]), :] -= lr * d_well / n

        self._train_loss_history = loss_history
        return loss_history

    # ------------------------------------------------------------------
    # MPC backpropagation (chain rule through linear operator)
    # ------------------------------------------------------------------
    def backprop_mpc(
        self,
        h_t: np.ndarray,
        q_wells: np.ndarray,
        n_steps: int = 5,
        target_well_pos: Tuple[int, int] = (0, 0),
    ) -> Tuple[np.ndarray, float]:
        """Compute gradient of cumulative head-at-target w.r.t. q_wells.

        Returns (optimal_q_sequence, reward) where optimal_q_sequence is
        a (n_steps, n_wells) array of suggested rates.
        """
        rows, cols = self.config.grid_size
        n_wells = len(q_wells)
        target_idx = target_well_pos[0] * cols + target_well_pos[1]

        # Build the state transition matrix A (n_cells x n_cells):
        # h_{t+1} = A @ h_t.flatten() + w @ q
        # A = I + alpha * dt * L
        n_cells = rows * cols
        L = np.zeros((n_cells, n_cells))
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                up = (max(i - 1, 0)) * cols + j
                down = (min(i + 1, rows - 1)) * cols + j
                left = i * cols + max(j - 1, 0)
                right = i * cols + min(j + 1, cols - 1)
                L[idx, idx] = -4.0
                L[idx, up] += 1.0
                L[idx, down] += 1.0
                L[idx, left] += 1.0
                L[idx, right] += 1.0

        A = np.eye(n_cells) + self.alpha * self.config.dt * L

        # Well influence vector for each well
        W = self.well_weights[:n_wells, :].T  # (n_cells, n_wells)

        # Forward pass to get predicted heads and reward
        h_curr = h_t.flatten().astype(float)
        total_reward = 0.0
        # Accumulate d(reward)/d(q_k) for each step
        # reward = sum_t h[t][target]
        # h[t] = A^t @ h_0 + sum_{s=0}^{t-1} A^{t-1-s} @ W @ q[s]
        # d(h[t][target])/d(q[s]) = e_target^T @ A^{t-1-s} @ W

        d_reward_dq = np.zeros((n_steps, n_wells))

        # Precompute powers of A
        A_powers = [np.eye(n_cells)]
        for t in range(1, n_steps + 1):
            A_powers.append(A_powers[-1] @ A)

        e_target = np.zeros(n_cells)
        e_target[target_idx] = 1.0

        h_t_flat = h_t.flatten().astype(float)
        reward = 0.0
        for t in range(n_steps):
            # h[t] = A^t @ h_0 + sum_{s=0}^{t-1} A^{t-1-s} @ W @ q[s]
            h_t_vec = A_powers[t] @ h_t_flat
            for s in range(t):
                h_t_vec += A_powers[t - 1 - s] @ (W @ q_wells)
            reward += h_t_vec[target_idx]

            # Gradient: d(h[t][target])/d(q[s]) = e^T A^{t-1-s} W
            for s in range(t + 1):
                # contribution of q[s] to h[t]
                # h[t] depends on q[s] via A^{t-1-s} W q[s] (if t > s)
                if t > s:
                    contrib = e_target @ A_powers[t - 1 - s] @ W  # (n_wells,)
                    d_reward_dq[s] += contrib

        # Simple gradient ascent on q to maximize reward (keep bounded)
        optimal_q = np.zeros((n_steps, n_wells))
        for s in range(n_steps):
            optimal_q[s] = q_wells + 0.1 * d_reward_dq[s]
            optimal_q[s] = np.clip(optimal_q[s], -1.0, 1.0)

        return optimal_q, float(reward)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save_weights(self, path: str | Path) -> None:
        """Save learned parameters to JSON."""
        data = {
            "alpha": self.alpha,
            "well_weights": self.well_weights.tolist(),
            "n_wells_active": self.n_wells_active,
            "config": {
                "grid_size": list(self.config.grid_size),
                "n_channels": self.config.n_channels,
                "learning_rate": self.config.learning_rate,
                "n_epochs": self.config.n_epochs,
                "batch_size": self.config.batch_size,
                "dt": self.config.dt,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load_weights(cls, path: str | Path) -> "HydraulicSurrogate":
        """Load parameters from JSON into a new surrogate."""
        data = json.loads(Path(path).read_text())
        cfg = SurrogateConfig(
            grid_size=tuple(data["config"]["grid_size"]),
            n_channels=data["config"]["n_channels"],
            learning_rate=data["config"]["learning_rate"],
            n_epochs=data["config"]["n_epochs"],
            batch_size=data["config"]["batch_size"],
            dt=data["config"]["dt"],
        )
        surr = cls(cfg)
        surr.alpha = data["alpha"]
        surr.well_weights = np.array(data["well_weights"])
        surr.n_wells_active = data["n_wells_active"]
        return surr


# ----------------------------------------------------------------------
# MPC planning loop
# ----------------------------------------------------------------------
def run_mpc_planning(
    surrogate: HydraulicSurrogate,
    initial_head: np.ndarray,
    n_wells: int,
    n_steps: int = 10,
) -> MPCStepResult:
    """Run full MPC loop: predict, optimize q, roll out."""
    rows, cols = surrogate.config.grid_size
    target_well_pos = (rows // 2, cols // 2)
    q_init = np.zeros(n_wells)

    h_curr = initial_head.copy()
    result = MPCStepResult()

    for step in range(n_steps):
        # Backprop to get optimal q for this step
        q_opt, reward = surrogate.backprop_mpc(
            h_curr, q_init, n_steps=3, target_well_pos=target_well_pos
        )
        q_step = q_opt[0]  # take first step of the horizon

        # Forward predict
        h_next = surrogate.predict(h_curr, q_step)

        result.steps.append({"step": step, "q": q_step.tolist(), "reward": reward})
        result.q_sequence.append(q_step)
        result.predicted_heads.append(h_next)
        result.total_reward += h_next[target_well_pos[0], target_well_pos[1]]

        h_curr = h_next

    return result


# ----------------------------------------------------------------------
# Report formatting
# ----------------------------------------------------------------------
def format_surrogate_report(surrogate: HydraulicSurrogate, train_losses: list[float]) -> str:
    """Generate a human-readable training/surrogate report."""
    lines = [
        "=" * 60,
        "  Differentiable MODFLOW Surrogate Report",
        "=" * 60,
        f"  Grid size:         {surrogate.config.grid_size[0]}x{surrogate.config.grid_size[1]}",
        f"  Learned alpha:     {surrogate.alpha:.6f}",
        f"  Well weights norm: {np.linalg.norm(surrogate.well_weights):.6f}",
        f"  n_wells_active:    {surrogate.n_wells_active}",
        "-" * 60,
        f"  Training epochs:   {len(train_losses)}",
        f"  Initial loss:      {train_losses[0]:.6f}" if train_losses else "  Initial loss:      N/A",
        f"  Final loss:        {train_losses[-1]:.6f}" if train_losses else "  Final loss:        N/A",
        f"  Loss reduction:    {(1 - train_losses[-1]/train_losses[0])*100:.2f}%" if train_losses and train_losses[0] > 0 else "  Loss reduction:    N/A",
        "=" * 60,
    ]
    return "\n".join(lines)
