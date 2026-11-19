"""R3: Physics-informed loss term for FloPyArcade.

Adds a penalty term to the reward that enforces the policy's head
field to respect the governing transient groundwater-flow PDE:

    ∇·(K ∇h) − S_s ∂h/∂t = q

where
    K   — hydraulic conductivity (scalar or tensor),
    h   — hydraulic head field,
    S_s — specific storage coefficient,
    q   — source/sink term (well pumping rates interpolated to grid).

The loss is:

    L_phys = λ · mean( ‖ residual ‖² )

with residual computed via second-order central finite differences
on a structured 3D grid.

This module is **code-only**: it provides the computation and the
class API for integrating into training loops. It does **not**
execute any training. The human integrates ``PhysicsInformedLoss``
into their reward function or loss pipeline.

Usage::

    from flopyarcade.physics_loss import PhysicsInformedLoss, PhysicsLossConfig

    config = PhysicsLossConfig(
        lambda_weight=0.01,
        dx=10.0, dy=10.0, dz=5.0,
        dt=1.0,
        K=100.0,  # isotropic
        S_s=1e-4,
    )
    loss_fn = PhysicsInformedLoss(config)

    # h_t: shape (nx, ny, nz) — heads at time t
    # h_t1: shape (nx, ny, nz) — heads at time t+1
    # q: shape (nx, ny, nz) — source/sink terms
    residual = loss_fn.compute_residual(h_t, h_t1, q)
    loss = loss_fn.compute_loss(h_t, h_t1, q)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PhysicsLossConfig:
    """Configuration for the physics-informed loss.

    Attributes:
        lambda_weight: Scaling factor λ for the loss.  Typical range
            1e-4 to 1e-2 depending on reward scale.
        dx, dy, dz: Grid spacings in each dimension (model units).
        dt: Time step (days) for the ∂h/∂t term.
        K: Hydraulic conductivity. Scalar (isotropic) or a 3-tuple
            (Kxx, Kyy, Kzz) for anisotropic.
        S_s: Specific storage coefficient (1/length).
        boundary: Boundary condition type for finite differences.
            ``'neumann_zero'`` (zero-gradient / no-flow) or
            ``'periodic'``. Defaults to ``'neumann_zero'``.
    """
    lambda_weight: float = 0.01
    dx: float = 10.0
    dy: float = 10.0
    dz: float = 5.0
    dt: float = 1.0
    K: Union[float, tuple] = 100.0
    S_s: float = 1e-4
    boundary: str = "neumann_zero"

    def __post_init__(self) -> None:
        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ValueError("Grid spacings dx, dy, dz must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.lambda_weight < 0:
            raise ValueError("lambda_weight must be >= 0")
        if self.S_s < 0:
            raise ValueError("S_s must be >= 0")
        if isinstance(self.K, tuple):
            if len(self.K) != 3:
                raise ValueError("Anisotropic K must be a 3-tuple (Kxx, Kyy, Kzz)")
            for k in self.K:
                if k < 0:
                    raise ValueError("Conductivities must be >= 0")
        else:
            if self.K < 0:
                raise ValueError("K must be >= 0")
        if self.boundary not in ("neumann_zero", "periodic"):
            raise ValueError(f"boundary must be 'neumann_zero' or 'periodic', got '{self.boundary}'")


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def _laplacian(
    h: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    Kxx: float,
    Kyy: float,
    Kzz: float,
    boundary: str,
) -> np.ndarray:
    """Compute ∇·(K ∇h) on a 3D grid via central finite differences.

    For isotropic K: K * (∂²h/∂x² + ∂²h/∂y² + ∂²h/∂z²).
    For anisotropic: Kxx*∂²h/∂x² + Kyy*∂²h/∂y² + Kzz*∂²h/∂z².

    Boundary handling:
        - ``neumann_zero``: zero-gradient (∂h/∂n = 0) at boundaries,
          implemented by mirroring the adjacent interior value.
        - ``periodic``: wrap-around indexing.
    """
    nx, ny, nz = h.shape
    result = np.zeros_like(h, dtype=np.float64)

    for axis, spacing in enumerate((dx, dy, dz)):
        K_axis = (Kxx, Kyy, Kzz)[axis]
        n = (nx, ny, nz)[axis]
        if n < 3:
            # Not enough points for central difference; skip this axis
            continue

        # Build index slices for i-1, i, i+1 along `axis`
        # Using np.roll for periodic, or manual slicing for Neumann
        # Shift for all points (interior gets i±1, boundary gets wrapped)
        h_prev = np.roll(h, 1, axis=axis)
        h_next = np.roll(h, -1, axis=axis)

        if boundary != "periodic":
            # Neumann zero: override boundary ghost points
            # Zero-gradient: h_ghost = h_adjacent_interior
            if axis == 0:
                h_prev[0, :, :] = h[1, :, :]
                h_next[-1, :, :] = h[-2, :, :]
            elif axis == 1:
                h_prev[:, 0, :] = h[:, 1, :]
                h_next[:, -1, :] = h[:, -2, :]
            else:
                h_prev[:, :, 0] = h[:, :, 1]
                h_next[:, :, -1] = h[:, :, -2]

        d2h = (h_next - 2.0 * h + h_prev) / (spacing * spacing)
        result += K_axis * d2h

    return result


def compute_residual(
    h_t: np.ndarray,
    h_t1: np.ndarray,
    q: np.ndarray,
    config: PhysicsLossConfig,
) -> np.ndarray:
    """Compute the PDE residual on a 3D grid.

    residual = ∇·(K∇h) − S_s · ∂h/∂t − q

    Args:
        h_t: Head field at time t, shape (nx, ny, nz).
        h_t1: Head field at time t+dt, shape (nx, ny, nz).
        q: Source/sink term, shape (nx, ny, nz).
            Positive q = extraction (well pumping).
        config: PhysicsLossConfig with grid and parameter settings.

    Returns:
        Residual array, same shape as h_t.

    Raises:
        ValueError: If shapes mismatch or K is invalid.
    """
    if h_t.shape != h_t1.shape:
        raise ValueError(
            f"h_t shape {h_t.shape} != h_t1 shape {h_t1.shape}"
        )
    if h_t.shape != q.shape:
        raise ValueError(
            f"h_t shape {h_t.shape} != q shape {q.shape}"
        )
    if h_t.ndim != 3:
        raise ValueError(f"Expected 3D array, got {h_t.ndim}D")

    # Unpack K
    if isinstance(config.K, tuple):
        Kxx, Kyy, Kzz = config.K
    else:
        Kxx = Kyy = Kzz = config.K

    # Diffusion term: ∇·(K∇h) evaluated at time t (use h_t for spatial, average for accuracy)
    h_avg = 0.5 * (h_t + h_t1)
    diffusion = _laplacian(
        h_avg, config.dx, config.dy, config.dz,
        Kxx, Kyy, Kzz, config.boundary,
    )

    # Time derivative: S_s · ∂h/∂t
    dhdt = (h_t1 - h_t) / config.dt
    storage = config.S_s * dhdt

    # Residual
    residual = diffusion - storage - q
    return residual


# ---------------------------------------------------------------------------
# Loss class
# ---------------------------------------------------------------------------

class PhysicsInformedLoss:
    """Computes the physics-informed loss for a head field pair.

    The loss is:
        L = λ · mean(residual²)

    where residual = ∇·(K∇h) − S_s ∂h/∂t − q.

    Args:
        config: PhysicsLossConfig (or None for defaults).

    Example::

        loss_fn = PhysicsInformedLoss(PhysicsLossConfig(
            lambda_weight=0.01, dx=10, dy=10, dz=5, dt=1,
            K=100.0, S_s=1e-4,
        ))

        # Steady-state, no sources: residual should be ~0
        h = np.ones((10, 10, 5))
        loss = loss_fn.compute_loss(h, h, np.zeros((10, 10, 5)))
        assert loss == 0.0
    """

    def __init__(self, config: Optional[PhysicsLossConfig] = None) -> None:
        self.config = config or PhysicsLossConfig()

    def compute_residual(
        self,
        h_t: np.ndarray,
        h_t1: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        """Compute the PDE residual array."""
        return compute_residual(h_t, h_t1, q, self.config)

    def compute_loss(
        self,
        h_t: np.ndarray,
        h_t1: np.ndarray,
        q: np.ndarray,
    ) -> float:
        """Compute the scalar physics loss.

        Returns:
            λ · mean(residual²)
        """
        residual = self.compute_residual(h_t, h_t1, q)
        mse = float(np.mean(residual ** 2))
        return self.config.lambda_weight * mse

    def compute_loss_weighted(
        self,
        h_t: np.ndarray,
        h_t1: np.ndarray,
        q: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the physics loss with optional per-cell weights.

        Useful for emphasising the well vicinity or the capture zone.

        Args:
            weights: Optional array same shape as h_t. Cells with
                weight 0 are excluded from the loss.

        Returns:
            λ · weighted_mean(residual²)
        """
        residual = self.compute_residual(h_t, h_t1, q)
        sq = residual ** 2
        if weights is not None:
            w = weights.astype(np.float64)
            total_w = np.sum(w)
            if total_w == 0:
                return 0.0
            wmse = float(np.sum(w * sq) / total_w)
        else:
            wmse = float(np.mean(sq))
        return self.config.lambda_weight * wmse

    def penalty_for_reward(
        self,
        h_t: np.ndarray,
        h_t1: np.ndarray,
        q: np.ndarray,
        reward: float,
    ) -> float:
        """Compute the physics penalty to subtract from reward.

        Returns:
            reward − L_phys
        """
        loss = self.compute_loss(h_t, h_t1, q)
        return reward - loss

    def to_dict(self) -> dict:
        """Serialize config to dict."""
        c = self.config
        return {
            "lambda_weight": c.lambda_weight,
            "dx": c.dx, "dy": c.dy, "dz": c.dz,
            "dt": c.dt,
            "K": c.K,
            "S_s": c.S_s,
            "boundary": c.boundary,
        }

    def __repr__(self) -> str:
        c = self.config
        return (
            f"PhysicsInformedLoss(λ={c.lambda_weight}, "
            f"K={c.K}, S_s={c.S_s}, "
            f"grid=({c.dx},{c.dy},{c.dz}), "
            f"dt={c.dt}, bc={c.boundary})"
        )
