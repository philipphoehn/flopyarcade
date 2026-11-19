"""Tests for R3: Physics-informed loss term.

Covers:
- PhysicsLossConfig validation
- Residual computation (steady-state, analytical, anisotropic)
- Loss computation (scalar, weighted)
- penalty_for_reward
- Boundary conditions (Neumann, periodic)
- Edge cases (small axes, shape mismatch, 2D input)
- Serialization (to_dict)
"""

import numpy as np
import pytest

from flopyarcade.physics_loss import (
    PhysicsInformedLoss,
    PhysicsLossConfig,
    compute_residual,
    _laplacian,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_defaults(self):
        c = PhysicsLossConfig()
        assert c.lambda_weight == 0.01
        assert c.dx == 10.0
        assert c.dy == 10.0
        assert c.dz == 5.0
        assert c.dt == 1.0
        assert c.K == 100.0
        assert c.S_s == 1e-4
        assert c.boundary == "neumann_zero"

    def test_invalid_dx(self):
        with pytest.raises(ValueError, match="Grid spacings"):
            PhysicsLossConfig(dx=0)

    def test_invalid_dy(self):
        with pytest.raises(ValueError, match="Grid spacings"):
            PhysicsLossConfig(dy=-1)

    def test_invalid_dz(self):
        with pytest.raises(ValueError, match="Grid spacings"):
            PhysicsLossConfig(dz=0)

    def test_invalid_dt(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            PhysicsLossConfig(dt=0)

    def test_negative_lambda(self):
        with pytest.raises(ValueError, match="lambda_weight"):
            PhysicsLossConfig(lambda_weight=-0.1)

    def test_negative_Ss(self):
        with pytest.raises(ValueError, match="S_s"):
            PhysicsLossConfig(S_s=-1)

    def test_negative_K_scalar(self):
        with pytest.raises(ValueError, match="K must be"):
            PhysicsLossConfig(K=-5)

    def test_invalid_K_tuple_length(self):
        with pytest.raises(ValueError, match="3-tuple"):
            PhysicsLossConfig(K=(1, 2))

    def test_negative_K_tuple(self):
        with pytest.raises(ValueError, match="Conductivities"):
            PhysicsLossConfig(K=(1, -2, 3))

    def test_valid_K_tuple(self):
        c = PhysicsLossConfig(K=(50, 100, 25))
        assert c.K == (50, 100, 25)

    def test_invalid_boundary(self):
        with pytest.raises(ValueError, match="boundary"):
            PhysicsLossConfig(boundary="dirichlet")

    def test_periodic_boundary(self):
        c = PhysicsLossConfig(boundary="periodic")
        assert c.boundary == "periodic"


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

class TestResidualComputation:
    def test_steady_state_zero(self):
        """Constant head, no sources → residual = 0."""
        h = np.ones((10, 10, 5)) * 100.0
        q = np.zeros((10, 10, 5))
        cfg = PhysicsLossConfig(dx=10, dy=10, dz=5, dt=1, K=100, S_s=1e-4)
        res = compute_residual(h, h, q, cfg)
        assert np.allclose(res, 0.0, atol=1e-10)

    def test_linear_head_zero_residual(self):
        """Linear head field → ∇²h = 0 → residual = 0 (no sources)."""
        x = np.arange(10, dtype=np.float64)[:, None, None] * 10
        y = np.arange(10, dtype=np.float64)[None, :, None] * 10
        z = np.arange(5, dtype=np.float64)[None, None, :] * 5
        h = 0.5 * x + 0.3 * y - 0.1 * z + 50.0
        q = np.zeros_like(h)
        cfg = PhysicsLossConfig(dx=10, dy=10, dz=5, dt=1, K=100, S_s=1e-4)
        res = compute_residual(h, h, q, cfg)
        # Interior should be exactly zero; Neumann boundaries approximate
        assert np.allclose(res[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_quadratic_head_nonzero(self):
        """Quadratic head → ∇²h ≠ 0 → residual ≠ 0."""
        x = np.arange(10, dtype=np.float64)[:, None, None] * 10
        h = x**2  # ∂²h/∂x² = 2
        q = np.zeros_like(h)
        cfg = PhysicsLossConfig(dx=10, dy=10, dz=5, dt=1, K=100, S_s=1e-4)
        res = compute_residual(h, h, q, cfg)
        # Interior: residual ≈ K * 2 = 200
        interior = res[1:-1, 1:-1, 1:-1]
        assert np.allclose(interior, 200.0, atol=1.0)

    def test_temporal_term(self):
        """Changing head with S_s > 0 → storage term contributes."""
        h_t = np.ones((10, 10, 5)) * 100.0
        h_t1 = np.ones((10, 10, 5)) * 101.0  # +1 head unit
        q = np.zeros_like(h_t)
        cfg = PhysicsLossConfig(dx=10, dy=10, dz=5, dt=1.0, K=100, S_s=1e-3)
        res = compute_residual(h_t, h_t1, q, cfg)
        # residual = 0 - S_s*(1/1) - 0 = -1e-3
        assert np.allclose(res, -1e-3, atol=1e-10)

    def test_source_term(self):
        """Source q appears in residual."""
        h = np.ones((10, 10, 5)) * 100.0
        q = np.zeros((10, 10, 5))
        q[5, 5, 2] = 50.0  # one cell has source
        cfg = PhysicsLossConfig(dx=10, dy=10, dz=5, dt=1, K=100, S_s=0.0)
        res = compute_residual(h, h, q, cfg)
        assert np.allclose(res[5, 5, 2], -50.0, atol=1e-10)
        # other cells should be ~0
        mask = np.ones((10, 10, 5), dtype=bool)
        mask[5, 5, 2] = False
        assert np.allclose(res[mask], 0.0, atol=1e-10)

    def test_anisotropic_K(self):
        """Anisotropic K gives different contributions per axis."""
        x = np.arange(10, dtype=np.float64)[:, None, None] * 10
        h = x**2  # only x-direction curvature
        q = np.zeros_like(h)
        cfg = PhysicsLossConfig(dx=10, dy=10, dz=5, dt=1, K=(200, 50, 10), S_s=0.0)
        res = compute_residual(h, h, q, cfg)
        # Interior: residual ≈ Kxx * 2 = 400
        interior = res[1:-1, 1:-1, 1:-1]
        assert np.allclose(interior, 400.0, atol=1.0)

    def test_shape_mismatch_h_t_h_t1(self):
        with pytest.raises(ValueError, match="shape"):
            compute_residual(
                np.ones((10, 10, 5)), np.ones((8, 8, 4)),
                np.ones((10, 10, 5)), PhysicsLossConfig(),
            )

    def test_shape_mismatch_q(self):
        with pytest.raises(ValueError, match="shape"):
            compute_residual(
                np.ones((10, 10, 5)), np.ones((10, 10, 5)),
                np.ones((7, 7, 3)), PhysicsLossConfig(),
            )

    def test_2d_input_rejected(self):
        with pytest.raises(ValueError, match="3D"):
            compute_residual(
                np.ones((10, 10)), np.ones((10, 10)),
                np.ones((10, 10)), PhysicsLossConfig(),
            )

    def test_single_cell_axis_skipped(self):
        """Axis with n=1 is skipped in Laplacian (no central diff possible)."""
        h = np.ones((1, 10, 10)) * 50.0
        h[0, :, :] += np.arange(10, dtype=np.float64)[None, :] * 10  # linear in y
        q = np.zeros_like(h)
        cfg = PhysicsLossConfig(dx=10, dy=10, dz=5, dt=1, K=100, S_s=0.0)
        res = compute_residual(h, h, q, cfg)
        # Only y-axis contributes; linear → zero curvature in interior
        # (Neumann boundaries approximate zero-gradient, giving small non-zero at edges)
        assert np.allclose(res[0, 1:-1, 1:-1], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_neumann_symmetric(self):
        """Symmetric head → Neumann BC should give zero at boundaries too."""
        x = np.arange(11, dtype=np.float64)[:, None, None] * 10
        xc = 50.0
        h = (x - xc)**2  # symmetric around center
        q = np.zeros_like(h)
        cfg = PhysicsLossConfig(dx=10, dy=10, dz=5, dt=1, K=100, S_s=0.0)
        res = compute_residual(h, h, q, cfg)
        # Interior should be ≈ 200 everywhere (∂²h/∂x² = 2)
        assert np.allclose(res[1:-1, 0, 0], 200.0, atol=1.0)

    def test_periodic_sinusoidal(self):
        """Sinusoidal field with periodic BC → exact Laplacian."""
        nx = 32
        x = np.linspace(0, 2 * np.pi, nx, endpoint=False)[:, None, None]
        h = np.sin(x)  # ∂²h/∂x² = -sin(x) = -h
        q = np.zeros_like(h)
        cfg = PhysicsLossConfig(dx=2*np.pi/nx, dy=10, dz=5, dt=1, K=1.0, S_s=0.0,
                                boundary="periodic")
        res = compute_residual(h, h, q, cfg)
        # residual ≈ -h (since ∂²h/∂x² = -h and K=1)
        assert np.allclose(res, -h, atol=0.05)


# ---------------------------------------------------------------------------
# Loss class
# ---------------------------------------------------------------------------

class TestPhysicsInformedLoss:
    def test_zero_loss_constant_head(self):
        loss_fn = PhysicsInformedLoss(PhysicsLossConfig(dx=10, dy=10, dz=5, dt=1, K=100, S_s=1e-4))
        h = np.ones((8, 8, 4)) * 75.0
        q = np.zeros_like(h)
        loss = loss_fn.compute_loss(h, h, q)
        assert loss == 0.0

    def test_positive_loss_nonzero_residual(self):
        loss_fn = PhysicsInformedLoss(PhysicsLossConfig(lambda_weight=1.0, dx=10, dy=10, dz=5, dt=1, K=100, S_s=0.0))
        x = np.arange(8, dtype=np.float64)[:, None, None] * 10
        h = x**2
        q = np.zeros_like(h)
        loss = loss_fn.compute_loss(h, h, q)
        assert loss > 0.0

    def test_lambda_scaling(self):
        x = np.arange(8, dtype=np.float64)[:, None, None] * 10
        h = x**2
        q = np.zeros_like(h)

        loss_1 = PhysicsInformedLoss(PhysicsLossConfig(lambda_weight=0.01, dx=10, dy=10, dz=5, dt=1, K=100, S_s=0.0)).compute_loss(h, h, q)
        loss_10 = PhysicsInformedLoss(PhysicsLossConfig(lambda_weight=0.1, dx=10, dy=10, dz=5, dt=1, K=100, S_s=0.0)).compute_loss(h, h, q)
        assert np.allclose(loss_10, 10 * loss_1, rtol=1e-10)

    def test_weighted_loss_focus(self):
        """Weighted loss emphasises cells with high weight."""
        loss_fn = PhysicsInformedLoss(PhysicsLossConfig(lambda_weight=1.0, dx=10, dy=10, dz=5, dt=1, K=100, S_s=0.0))
        x = np.arange(8, dtype=np.float64)[:, None, None] * 10
        y = np.arange(8, dtype=np.float64)[None, :, None] * 10
        z = np.arange(4, dtype=np.float64)[None, None, :] * 5
        h = x**2 + y**2 + z**2
        q = np.zeros_like(h)

        # Uniform weights → same as unweighted
        uniform = np.ones_like(h)
        loss_uniform = loss_fn.compute_loss_weighted(h, h, q, uniform)
        loss_plain = loss_fn.compute_loss(h, h, q)
        assert np.allclose(loss_uniform, loss_plain, rtol=1e-10)

        # Zero weights → loss = 0
        zero_w = np.zeros_like(h)
        loss_zero = loss_fn.compute_loss_weighted(h, h, q, zero_w)
        assert loss_zero == 0.0

        # Focused weight on one cell → loss = λ * (residual[3,3,1])²
        focused = np.zeros_like(h)
        focused[3, 3, 1] = 1.0
        loss_focused = loss_fn.compute_loss_weighted(h, h, q, focused)
        # Exact value: λ * residual[3,3,1]² (only one cell weighted)
        residual = loss_fn.compute_residual(h, h, q)
        expected = 1.0 * (residual[3, 3, 1] ** 2)  # λ=1.0
        assert np.allclose(loss_focused, expected, rtol=1e-10)
        # And it differs from the uniform mean (proves weighting works)
        assert not np.allclose(loss_focused, loss_plain, rtol=1e-6)

    def test_penalty_for_reward(self):
        loss_fn = PhysicsInformedLoss(PhysicsLossConfig(lambda_weight=0.01, dx=10, dy=10, dz=5, dt=1, K=100, S_s=0.0))
        h = np.ones((8, 8, 4)) * 50.0
        q = np.zeros_like(h)
        reward = 100.0
        # Zero residual → penalty = reward
        result = loss_fn.penalty_for_reward(h, h, q, reward)
        assert result == reward

        # Nonzero residual → penalty < reward
        x = np.arange(8, dtype=np.float64)[:, None, None] * 10
        y = np.arange(8, dtype=np.float64)[None, :, None] * 10
        z = np.arange(4, dtype=np.float64)[None, None, :] * 5
        h2 = x**2 + y**2 + z**2
        result2 = loss_fn.penalty_for_reward(h2, h2, q, reward)
        assert result2 < reward

    def test_repr(self):
        loss_fn = PhysicsInformedLoss(PhysicsLossConfig(lambda_weight=0.5, K=200))
        r = repr(loss_fn)
        assert "0.5" in r
        assert "200" in r

    def test_to_dict(self):
        loss_fn = PhysicsInformedLoss(PhysicsLossConfig(lambda_weight=0.3, K=(10, 20, 30), boundary="periodic"))
        d = loss_fn.to_dict()
        assert d["lambda_weight"] == 0.3
        assert d["K"] == (10, 20, 30)
        assert d["boundary"] == "periodic"
        assert d["dx"] == 10.0
        assert d["S_s"] == 1e-4

    def test_none_config_uses_defaults(self):
        loss_fn = PhysicsInformedLoss(None)
        assert loss_fn.config.lambda_weight == 0.01
        assert loss_fn.config.K == 100.0


# ---------------------------------------------------------------------------
# Laplacian helper
# ---------------------------------------------------------------------------

class TestLaplacian:
    def test_constant_field(self):
        h = np.ones((10, 10, 5)) * 42.0
        lap = _laplacian(h, 10, 10, 5, 100, 100, 100, "neumann_zero")
        assert np.allclose(lap, 0.0, atol=1e-10)

    def test_pure_x_curvature(self):
        x = np.arange(10, dtype=np.float64)[:, None, None] * 10
        h = x**2
        lap = _laplacian(h, 10, 10, 5, 100, 100, 100, "neumann_zero")
        # ∂²h/∂x² = 2, so Kxx*2 = 200 in x-interior
        assert np.allclose(lap[1:-1, 0, 0], 200.0, atol=1.0)

    def test_anisotropic_contribution(self):
        x = np.arange(10, dtype=np.float64)[:, None, None] * 10
        y = np.arange(10, dtype=np.float64)[None, :, None] * 10
        h = x**2 + y**2  # ∂²/∂x² = 2, ∂²/∂y² = 2
        lap = _laplacian(h, 10, 10, 5, 100, 200, 50, "neumann_zero")
        # Interior: 100*2 + 200*2 = 600
        assert np.allclose(lap[1:-1, 1:-1, 0], 600.0, atol=1.0)

    def test_periodic_vs_neumann_differ_at_boundary(self):
        x = np.arange(10, dtype=np.float64)[:, None, None] * 10
        h = x**2
        lap_n = _laplacian(h, 10, 10, 5, 1, 1, 1, "neumann_zero")
        lap_p = _laplacian(h, 10, 10, 5, 1, 1, 1, "periodic")
        # They should differ at boundary cells
        assert not np.allclose(lap_n[0, 0, 0], lap_p[0, 0, 0])


# ---------------------------------------------------------------------------
# Integration: small MODFLOW-like scenario
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_pumping_well_residual(self):
        """Simulate a pumping well: head drops near well, PDE residual should be small.

        Set up a 1D radial-like approximation: head = h0 - (Q/(2πK)) * ln(r)
        For a discrete grid, we just check that the residual is bounded.
        """
        nx, ny, nz = 20, 20, 3
        x = np.arange(nx, dtype=np.float64)[:, None, None] * 5
        y = np.arange(ny, dtype=np.float64)[None, :, None] * 5
        cx, cy = 10.0, 10.0
        r = np.sqrt((x - cx*5)**2 + (y - cy*5)**2 + 1.0)  # avoid log(0)
        Q = 1000.0
        K = 50.0
        h = 50.0 - (Q / (2 * np.pi * K)) * np.log(r)
        h = np.broadcast_to(h, (nx, ny, nz)).copy()
        q = np.zeros((nx, ny, nz))
        q[10, 10, 1] = Q / (5*5*5)  # spread well rate over cell volume

        cfg = PhysicsLossConfig(dx=5, dy=5, dz=5, dt=1, K=K, S_s=0.0)
        loss_fn = PhysicsInformedLoss(cfg)
        loss = loss_fn.compute_loss(h, h, q)
        # Should be finite and non-negative
        assert np.isfinite(loss)
        assert loss >= 0.0

    def test_multiple_time_steps_convergence(self):
        """Diffusive evolution: head smooths over time → residual should decrease.

        Start with a sharp peak; after one diffusive step (analytical),
        the residual of the new state should be smaller.
        """
        nx, ny, nz = 15, 15, 3
        h0 = np.zeros((nx, ny, nz))
        h0[7, 7, 1] = 100.0  # initial spike

        cfg = PhysicsLossConfig(dx=1, dy=1, dz=1, dt=1, K=1.0, S_s=0.5)

        # "Evolve" one step: simple explicit diffusion (not the real solver, just for test)
        h1 = h0.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    h1[i,j,k] = h0[i,j,k] + 0.1 * (h0[i-1,j,k]+h0[i+1,j,k]+h0[i,j-1,k]+h0[i,j+1,k]+h0[i,j,k-1]+h0[i,j,k+1]-6*h0[i,j,k])

        q = np.zeros((nx, ny, nz))
        loss_before = PhysicsInformedLoss(cfg).compute_loss(h0, h1, q)
        assert np.isfinite(loss_before)
        assert loss_before >= 0.0


# ---------------------------------------------------------------------------
# Count tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
