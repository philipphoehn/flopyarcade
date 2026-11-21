"""Tests for reactive_transport (S1 - 7r-d)."""
import math
import os
import tempfile

import pytest
import numpy as np

from flopyarcade.reactive_transport import (
    ReactiveParams,
    ReactiveTransportEnv,
    SpeciesState,
    compare_reactive,
    format_reactive_report,
    kinetic_curve,
    multi_step_simulation,
)


# -- SpeciesState ---------------------------------------------------------------

class TestSpeciesState:
    def test_creation(self):
        s = SpeciesState(name="C", concentration=1.0, parent=None, decay_constant=0.001)
        assert s.name == "C"
        assert s.concentration == 1.0
        assert s.parent is None
        assert s.decay_constant == 0.001

    def test_defaults(self):
        s = SpeciesState(name="N", concentration=0.5)
        assert s.parent is None
        assert s.decay_constant == 0.0


# -- ReactiveParams -------------------------------------------------------------

class TestReactiveParams:
    def test_defaults(self):
        p = ReactiveParams()
        assert p.parent_name == "C"
        assert p.daughter_name == "N"
        assert p.decay_lambda == 0.001
        assert p.initial_parent == 1.0
        assert p.initial_daughter == 0.0

    def test_half_life_steps(self):
        p = ReactiveParams(decay_lambda=0.001)
        expected = math.log(2) / 0.001
        assert p.half_life_steps == pytest.approx(expected, rel=1e-12)

    def test_custom_params(self):
        p = ReactiveParams(parent_name="U", daughter_name="Th", decay_lambda=0.01)
        assert p.parent_name == "U"
        assert p.daughter_name == "Th"
        assert p.half_life_steps == pytest.approx(math.log(2) / 0.01)


# -- ReactiveTransportEnv init ---------------------------------------------------

class TestEnvInit:
    def test_init(self):
        p = ReactiveParams()
        env = ReactiveTransportEnv(p, grid_shape=(5, 5))
        assert env.grid_shape == (5, 5)
        assert env.n_steps() == 0
        assert env.total_concentration() == pytest.approx(25.0)  # 25 cells * 1.0

    def test_custom_grid(self):
        p = ReactiveParams(initial_parent=2.0)
        env = ReactiveTransportEnv(p, grid_shape=(3, 4))
        assert env.total_concentration() == pytest.approx(24.0)  # 12 * 2.0

    def test_parent_field_shape(self):
        env = ReactiveTransportEnv(ReactiveParams())
        field = env.parent_field()
        assert len(field) == 5
        assert all(len(row) == 5 for row in field)

    def test_daughter_field_initial_zero(self):
        env = ReactiveTransportEnv(ReactiveParams())
        field = env.daughter_field()
        assert all(v == 0.0 for row in field for v in row)


# -- step -----------------------------------------------------------------------

class TestStep:
    def test_step_updates_both_fields(self):
        p = ReactiveParams(decay_lambda=0.01, initial_parent=1.0, initial_daughter=0.0)
        env = ReactiveTransportEnv(p, grid_shape=(1, 1))
        result = env.step()
        assert result["t"] == 1
        assert result["parent_total"] == pytest.approx(1.0 * 0.99)
        assert result["daughter_total"] == pytest.approx(1.0 * 0.01)

    def test_mass_conservation(self):
        p = ReactiveParams(decay_lambda=0.01)
        env = ReactiveTransportEnv(p, grid_shape=(5, 5))
        initial = env.total_concentration()
        for _ in range(50):
            env.step()
        assert env.total_concentration() == pytest.approx(initial, rel=1e-6)

    def test_n_steps_increments(self):
        env = ReactiveTransportEnv(ReactiveParams())
        assert env.n_steps() == 0
        env.step()
        assert env.n_steps() == 1
        env.step()
        assert env.n_steps() == 2


# -- metrics --------------------------------------------------------------------

class TestMetrics:
    def test_total_concentration_positive(self):
        env = ReactiveTransportEnv(ReactiveParams())
        assert env.total_concentration() > 0

    def test_mass_balance_error_small(self):
        env = ReactiveTransportEnv(ReactiveParams(decay_lambda=0.001))
        for _ in range(100):
            env.step()
        assert env.mass_balance_error() < 1e-6

    def test_daughter_fraction_increases(self):
        env = ReactiveTransportEnv(ReactiveParams(decay_lambda=0.01))
        d0 = env.daughter_fraction()
        for _ in range(20):
            env.step()
        d1 = env.daughter_fraction()
        assert d1 > d0

    def test_daughter_fraction_lt_one(self):
        p = ReactiveParams(decay_lambda=0.01)
        env = ReactiveTransportEnv(p, grid_shape=(3, 3))
        for _ in range(500):
            env.step()
        assert env.daughter_fraction() < 1.0

    def test_activity_ratio_positive(self):
        env = ReactiveTransportEnv(ReactiveParams(decay_lambda=0.01))
        env.step()
        assert env.activity_ratio() > 0

    def test_half_life_reached(self):
        p = ReactiveParams(decay_lambda=0.01)
        env = ReactiveTransportEnv(p, grid_shape=(1, 1))
        assert not env.half_life_reached()
        # ln(2)/0.01 ~ 69 steps; use 100 to be safe
        for _ in range(100):
            env.step()
        assert env.half_life_reached()

    def test_equilibrium(self):
        p = ReactiveParams(decay_lambda=0.01)
        env = ReactiveTransportEnv(p, grid_shape=(1, 1))
        # after many steps parent ~ daughter is not reached for first-order
        # but with enough steps the fraction approaches 1, so |p-d| shrinks
        for _ in range(2000):
            env.step()
        # parent is very small, daughter ~ 1; |p-d| is large relative to total
        # so equilibrium should be False for pure first-order (no reverse reaction)
        # but the method just checks |parent - daughter| < tol * total
        # parent ~ exp(-0.01*2000) ~ 4.5e-9, daughter ~ 1 - 4.5e-9
        # |p - d| ~ 1, total ~ 1, so 1 < 0.01*1 is False
        assert not env.equilibrium(tol=0.01)

    def test_capture_efficiency_range(self):
        env = ReactiveTransportEnv(ReactiveParams(decay_lambda=0.01))
        for _ in range(10):
            env.step()
        zone = [(0, 0), (1, 1)]
        eff = env.capture_efficiency(zone)
        assert 0.0 <= eff <= 1.0

    def test_capture_efficiency_zero_daughter(self):
        env = ReactiveTransportEnv(ReactiveParams(initial_daughter=0.0))
        # no steps taken, daughter is all zero
        assert env.capture_efficiency([(0, 0)]) == 0.0


# -- reset ----------------------------------------------------------------------

class TestReset:
    def test_reset_restores_initial(self):
        p = ReactiveParams(decay_lambda=0.01, initial_parent=1.0)
        env = ReactiveTransportEnv(p, grid_shape=(3, 3))
        for _ in range(20):
            env.step()
        assert env.n_steps() == 20
        assert env.daughter_fraction() > 0
        env.reset()
        assert env.n_steps() == 0
        assert env.daughter_fraction() == 0.0
        assert env.total_concentration() == pytest.approx(9.0)


# -- serialisation ---------------------------------------------------------------

class TestSerialisation:
    def test_save_load_roundtrip(self):
        p = ReactiveParams(decay_lambda=0.01)
        env = ReactiveTransportEnv(p, grid_shape=(4, 4))
        for _ in range(10):
            env.step()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            env.save_state(path)
            env2 = ReactiveTransportEnv.load_state(path)
            assert env2.n_steps() == 10
            assert env2.total_concentration() == pytest.approx(env.total_concentration())
            assert env2.daughter_fraction() == pytest.approx(env.daughter_fraction())
            np.testing.assert_allclose(env2._parent, env._parent)
            np.testing.assert_allclose(env2._daughter, env._daughter)
        finally:
            os.unlink(path)

    def test_export_field(self):
        env = ReactiveTransportEnv(ReactiveParams(decay_lambda=0.01))
        env.step()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            env.export_field(path)
            data = np.load(path)
            assert "parent" in data
            assert "daughter" in data
            assert data["parent"].shape == (5, 5)
        finally:
            os.unlink(path)


# -- module helpers ---------------------------------------------------------------

class TestModuleHelpers:
    def test_multi_step_simulation_length(self):
        p = ReactiveParams(decay_lambda=0.01)
        records = multi_step_simulation(p, n_steps=50)
        assert len(records) == 50
        assert records[-1]["t"] == 50

    def test_kinetic_curve_monotonic_daughter(self):
        p = ReactiveParams(decay_lambda=0.01)
        curve = kinetic_curve(p, n_points=100)
        assert len(curve) == 101  # n_points + 1
        daughters = [d for _, _, d in curve]
        for i in range(1, len(daughters)):
            assert daughters[i] >= daughters[i - 1], f"daughter not monotonic at index {i}"

    def test_compare_reactive_structure(self):
        p1 = ReactiveParams(decay_lambda=0.01)
        p2 = ReactiveParams(decay_lambda=0.05)
        env_a = ReactiveTransportEnv(p1, grid_shape=(2, 2))
        env_b = ReactiveTransportEnv(p2, grid_shape=(2, 2))
        for _ in range(20):
            env_a.step()
            env_b.step()
        result = compare_reactive(env_a, env_b)
        assert "a_daughter_frac" in result
        assert "b_daughter_frac" in result
        assert "a_mass_err" in result
        assert "b_mass_err" in result
        assert "faster_decay" in result
        assert isinstance(result["faster_decay"], bool)
        # env_b has higher lambda, so should have more daughter
        assert result["faster_decay"] is False  # a_df > b_df is False

    def test_format_reactive_report(self):
        env = ReactiveTransportEnv(ReactiveParams(decay_lambda=0.01))
        for _ in range(10):
            env.step()
        report = format_reactive_report(env)
        assert "Reactive Transport Report" in report
        assert "Time step" in report
        assert "Mass balance" in report
        assert "Half-life" in report

    def test_1000_steps_mass_conservation(self):
        p = ReactiveParams(decay_lambda=0.001)
        env = ReactiveTransportEnv(p, grid_shape=(3, 3))
        initial = env.total_concentration()
        for _ in range(1000):
            env.step()
        err = abs(env.total_concentration() - initial) / initial
        assert err < 1e-4
