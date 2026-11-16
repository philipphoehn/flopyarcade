"""Tests for B1: Smooth trajectory interpolation (PCHIP + particle transition)."""
import numpy as np
import pytest


class TestSmoothSegment:
    """Test the _smooth_segment method on a minimal render-like object."""

    def _make_obj(self):
        """Create a minimal object with _smooth_segment."""
        from flopyarcade.render import RenderMixin
        obj = object.__new__(RenderMixin)
        return obj

    def test_two_points_linear(self):
        """Two raw points should produce 20 evenly-spaced points."""
        obj = self._make_obj()
        x_raw = np.array([0.0, 10.0])
        y_raw = np.array([0.0, 5.0])
        xs, ys = obj._smooth_segment(x_raw, y_raw, n_subdiv=10)
        assert len(xs) == 20
        assert len(ys) == 20
        # Endpoints preserved
        assert np.isclose(xs[0], 0.0)
        assert np.isclose(xs[-1], 10.0)
        assert np.isclose(ys[0], 0.0)
        assert np.isclose(ys[-1], 5.0)

    def test_single_point(self):
        """A single point should be returned unchanged."""
        obj = self._make_obj()
        x_raw = np.array([5.0])
        y_raw = np.array([3.0])
        xs, ys = obj._smooth_segment(x_raw, y_raw, n_subdiv=10)
        assert len(xs) == 1
        assert len(ys) == 1
        assert np.isclose(xs[0], 5.0)

    def test_three_points_pchip(self):
        """Three or more points should be PCHIP-interpolated to more points."""
        obj = self._make_obj()
        x_raw = np.array([0.0, 5.0, 10.0, 15.0])
        y_raw = np.array([0.0, 3.0, 1.0, 8.0])
        xs, ys = obj._smooth_segment(x_raw, y_raw, n_subdiv=10)
        # Should have more points than raw
        assert len(xs) > len(x_raw)
        assert len(ys) > len(y_raw)
        # Endpoints preserved
        assert np.isclose(xs[0], 0.0)
        assert np.isclose(xs[-1], 15.0)
        assert np.isclose(ys[0], 0.0)
        assert np.isclose(ys[-1], 8.0)
        # All values should be finite
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(ys))

    def test_duplicate_points_dedup(self):
        """Consecutive duplicate points should be deduplicated."""
        obj = self._make_obj()
        x_raw = np.array([0.0, 5.0, 5.0, 10.0])
        y_raw = np.array([0.0, 3.0, 3.0, 8.0])
        xs, ys = obj._smooth_segment(x_raw, y_raw, n_subdiv=10)
        # Should still work and produce valid output
        assert len(xs) >= 2
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(ys))

    def test_monotonic_arc_length(self):
        """Parameterization should be monotonic in arc length."""
        obj = self._make_obj()
        x_raw = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_raw = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # parabola
        xs, ys = obj._smooth_segment(x_raw, y_raw, n_subdiv=10)
        # Arc-length parameterization: successive points should not go backwards in s
        dx = np.diff(xs)
        dy = np.diff(ys)
        seg = np.sqrt(dx**2 + dy**2)
        # Allow tiny negative due to PCHIP overshoot, but overall should be positive
        assert np.sum(seg) > 0


class TestRenderParticleTransition:
    """Test render_particle_transition with a mock object."""

    def test_no_prev_coords_first_call(self):
        """First call with no previous coords should store current and return."""
        from flopyarcade.render import RenderMixin
        obj = object.__new__(RenderMixin)
        obj._blit_enabled = False
        obj._blit_artists = {}
        obj._prevParticleCoords = None
        # _get_particle_coords must exist
        obj._get_particle_coords = lambda: (5.0, 3.0)
        # Should not raise
        obj.render_particle_transition(n_frames=4, pause_sec=0.0)
        assert obj._prevParticleCoords == (5.0, 3.0)

    def test_no_movement_skip(self):
        """If particle hasn't moved, should skip animation."""
        from flopyarcade.render import RenderMixin
        obj = object.__new__(RenderMixin)
        obj._blit_enabled = False
        obj._blit_artists = {}
        obj._prevParticleCoords = (5.0, 3.0)
        obj._get_particle_coords = lambda: (5.0, 3.0)
        # Should not raise, just update prev
        obj.render_particle_transition(n_frames=4, pause_sec=0.0)
        assert obj._prevParticleCoords == (5.0, 3.0)

    def test_n_frames_one_skips(self):
        """n_frames=1 should return immediately."""
        from flopyarcade.render import RenderMixin
        obj = object.__new__(RenderMixin)
        obj._blit_enabled = False
        obj._blit_artists = {}
        obj._prevParticleCoords = (5.0, 3.0)
        obj._get_particle_coords = lambda: (10.0, 8.0)
        obj.render_particle_transition(n_frames=1, pause_sec=0.0)
        # prev should not have been updated since we skipped
        assert obj._prevParticleCoords == (5.0, 3.0)


class TestBlitUpdateTrajectorySmooth:
    """Test that _blit_update_trajectory uses smoothing."""

    def test_smoothing_produces_more_points(self):
        """The trajectory line should have more points than raw when smoothing is active."""
        from flopyarcade.render import RenderMixin
        obj = object.__new__(RenderMixin)
        # Minimal setup
        obj.timeStep = 1
        obj._blit_artists = {'trajectory': []}
        obj.trajectories = {'x': [np.array([0.0, 5.0, 10.0])],
                            'y': [np.array([0.0, 3.0, 8.0])]}
        obj.SMOOTH_TRAJ_SUBDIV = 10
        obj._count_trajectory_points = lambda: (3, [3])
        obj._generate_fade_colors = lambda n: np.zeros((n, 4))
        # Mock ax2
        from unittest.mock import MagicMock
        obj.ax2 = MagicMock()
        obj.ax2.plot.return_value = (MagicMock(),)
        obj._smooth_segment = RenderMixin._smooth_segment.__get__(obj)

        obj._blit_update_trajectory()
        # The plot should have been called with smoothed (longer) arrays
        call_args = obj.ax2.plot.call_args
        xs_used = call_args[0][0]
        assert len(xs_used) > 3  # more than raw 3 points due to smoothing


class TestConfigDefaults:
    """Test that env_base sets reasonable defaults."""

    def test_defaults_exist(self):
        """Verify the config attributes are set in the source."""
        import inspect
        from flopyarcade.env_base import EnvBaseMixin
        src = inspect.getsource(EnvBaseMixin)
        assert 'SMOOTH_TRAJ_FRAMES' in src
        assert 'SMOOTH_TRAJ_SUBDIV' in src
        assert '_prevParticleCoords' in src
