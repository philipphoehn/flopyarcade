"""Tests for flopyarcade.prognostic_particle (C14: prognostic auxiliary particle)."""

import numpy as np
import pytest

from flopyarcade.prognostic_particle import PrognosticParticle, PrognosticResult


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestPrognosticResult:
    def test_defaults(self):
        r = PrognosticResult()
        assert r.positions == []
        assert r.endpoint == (0.0, 0.0, 0.0)
        assert r.will_capture is False
        assert r.steps == 0
        assert r.horizon_steps == 0
        assert r.confidence == 0.0


# ---------------------------------------------------------------------------
# PrognosticParticle init
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        p = PrognosticParticle()
        assert p.horizon_steps == 10
        assert p.modpath_runner is None
        assert p.capture_radius == 5.0
        assert p.well_coords is None

    def test_custom_params(self):
        p = PrognosticParticle(
            horizon_steps=5,
            capture_radius=3.0,
            well_coords=(1.0, 2.0, 3.0),
        )
        assert p.horizon_steps == 5
        assert p.capture_radius == 3.0
        assert p.well_coords == (1.0, 2.0, 3.0)

    def test_horizon_min_1(self):
        p = PrognosticParticle(horizon_steps=0)
        assert p.horizon_steps == 1


# ---------------------------------------------------------------------------
# effective_well property
# ---------------------------------------------------------------------------

class TestEffectiveWell:
    def test_explicit_well(self):
        p = PrognosticParticle(well_coords=(1, 2, 3))
        assert p.effective_well == (1, 2, 3)

    def test_runner_fallback(self):
        class FakeRunner:
            wellCoords = [4.0, 5.0, 6.0]
        p = PrognosticParticle(modpath_runner=FakeRunner())
        assert p.effective_well == (4.0, 5.0, 6.0)

    def test_no_well(self):
        p = PrognosticParticle()
        assert p.effective_well is None


# ---------------------------------------------------------------------------
# _check_capture
# ---------------------------------------------------------------------------

class TestCheckCapture:
    def test_within_radius(self):
        p = PrognosticParticle(well_coords=(0, 0, 0), capture_radius=5.0)
        assert p._check_capture((3.0, 0.0, 0.0)) is True

    def test_outside_radius(self):
        p = PrognosticParticle(well_coords=(0, 0, 0), capture_radius=5.0)
        assert p._check_capture((10.0, 0.0, 0.0)) is False

    def test_no_well(self):
        p = PrognosticParticle()
        assert p._check_capture((1.0, 1.0, 1.0)) is False

    def test_exact_radius(self):
        p = PrognosticParticle(well_coords=(0, 0, 0), capture_radius=5.0)
        assert p._check_capture((5.0, 0.0, 0.0)) is True


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_perfectly_straight(self):
        p = PrognosticParticle()
        positions = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        c = p._compute_confidence(positions)
        assert c == pytest.approx(1.0)

    def test_fully_reversed(self):
        p = PrognosticParticle()
        positions = [(0, 0, 0), (1, 0, 0), (0, 0, 0), (-1, 0, 0)]
        c = p._compute_confidence(positions)
        # diffs: (1,0,0), (-1,0,0), (-1,0,0) → cosines: -1, 1 → mean 0 → conf 0.5
        assert c == pytest.approx(0.5)

    def test_perfect_reversal_two_segments(self):
        p = PrognosticParticle()
        positions = [(0, 0, 0), (1, 0, 0), (0, 0, 0)]
        c = p._compute_confidence(positions)
        # diffs: (1,0,0), (-1,0,0) → cosine: -1 → mean -1 → conf 0.0
        assert c == pytest.approx(0.0)

    def test_too_few_points(self):
        p = PrognosticParticle()
        assert p._compute_confidence([(0, 0, 0)]) == 0.5
        assert p._compute_confidence([(0, 0, 0), (1, 1, 1)]) == 0.5

    def test_stationary(self):
        p = PrognosticParticle()
        positions = [(1, 1, 1), (1, 1, 1), (1, 1, 1)]
        assert p._compute_confidence(positions) == 0.5

    def test_curved(self):
        p = PrognosticParticle()
        positions = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
        c = p._compute_confidence(positions)
        assert 0.0 < c < 1.0


# ---------------------------------------------------------------------------
# _track_ballistic
# ---------------------------------------------------------------------------

class TestTrackBallistic:
    def test_toward_well(self):
        p = PrognosticParticle(horizon_steps=5, well_coords=(5, 0, 0))
        r = p._track_ballistic((0, 0, 0))
        assert r.steps == 5
        # Should move toward well
        assert r.endpoint[0] > 0
        assert r.will_capture is True  # 5 steps * 1 unit = 5 units = well at 5

    def test_no_well(self):
        p = PrognosticParticle(horizon_steps=3)
        r = p._track_ballistic((1, 2, 3))
        assert r.endpoint == (1.0, 2.0, 3.0)
        assert r.will_capture is False
        assert r.confidence == 0.3

    def test_at_well(self):
        p = PrognosticParticle(horizon_steps=5, well_coords=(0, 0, 0))
        r = p._track_ballistic((0, 0, 0))
        assert r.endpoint == (0.0, 0.0, 0.0)
        assert r.will_capture is True


# ---------------------------------------------------------------------------
# _track_with_velocity
# ---------------------------------------------------------------------------

class TestTrackWithVelocity:
    def test_constant_velocity(self):
        p = PrognosticParticle(horizon_steps=5, well_coords=(10, 0, 0))
        vel = np.array([[1.0, 0.0, 0.0]] * 10)
        r = p._track_with_velocity((0, 0, 0), vel)
        assert r.steps == 5
        assert r.endpoint[0] == pytest.approx(5.0)
        # dist from (5,0,0) to (10,0,0) = 5.0, radius = 5.0 → within capture
        assert r.will_capture is True

    def test_constant_velocity_outside(self):
        p = PrognosticParticle(horizon_steps=3, well_coords=(10, 0, 0))
        vel = np.array([[1.0, 0.0, 0.0]] * 10)
        r = p._track_with_velocity((0, 0, 0), vel)
        assert r.steps == 3
        assert r.endpoint[0] == pytest.approx(3.0)
        # dist from (3,0,0) to (10,0,0) = 7.0 > 5.0 → outside capture
        assert r.will_capture is False

    def test_capture_reached(self):
        p = PrognosticParticle(horizon_steps=5, well_coords=(3, 0, 0), capture_radius=5.0)
        vel = np.array([[1.0, 0.0, 0.0]] * 10)
        r = p._track_with_velocity((0, 0, 0), vel)
        assert r.will_capture is True

    def test_shorter_field_than_horizon(self):
        p = PrognosticParticle(horizon_steps=10)
        vel = np.array([[1.0, 0.0, 0.0]] * 3)  # only 3 steps
        r = p._track_with_velocity((0, 0, 0), vel)
        assert r.steps == 3


# ---------------------------------------------------------------------------
# track() dispatch
# ---------------------------------------------------------------------------

class TestTrackDispatch:
    def test_velocity_field_priority(self):
        p = PrognosticParticle(horizon_steps=3, well_coords=(5, 0, 0))
        vel = np.array([[1.0, 0.0, 0.0]] * 5)
        r = p.track((0, 0, 0), {'velocity_field': vel})
        assert r.steps == 3

    def test_no_runner_no_velocity(self):
        p = PrognosticParticle(horizon_steps=3, well_coords=(5, 0, 0))
        r = p.track((0, 0, 0))
        assert r.steps == 3  # ballistic fallback


# ---------------------------------------------------------------------------
# get_observation_vector
# ---------------------------------------------------------------------------

class TestObservationVector:
    def test_shape_and_values(self):
        p = PrognosticParticle(horizon_steps=3, well_coords=(5, 0, 0))
        r = p._track_ballistic((0, 0, 0))
        obs = p.get_observation_vector(r)
        assert len(obs) == 6
        assert obs[0] == r.endpoint[0]
        assert obs[3] == float(r.will_capture)
        assert obs[4] == r.confidence
        assert obs[5] == float(r.steps)

    def test_types(self):
        p = PrognosticParticle(horizon_steps=2, well_coords=(1, 0, 0))
        r = p._track_ballistic((0, 0, 0))
        obs = p.get_observation_vector(r)
        for v in obs:
            assert isinstance(v, float)
