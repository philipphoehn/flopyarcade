"""Tests for flopyarcade.k_region (A5: K-region abstraction)."""

import math
import pytest

from flopyarcade.k_region import KRegion, KRegionManager


# ---------------------------------------------------------------------------
# KRegion dataclass
# ---------------------------------------------------------------------------

class TestKRegion:
    def test_defaults(self):
        r = KRegion(label="main", position=(100, 200, 20), radius=5.0)
        assert r.label == "main"
        assert r.position == (100, 200, 20)
        assert r.radius == 5.0
        assert r.flow_rate == 0.0
        assert r.reward_weight == 1.0
        assert r.active is True

    def test_custom_flow_rate(self):
        r = KRegion(label="w", position=(0, 0, 0), radius=1, flow_rate=500.0)
        assert r.flow_rate == 500.0

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="radius must be >= 0"):
            KRegion(label="bad", position=(0, 0, 0), radius=-1)

    def test_bad_position_raises(self):
        with pytest.raises(ValueError, match="position must be"):
            KRegion(label="bad", position=(0, 0), radius=1)

    # -- Spatial queries --

    def test_distance_exact(self):
        r = KRegion(label="t", position=(0, 0, 0), radius=10)
        assert r.distance((3, 4, 0)) == pytest.approx(5.0)

    def test_distance_3d(self):
        r = KRegion(label="t", position=(0, 0, 0), radius=10)
        assert r.distance((3, 4, 12)) == pytest.approx(13.0)

    def test_contains_inside(self):
        r = KRegion(label="t", position=(0, 0, 0), radius=5)
        assert r.contains((1, 1, 1)) is True

    def test_contains_outside(self):
        r = KRegion(label="t", position=(0, 0, 0), radius=5)
        assert r.contains((10, 0, 0)) is False

    def test_contains_inactive(self):
        r = KRegion(label="t", position=(0, 0, 0), radius=5, active=False)
        assert r.contains((0, 0, 0)) is False

    def test_in_xy(self):
        r = KRegion(label="t", position=(0, 0, 100), radius=5)
        # XY distance is 0, so in_xy is True even though Z is far
        assert r.in_xy((0, 0, 999)) is True

    def test_in_xy_outside(self):
        r = KRegion(label="t", position=(0, 0, 0), radius=5)
        assert r.in_xy((10, 0, 0)) is False

    def test_compute_reward(self):
        r = KRegion(label="t", position=(0, 0, 0), radius=1, reward_weight=2.5)
        assert r.compute_reward(base_reward=10.0) == pytest.approx(25.0)

    def test_to_dict(self):
        r = KRegion(label="t", position=(1, 2, 3), radius=4, flow_rate=5, reward_weight=0.5)
        d = r.to_dict()
        assert d["label"] == "t"
        assert d["position"] == [1, 2, 3]
        assert d["radius"] == 4
        assert d["flow_rate"] == 5
        assert d["reward_weight"] == 0.5

    def test_repr(self):
        r = KRegion(label="main", position=(100, 200, 20), radius=5)
        s = repr(r)
        assert "KRegion" in s
        assert "main" in s


# ---------------------------------------------------------------------------
# KRegionManager – add / remove / get
# ---------------------------------------------------------------------------

class TestManagerAddRemove:
    def _mgr_with_two(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="a", position=(0, 0, 0), radius=5))
        mgr.add_region(KRegion(label="b", position=(10, 0, 0), radius=3))
        return mgr

    def test_add_and_count(self):
        mgr = self._mgr_with_two()
        assert mgr.region_count == 2
        assert len(mgr) == 2

    def test_duplicate_label_raises(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="x", position=(0, 0, 0), radius=1))
        with pytest.raises(ValueError, match="already exists"):
            mgr.add_region(KRegion(label="x", position=(1, 0, 0), radius=1))

    def test_get(self):
        mgr = self._mgr_with_two()
        assert mgr.get("a").label == "a"

    def test_get_missing_raises(self):
        mgr = self._mgr_with_two()
        with pytest.raises(KeyError):
            mgr.get("z")

    def test_contains(self):
        mgr = self._mgr_with_two()
        assert "a" in mgr
        assert "z" not in mgr

    def test_remove(self):
        mgr = self._mgr_with_two()
        removed = mgr.remove_region("a")
        assert removed.label == "a"
        assert mgr.region_count == 1
        assert "a" not in mgr
        assert "b" in mgr

    def test_remove_missing_raises(self):
        mgr = self._mgr_with_two()
        with pytest.raises(KeyError):
            mgr.remove_region("z")

    def test_remove_then_add_same_label(self):
        mgr = self._mgr_with_two()
        mgr.remove_region("a")
        mgr.add_region(KRegion(label="a", position=(50, 50, 50), radius=1))
        assert mgr.get("a").position == (50, 50, 50)

    def test_clear(self):
        mgr = self._mgr_with_two()
        mgr.clear()
        assert mgr.region_count == 0


# ---------------------------------------------------------------------------
# KRegionManager – spatial queries
# ---------------------------------------------------------------------------

class TestManagerSpatial:
    def _mgr(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="close", position=(10, 0, 0), radius=3))
        mgr.add_region(KRegion(label="far", position=(100, 0, 0), radius=3))
        return mgr

    def test_nearest_region(self):
        mgr = self._mgr()
        r = mgr.nearest_region((5, 0, 0))
        assert r.label == "close"

    def test_nearest_region_empty(self):
        mgr = KRegionManager()
        assert mgr.nearest_region((0, 0, 0)) is None

    def test_region_at_inside(self):
        mgr = self._mgr()
        r = mgr.region_at((10, 0, 0))
        assert r.label == "close"

    def test_region_at_outside(self):
        mgr = self._mgr()
        assert mgr.region_at((50, 0, 0)) is None

    def test_region_at_xy(self):
        mgr = self._mgr()
        r = mgr.region_at_xy((10, 0, 999))
        assert r.label == "close"

    def test_all_containing(self):
        mgr = KRegionManager()
        # Two overlapping regions
        mgr.add_region(KRegion(label="a", position=(0, 0, 0), radius=10))
        mgr.add_region(KRegion(label="b", position=(5, 0, 0), radius=10))
        result = mgr.all_containing((5, 0, 0))
        assert len(result) == 2

    def test_all_containing_none(self):
        mgr = self._mgr()
        assert mgr.all_containing((50, 50, 50)) == []

    def test_inactive_excluded_from_nearest(self):
        mgr = self._mgr()
        mgr.deactivate("close")
        r = mgr.nearest_region((5, 0, 0))
        assert r.label == "far"


# ---------------------------------------------------------------------------
# KRegionManager – reward aggregation
# ---------------------------------------------------------------------------

class TestManagerReward:
    def test_total_reward(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="a", position=(0, 0, 0), radius=1, reward_weight=1.0))
        mgr.add_region(KRegion(label="b", position=(1, 0, 0), radius=1, reward_weight=2.0))
        assert mgr.total_reward(base_reward=10) == pytest.approx(30.0)

    def test_max_reward(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="a", position=(0, 0, 0), radius=1, reward_weight=1.0))
        mgr.add_region(KRegion(label="b", position=(1, 0, 0), radius=1, reward_weight=5.0))
        assert mgr.max_reward(base_reward=10) == pytest.approx(50.0)

    def test_max_reward_empty(self):
        mgr = KRegionManager()
        assert mgr.max_reward() == 0.0


# ---------------------------------------------------------------------------
# KRegionManager – flow rate aggregation
# ---------------------------------------------------------------------------

class TestManagerFlowRate:
    def _mgr(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="pump1", position=(0, 0, 0), radius=1, flow_rate=500))
        mgr.add_region(KRegion(label="pump2", position=(1, 0, 0), radius=1, flow_rate=300))
        mgr.add_region(KRegion(label="recharge1", position=(2, 0, 0), radius=1, flow_rate=-200))
        return mgr

    def test_total_flow_rate(self):
        mgr = self._mgr()
        assert mgr.total_flow_rate() == pytest.approx(600.0)

    def test_net_extraction(self):
        mgr = self._mgr()
        assert mgr.net_extraction() == pytest.approx(800.0)

    def test_net_recharge(self):
        mgr = self._mgr()
        assert mgr.net_recharge() == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# KRegionManager – activation toggling
# ---------------------------------------------------------------------------

class TestManagerActivation:
    def test_deactivate(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="a", position=(0, 0, 0), radius=1))
        mgr.deactivate("a")
        assert mgr.get("a").active is False
        assert mgr.active_regions == []

    def test_reactivate(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="a", position=(0, 0, 0), radius=1))
        mgr.deactivate("a")
        mgr.activate("a")
        assert mgr.get("a").active is True

    def test_set_flow_rate(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="a", position=(0, 0, 0), radius=1, flow_rate=100))
        mgr.set_flow_rate("a", 250)
        assert mgr.get("a").flow_rate == 250


# ---------------------------------------------------------------------------
# KRegionManager – from_helper_wells_dict
# ---------------------------------------------------------------------------

class TestFromHelperWellsDict:
    def test_basic(self):
        hw = {
            "wellX1": 160.0, "wellY1": 200.0, "wellZ1": 20.0, "wellQ1": 300.0,
            "wellX2": 170.0, "wellY2": 210.0, "wellZ2": 20.0, "wellQ2": -150.0,
        }
        mgr = KRegionManager.from_helper_wells_dict(hw, base_radius=2.0)
        assert mgr.region_count == 2
        assert mgr.get("helper1").position == (160.0, 200.0, 20.0)
        assert mgr.get("helper1").flow_rate == 300.0
        assert mgr.get("helper2").flow_rate == -150.0

    def test_empty(self):
        mgr = KRegionManager.from_helper_wells_dict({})
        assert mgr.region_count == 0

    def test_missing_q_defaults_zero(self):
        hw = {"wellX1": 10.0, "wellY1": 20.0, "wellZ1": 30.0}
        mgr = KRegionManager.from_helper_wells_dict(hw)
        assert mgr.get("helper1").flow_rate == 0.0


# ---------------------------------------------------------------------------
# KRegionManager – serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="a", position=(1, 2, 3), radius=4, flow_rate=5, reward_weight=0.5))
        mgr.add_region(KRegion(label="b", position=(4, 5, 6), radius=7, active=False))

        d = mgr.to_dict()
        mgr2 = KRegionManager.from_dict(d)

        assert mgr2.region_count == 2
        assert mgr2.get("a").position == (1, 2, 3)
        assert mgr2.get("a").radius == 4
        assert mgr2.get("a").flow_rate == 5
        assert mgr2.get("a").reward_weight == 0.5
        assert mgr2.get("b").active is False

    def test_to_dict_keys(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="x", position=(0, 0, 0), radius=1))
        d = mgr.to_dict()
        assert "region_count" in d
        assert "regions" in d
        assert len(d["regions"]) == 1


# ---------------------------------------------------------------------------
# KRegionManager – iteration
# ---------------------------------------------------------------------------

class TestIteration:
    def test_iter(self):
        mgr = KRegionManager()
        mgr.add_region(KRegion(label="a", position=(0, 0, 0), radius=1))
        mgr.add_region(KRegion(label="b", position=(1, 0, 0), radius=1))
        labels = [r.label for r in mgr]
        assert labels == ["a", "b"]


# ---------------------------------------------------------------------------
# Integration: realistic scenario
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_envtype5_scenario(self):
        """Simulate an ENVTYPE 5 scenario with main well + 2 helper wells."""
        mgr = KRegionManager()
        # Main well
        mgr.add_region(KRegion(
            label="main", position=(80, 320, 20), radius=math.sqrt(8), flow_rate=400
        ))
        # Helper wells
        mgr.add_region(KRegion(
            label="helper1", position=(160, 200, 20), radius=2.0, flow_rate=300
        ))
        mgr.add_region(KRegion(
            label="helper2", position=(160, 400, 20), radius=2.0, flow_rate=300
        ))

        assert mgr.region_count == 3
        assert mgr.total_flow_rate() == pytest.approx(1000.0)

        # Particle near main well
        particle = (80, 321, 20)
        region = mgr.region_at(particle)
        assert region is not None
        assert region.label == "main"

        # Particle near helper1
        particle2 = (160, 201, 20)
        region2 = mgr.region_at(particle2)
        assert region2 is not None
        assert region2.label == "helper1"

        # Deactivate helper2, re-query
        mgr.deactivate("helper2")
        assert mgr.region_count == 3
        assert len(mgr.active_regions) == 2
        assert mgr.region_at((160, 401, 20)) is None

        # Re-serialize
        d = mgr.to_dict()
        mgr2 = KRegionManager.from_dict(d)
        assert mgr2.get("helper2").active is False
