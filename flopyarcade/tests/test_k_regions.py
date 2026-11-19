"""Tests for flopyarcade.k_regions (A5: K-regions abstraction)."""

import math
import numpy as np
import pytest

from flopyarcade.k_regions import KRegion, KRegionManager


# ---------------------------------------------------------------------------
# KRegion
# ---------------------------------------------------------------------------

class TestKRegion:
    def test_defaults(self):
        r = KRegion(name="test")
        assert r.name == "test"
        assert r.well_ids == []
        assert r.center == (0.0, 0.0)
        assert r.radius == 10.0
        assert r.K == (25.0, 25.0, 25.0)
        assert r.is_capture is True

    def test_invalid_radius(self):
        with pytest.raises(ValueError, match="radius"):
            KRegion(name="bad", radius=0.0)
        with pytest.raises(ValueError, match="radius"):
            KRegion(name="bad", radius=-5.0)

    def test_invalid_K_length(self):
        with pytest.raises(ValueError, match="K must be"):
            KRegion(name="bad", K=(1.0, 2.0))

    def test_invalid_K_values(self):
        with pytest.raises(ValueError, match="K values"):
            KRegion(name="bad", K=(0.0, 25.0, 25.0))

    def test_invalid_well_id(self):
        with pytest.raises(ValueError, match="well_ids"):
            KRegion(name="bad", well_ids=[0])

    def test_contains_point_inside(self):
        r = KRegion(name="c", center=(5.0, 5.0), radius=3.0)
        assert r.contains_point(5.0, 5.0) is True
        assert r.contains_point(7.0, 5.0) is True
        assert r.contains_point(5.0, 7.0) is True

    def test_contains_point_outside(self):
        r = KRegion(name="c", center=(5.0, 5.0), radius=3.0)
        assert r.contains_point(10.0, 5.0) is False
        assert r.contains_point(5.0, 10.0) is False

    def test_contains_point_boundary(self):
        r = KRegion(name="c", center=(0.0, 0.0), radius=5.0)
        assert r.contains_point(3.0, 4.0) is True  # 3-4-5 triangle

    def test_area(self):
        r = KRegion(name="a", radius=2.0)
        assert r.area() == pytest.approx(4.0 * math.pi)

    def test_to_dict(self):
        r = KRegion(name="d", well_ids=[1, 2], center=(3.0, 4.0),
                    radius=7.0, K=(10.0, 20.0, 30.0), is_capture=False)
        d = r.to_dict()
        assert d['name'] == 'd'
        assert d['well_ids'] == [1, 2]
        assert d['center'] == [3.0, 4.0]
        assert d['radius'] == 7.0
        assert d['K'] == [10.0, 20.0, 30.0]
        assert d['is_capture'] is False

    def test_from_dict_roundtrip(self):
        r = KRegion(name="rt", well_ids=[3], center=(1.0, 2.0),
                    radius=5.0, K=(1.0, 2.0, 3.0), is_capture=False)
        d = r.to_dict()
        r2 = KRegion.from_dict(d)
        assert r2.name == r.name
        assert r2.well_ids == r.well_ids
        assert r2.center == r.center
        assert r2.radius == r.radius
        assert r2.K == r.K
        assert r2.is_capture == r.is_capture


# ---------------------------------------------------------------------------
# KRegionManager
# ---------------------------------------------------------------------------

class TestKRegionManager:
    def _make_mgr(self, ncol=20, nrow=15):
        return KRegionManager(dx=10.0, dy=10.0, ncol=ncol, nrow=nrow)

    def test_invalid_grid(self):
        with pytest.raises(ValueError, match="dx and dy"):
            KRegionManager(dx=0.0)
        with pytest.raises(ValueError, match="ncol"):
            KRegionManager(ncol=0)

    def test_empty(self):
        mgr = self._make_mgr()
        assert mgr.n_regions == 0
        assert len(mgr) == 0
        assert list(mgr) == []

    def test_add_and_get(self):
        mgr = self._make_mgr()
        r = KRegion(name="zone1", center=(100.0, 75.0), radius=30.0)
        mgr.add_region(r)
        assert mgr.n_regions == 1
        assert mgr.get("zone1") is r
        assert "zone1" in mgr
        assert "zone2" not in mgr

    def test_duplicate_name_raises(self):
        mgr = self._make_mgr()
        mgr.add_region(KRegion(name="x", radius=10.0))
        with pytest.raises(ValueError, match="already exists"):
            mgr.add_region(KRegion(name="x", radius=20.0))

    def test_remove(self):
        mgr = self._make_mgr()
        mgr.add_region(KRegion(name="a", radius=10.0))
        mgr.add_region(KRegion(name="b", radius=20.0))
        removed = mgr.remove_region("a")
        assert removed.name == "a"
        assert mgr.n_regions == 1
        assert "b" in mgr
        with pytest.raises(KeyError):
            mgr.remove_region("a")

    def test_zone_membership_empty(self):
        mgr = self._make_mgr(ncol=5, nrow=4)
        labels = mgr.zone_membership()
        assert labels.shape == (4, 5)
        assert (labels == -1).all()

    def test_zone_membership_single(self):
        # Grid: 5 cols x 4 rows, dx=dy=10 → domain [0,50]x[0,40]
        mgr = self._make_mgr(ncol=5, nrow=4)
        mgr.add_region(KRegion(name="c", center=(25.0, 20.0), radius=15.0))
        labels = mgr.zone_membership()
        assert labels.shape == (4, 5)
        # Centre cell (row=1 or 2, col=2) should be inside
        assert (labels >= 0).any()
        assert (labels == 0).any()

    def test_zone_membership_nearest(self):
        mgr = self._make_mgr(ncol=10, nrow=10)
        # Two overlapping regions
        mgr.add_region(KRegion(name="a", center=(25.0, 50.0), radius=20.0))
        mgr.add_region(KRegion(name="b", center=(50.0, 50.0), radius=20.0))
        labels = mgr.zone_membership()
        # Cell near (25,50) should be region 0; near (50,50) should be region 1
        # Both should appear in the label array
        assert 0 in labels
        assert 1 in labels

    def test_region_at(self):
        mgr = self._make_mgr()
        mgr.add_region(KRegion(name="z", center=(50.0, 50.0), radius=10.0))
        assert mgr.region_at(50.0, 50.0) == 0
        assert mgr.region_at(55.0, 55.0) == 0
        assert mgr.region_at(200.0, 200.0) is None

    def test_detect_overlaps_none(self):
        mgr = self._make_mgr()
        mgr.add_region(KRegion(name="a", center=(0.0, 0.0), radius=5.0))
        mgr.add_region(KRegion(name="b", center=(100.0, 100.0), radius=5.0))
        assert mgr.detect_overlaps() == []

    def test_detect_overlaps_some(self):
        mgr = self._make_mgr()
        mgr.add_region(KRegion(name="a", center=(0.0, 0.0), radius=10.0))
        mgr.add_region(KRegion(name="b", center=(15.0, 0.0), radius=10.0))
        overlaps = mgr.detect_overlaps()
        assert len(overlaps) == 1
        assert overlaps[0][0] == "a"
        assert overlaps[0][1] == "b"
        assert 0 < overlaps[0][2] <= 1.0

    def test_wells_per_region(self):
        mgr = self._make_mgr()
        mgr.add_region(KRegion(name="cap", well_ids=[1, 2]))
        mgr.add_region(KRegion(name="rec", well_ids=[3]))
        wpr = mgr.wells_per_region()
        assert wpr == {'cap': [1, 2], 'rec': [3]}

    def test_region_for_well(self):
        mgr = self._make_mgr()
        mgr.add_region(KRegion(name="cap", well_ids=[1, 2]))
        assert mgr.region_for_well(1) == "cap"
        assert mgr.region_for_well(99) is None

    def test_build_K_field(self):
        mgr = self._make_mgr(ncol=4, nrow=4)
        # radius=10 → only centre cells covered; corners stay at default K=25
        mgr.add_region(KRegion(name="zone", center=(20.0, 20.0),
                               radius=10.0, K=(5.0, 10.0, 15.0)))
        K = mgr.build_K_field()
        assert K.shape == (4, 4, 3)
        # At least one cell should have K = (5, 10, 15)
        assert np.any((K[:, :, 0] == 5.0) & (K[:, :, 1] == 10.0) & (K[:, :, 2] == 15.0))
        # Cells outside should default to (25, 25, 25)
        assert np.any(K[:, :, 0] == 25.0)

    def test_to_dict_roundtrip(self):
        mgr = KRegionManager(dx=10.0, dy=10.0, ncol=10, nrow=8, origin=(5.0, 10.0))
        mgr.add_region(KRegion(name="r1", well_ids=[1], center=(30.0, 40.0),
                               radius=12.0, K=(7.0, 7.0, 7.0)))
        d = mgr.to_dict()
        mgr2 = KRegionManager.from_dict(d)
        assert mgr2.n_regions == 1
        assert mgr2.get("r1").radius == 12.0
        assert mgr2.dx == 10.0
        assert mgr2.origin == (5.0, 10.0)

    def test_repr(self):
        mgr = self._make_mgr(ncol=10, nrow=8)
        r = repr(mgr)
        assert "KRegionManager" in r
        assert "n=0" in r

    def test_iteration_order(self):
        mgr = self._make_mgr()
        mgr.add_region(KRegion(name="first", radius=5.0))
        mgr.add_region(KRegion(name="second", radius=5.0))
        names = [r.name for r in mgr]
        assert names == ["first", "second"]
