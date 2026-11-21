#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - tests
# A10: Protection-zone overlap prevention tests

"""Tests for flopyarcade.overlap_prevention module."""

import math

import pytest

from flopyarcade.overlap_prevention import (
    ProtectionZone,
    detect_overlaps,
    max_non_overlapping_radius,
    resolve_overlaps,
    validate_zones,
)


# ---------------------------------------------------------------------------
# ProtectionZone creation
# ---------------------------------------------------------------------------

class TestProtectionZoneCreation:
    def test_basic_creation(self):
        z = ProtectionZone(well_id=1, center=(5.0, 3.0), radius=2.0)
        assert z.well_id == 1
        assert z.center == (5.0, 3.0)
        assert z.radius == 2.0

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError, match="radius must be > 0"):
            ProtectionZone(well_id=0, center=(0.0, 0.0), radius=0.0)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="radius must be > 0"):
            ProtectionZone(well_id=0, center=(0.0, 0.0), radius=-1.0)


# ---------------------------------------------------------------------------
# detect_overlaps
# ---------------------------------------------------------------------------

class TestDetectOverlaps:
    def test_two_overlapping_zones(self):
        z1 = ProtectionZone(0, (0.0, 0.0), 5.0)
        z2 = ProtectionZone(1, (6.0, 0.0), 5.0)
        # distance = 6, sum of radii = 10, overlap = 4
        result = detect_overlaps([z1, z2])
        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][1] == 1
        assert result[0][2] == pytest.approx(4.0)

    def test_two_non_overlapping_zones(self):
        z1 = ProtectionZone(0, (0.0, 0.0), 3.0)
        z2 = ProtectionZone(1, (10.0, 0.0), 3.0)
        result = detect_overlaps([z1, z2])
        assert result == []

    def test_three_zones_one_pair_overlaps(self):
        z1 = ProtectionZone(0, (0.0, 0.0), 4.0)
        z2 = ProtectionZone(1, (7.0, 0.0), 4.0)  # overlaps z1: dist=7, sum=8
        z3 = ProtectionZone(2, (20.0, 0.0), 2.0)  # no overlap with either
        result = detect_overlaps([z1, z2, z3])
        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][1] == 1

    def test_identical_zones(self):
        z1 = ProtectionZone(0, (5.0, 5.0), 3.0)
        z2 = ProtectionZone(1, (5.0, 5.0), 3.0)
        # distance = 0, sum of radii = 6, overlap = 6
        result = detect_overlaps([z1, z2])
        assert len(result) == 1
        assert result[0][2] == pytest.approx(6.0)

    def test_touching_zones_no_overlap(self):
        """Zones that just touch (distance == sum of radii) should NOT be flagged."""
        z1 = ProtectionZone(0, (0.0, 0.0), 3.0)
        z2 = ProtectionZone(1, (6.0, 0.0), 3.0)  # dist=6, sum=6
        result = detect_overlaps([z1, z2])
        assert result == []

    def test_single_zone_no_overlap(self):
        z1 = ProtectionZone(0, (1.0, 1.0), 2.0)
        assert detect_overlaps([z1]) == []

    def test_empty_list(self):
        assert detect_overlaps([]) == []


# ---------------------------------------------------------------------------
# resolve_overlaps
# ---------------------------------------------------------------------------

class TestResolveOverlaps:
    def test_shrink_strategy(self):
        z1 = ProtectionZone(0, (0.0, 0.0), 5.0)
        z2 = ProtectionZone(1, (6.0, 0.0), 5.0)
        result = resolve_overlaps([z1, z2], strategy="shrink")
        assert len(result) == 2
        # After resolution there should be no overlaps
        assert detect_overlaps(result) == []
        # The larger (or equal) zone should have been shrunk
        assert result[0].radius < 5.0 or result[1].radius < 5.0

    def test_reject_strategy_raises(self):
        z1 = ProtectionZone(0, (0.0, 0.0), 5.0)
        z2 = ProtectionZone(1, (6.0, 0.0), 5.0)
        with pytest.raises(ValueError, match="overlap"):
            resolve_overlaps([z1, z2], strategy="reject")

    def test_reject_strategy_no_overlap_passes(self):
        z1 = ProtectionZone(0, (0.0, 0.0), 3.0)
        z2 = ProtectionZone(1, (10.0, 0.0), 3.0)
        result = resolve_overlaps([z1, z2], strategy="reject")
        assert len(result) == 2

    def test_unknown_strategy_raises(self):
        z1 = ProtectionZone(0, (0.0, 0.0), 3.0)
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolve_overlaps([z1], strategy="explode")


# ---------------------------------------------------------------------------
# validate_zones
# ---------------------------------------------------------------------------

class TestValidateZones:
    def test_all_within_domain(self):
        z1 = ProtectionZone(0, (5.0, 5.0), 2.0)
        z2 = ProtectionZone(1, (10.0, 8.0), 3.0)
        errors = validate_zones([z1, z2], domain_x=20.0, domain_y=20.0)
        assert errors == []

    def test_one_outside_domain(self):
        z1 = ProtectionZone(0, (1.0, 5.0), 3.0)  # extends below x=0
        errors = validate_zones([z1], domain_x=20.0, domain_y=20.0)
        assert len(errors) >= 1
        assert "x=0" in errors[0]

    def test_exceeds_domain_x(self):
        z1 = ProtectionZone(0, (19.0, 5.0), 3.0)  # 19+3=22 > 20
        errors = validate_zones([z1], domain_x=20.0, domain_y=20.0)
        assert any("x=20" in e for e in errors)

    def test_empty_list(self):
        errors = validate_zones([], domain_x=10.0, domain_y=10.0)
        assert errors == []


# ---------------------------------------------------------------------------
# max_non_overlapping_radius
# ---------------------------------------------------------------------------

class TestMaxNonOverlappingRadius:
    def test_space_available(self):
        existing = ProtectionZone(0, (0.0, 0.0), 3.0)
        candidate = ProtectionZone(1, (10.0, 0.0), 1.0)  # dist=10, avail=7
        max_r = max_non_overlapping_radius([existing], candidate)
        assert max_r == pytest.approx(7.0)

    def test_no_space_candidate_inside_existing(self):
        existing = ProtectionZone(0, (0.0, 0.0), 10.0)
        candidate = ProtectionZone(1, (3.0, 0.0), 1.0)  # dist=3 < radius=10
        max_r = max_non_overlapping_radius([existing], candidate)
        assert max_r == 0.0

    def test_no_existing_zones(self):
        candidate = ProtectionZone(0, (5.0, 5.0), 1.0)
        max_r = max_non_overlapping_radius([], candidate)
        assert max_r == float("inf")

    def test_multiple_existing_zones(self):
        e1 = ProtectionZone(0, (0.0, 0.0), 3.0)
        e2 = ProtectionZone(1, (10.0, 0.0), 5.0)
        candidate = ProtectionZone(2, (5.0, 0.0), 1.0)
        # dist to e1 = 5, avail = 5-3 = 2
        # dist to e2 = 5, avail = 5-5 = 0 -> return 0.0
        max_r = max_non_overlapping_radius([e1, e2], candidate)
        assert max_r == 0.0

    def test_multiple_existing_zones_min_wins(self):
        e1 = ProtectionZone(0, (0.0, 0.0), 2.0)
        e2 = ProtectionZone(1, (20.0, 0.0), 1.0)
        candidate = ProtectionZone(2, (5.0, 0.0), 1.0)
        # dist to e1 = 5, avail = 5-2 = 3
        # dist to e2 = 15, avail = 15-1 = 14
        max_r = max_non_overlapping_radius([e1, e2], candidate)
        assert max_r == pytest.approx(3.0)
