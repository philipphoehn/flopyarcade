#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - overlap_prevention
# A10: Prevent protection-zone overlap for multi-well envs
# Detects, resolves, and validates circular protection zones around wells.

"""Protection-zone overlap prevention for multi-well FloPyArcade environments.

This module provides utilities to detect and resolve overlapping circular
protection zones around wells, validate zones against domain bounds, and
compute the maximum non-overlapping radius for a candidate zone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ProtectionZone:
    """A circular protection zone around a well.

    Parameters
    ----------
    well_id : int
        Unique identifier for the well.
    center : tuple of float
        (x, y) coordinates of the zone center.
    radius : float
        Radius of the zone. Must be > 0.
    """

    well_id: int
    center: Tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(
                f"ProtectionZone radius must be > 0, got {self.radius}"
            )


def _distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def detect_overlaps(
    zones: List[ProtectionZone],
) -> List[Tuple[int, int, float]]:
    """Detect all pairs of overlapping protection zones.

    Parameters
    ----------
    zones : list of ProtectionZone
        The zones to check.

    Returns
    -------
    list of (zone_i, zone_j, overlap_distance)
        Pairs where overlap_distance = (r_i + r_j) - distance > 0.
    """
    overlaps: List[Tuple[int, int, float]] = []
    n = len(zones)
    for i in range(n):
        for j in range(i + 1, n):
            dist = _distance(zones[i].center, zones[j].center)
            overlap = zones[i].radius + zones[j].radius - dist
            if overlap > 0:
                overlaps.append((i, j, overlap))
    return overlaps


def resolve_overlaps(
    zones: List[ProtectionZone], strategy: str = "shrink"
) -> List[ProtectionZone]:
    """Adjust zone radii to eliminate all overlaps.

    Parameters
    ----------
    zones : list of ProtectionZone
        The zones to resolve.
    strategy : str
        'shrink' -- reduce the larger zone's radius per overlapping pair.
        'reject' -- raise ValueError if any overlap exists.

    Returns
    -------
    list of ProtectionZone
        Zones with adjusted radii (shrink) or original (reject, if no overlap).

    Raises
    ------
    ValueError
        If strategy is 'reject' and overlaps exist, or strategy is unknown.
    """
    if strategy == "reject":
        overlaps = detect_overlaps(zones)
        if overlaps:
            raise ValueError(
                f"Protection zone overlap detected: {overlaps}. "
                "Use strategy='shrink' to auto-resolve."
            )
        return list(zones)

    if strategy != "shrink":
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Use 'shrink' or 'reject'."
        )

    # Work on copies to avoid mutating the input
    result = [
        ProtectionZone(well_id=z.well_id, center=z.center, radius=z.radius)
        for z in zones
    ]

    # Iteratively shrink until no overlaps remain
    max_iter = max(len(result) * (len(result) - 1), 1) + 1
    for _ in range(max_iter):
        overlaps = detect_overlaps(result)
        if not overlaps:
            break
        for i, j, overlap in overlaps:
            if result[i].radius >= result[j].radius:
                result[i].radius = max(1e-9, result[i].radius - overlap)
            else:
                result[j].radius = max(1e-9, result[j].radius - overlap)

    return result


def validate_zones(
    zones: List[ProtectionZone], domain_x: float, domain_y: float
) -> List[str]:
    """Validate that all zones fit within the domain bounds.

    Parameters
    ----------
    zones : list of ProtectionZone
        Zones to validate.
    domain_x : float
        Width of the domain (x extent, 0 to domain_x).
    domain_y : float
        Height of the domain (y extent, 0 to domain_y).

    Returns
    -------
    list of str
        Error messages. Empty list means all zones are valid.
    """
    errors: List[str] = []
    for z in zones:
        cx, cy = z.center
        if cx - z.radius < 0:
            errors.append(
                f"Well {z.well_id}: zone extends below x=0 "
                f"(center_x={cx:.4f}, radius={z.radius:.4f})"
            )
        if cx + z.radius > domain_x:
            errors.append(
                f"Well {z.well_id}: zone exceeds x={domain_x} "
                f"(center_x={cx:.4f}, radius={z.radius:.4f})"
            )
        if cy - z.radius < 0:
            errors.append(
                f"Well {z.well_id}: zone extends below y=0 "
                f"(center_y={cy:.4f}, radius={z.radius:.4f})"
            )
        if cy + z.radius > domain_y:
            errors.append(
                f"Well {z.well_id}: zone exceeds y={domain_y} "
                f"(center_y={cy:.4f}, radius={z.radius:.4f})"
            )
    return errors


def max_non_overlapping_radius(
    zones: List[ProtectionZone], candidate: ProtectionZone
) -> float:
    """Compute the maximum radius for *candidate* without overlapping any zone.

    The candidate center is fixed; we find the largest radius that keeps the
    candidate circle from intersecting any existing zone circle.

    Parameters
    ----------
    zones : list of ProtectionZone
        Existing zones.
    candidate : ProtectionZone
        The candidate zone whose center is fixed; we find the max radius.

    Returns
    -------
    float
        Maximum non-overlapping radius. 0.0 if the candidate center is
        inside any existing zone (cannot fit at all).
        float('inf') if no existing zones are present.
    """
    if not zones:
        return float("inf")

    max_r = float("inf")
    for z in zones:
        dist = _distance(candidate.center, z.center)
        available = dist - z.radius
        if available <= 0:
            return 0.0
        max_r = min(max_r, available)

    return max_r
