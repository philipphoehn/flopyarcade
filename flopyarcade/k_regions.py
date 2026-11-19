"""A5: K-regions abstraction — formal capture zones with per-region K values.

This module provides a hydrogeological K-region abstraction that layers on top
of the existing helper-well ENVTYPEs (4s/5s/6s).  A K-region is a distinct
zone characterised by:

- a set of well IDs (capture or helper)
- a spatial extent (circular or rectangular bounds)
- its own hydraulic-conductivity tensor K

``KRegionManager`` is the primary API: add / remove / query regions, detect
overlaps, classify grid cells by nearest region, and serialise the full
configuration.

Usage::

    from flopyarcade.k_regions import KRegion, KRegionManager

    mgr = KRegionManager(dx=10.0, dy=10.0, ncol=20, nrow=15)
    mgr.add_region(KRegion(
        name="capture_zone_1",
        well_ids=[1],
        center=(100.0, 75.0),
        radius=30.0,
        K=(25.0, 25.0, 25.0),
    ))
    mgr.add_region(KRegion(
        name="recharge_zone",
        well_ids=[5, 6],
        center=(150.0, 50.0),
        radius=20.0,
        K=(5.0, 5.0, 5.0),
    ))

    # Classify grid cells
    labels = mgr.zone_membership()
    # labels shape: (nrow, ncol), values: region index or -1 (unassigned)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# KRegion
# ---------------------------------------------------------------------------

@dataclass
class KRegion:
    """A single hydrogeological K-region (capture / recharge / helper zone).

    Attributes:
        name: Human-readable identifier (unique within a manager).
        well_ids: List of 1-based well indices belonging to this region.
        center: (x, y) centre of the circular extent, in domain coordinates.
        radius: Circular radius in metres (the region is a disc).
        K: Hydraulic conductivity — scalar or (Kx, Ky, Kz) tuple.
        is_capture: True if this is a capture (pumping) zone; False for
            recharge or neutral zones.
    """
    name: str
    well_ids: List[int] = field(default_factory=list)
    center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 10.0
    K: Tuple[float, float, float] = (25.0, 25.0, 25.0)
    is_capture: bool = True

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"{self.name}: radius must be > 0, got {self.radius}")
        if len(self.K) == 1:
            self.K = (self.K[0], self.K[0], self.K[0])
        elif len(self.K) == 3:
            pass
        else:
            raise ValueError(
                f"{self.name}: K must be scalar or 3-tuple, got length {len(self.K)}"
            )
        if any(k <= 0 for k in self.K):
            raise ValueError(
                f"{self.name}: all K values must be > 0, got {self.K}"
            )
        for wid in self.well_ids:
            if wid < 1:
                raise ValueError(f"{self.name}: well_ids must be 1-based, got {wid}")

    def contains_point(self, x: float, y: float) -> bool:
        """Check whether (x, y) falls inside this region's disc."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        return (dx * dx + dy * dy) <= self.radius * self.radius

    def area(self) -> float:
        """Area of the circular region in m²."""
        return math.pi * self.radius ** 2

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'well_ids': list(self.well_ids),
            'center': list(self.center),
            'radius': self.radius,
            'K': list(self.K),
            'is_capture': self.is_capture,
        }

    @classmethod
    def from_dict(cls, d: dict) -> KRegion:
        return cls(
            name=d['name'],
            well_ids=list(d.get('well_ids', [])),
            center=tuple(d.get('center', (0.0, 0.0))),
            radius=d.get('radius', 10.0),
            K=tuple(d.get('K', (25.0, 25.0, 25.0))),
            is_capture=d.get('is_capture', True),
        )


# ---------------------------------------------------------------------------
# KRegionManager
# ---------------------------------------------------------------------------

class KRegionManager:
    """Manage a collection of KRegions on a simulation grid.

    Provides spatial queries, overlap detection, zone classification,
    and serialisation.

    Args:
        dx: Grid cell width (m).
        dy: Grid cell height (m).
        ncol: Number of grid columns.
        nrow: Number of grid rows.
        origin: (x, y) of the bottom-left grid corner. Defaults to (0, 0).
    """

    def __init__(
        self,
        dx: float = 10.0,
        dy: float = 10.0,
        ncol: int = 20,
        nrow: int = 15,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be > 0")
        if ncol < 1 or nrow < 1:
            raise ValueError("ncol and nrow must be >= 1")

        self.dx = dx
        self.dy = dy
        self.ncol = ncol
        self.nrow = nrow
        self.origin = origin
        self._regions: Dict[str, KRegion] = {}
        self._order: List[str] = []

    # -- properties ----------------------------------------------------------

    @property
    def regions(self) -> List[KRegion]:
        """Regions in insertion order."""
        return [self._regions[n] for n in self._order]

    @property
    def n_regions(self) -> int:
        return len(self._regions)

    # -- add / remove --------------------------------------------------------

    def add_region(self, region: KRegion) -> None:
        """Add a region. Raises if name already exists."""
        if region.name in self._regions:
            raise ValueError(f"Region '{region.name}' already exists")
        self._regions[region.name] = region
        self._order.append(region.name)

    def remove_region(self, name: str) -> KRegion:
        """Remove and return a region by name. Raises KeyError if missing."""
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found")
        region = self._regions.pop(name)
        self._order.remove(name)
        return region

    def get(self, name: str) -> KRegion:
        """Get a region by name."""
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found")
        return self._regions[name]

    def __contains__(self, name: str) -> bool:
        return name in self._regions

    def __len__(self) -> int:
        return len(self._regions)

    def __iter__(self):
        return iter(self.regions)

    # -- spatial queries -----------------------------------------------------

    def zone_membership(self) -> np.ndarray:
        """Classify each grid cell by nearest region.

        Returns:
            2D array of shape (nrow, ncol) with integer region indices
            (0-based) or -1 for cells not inside any region.
        """
        labels = np.full((self.nrow, self.ncol), -1, dtype=np.int32)
        if not self._regions:
            return labels

        # Grid cell centres
        xs = self.origin[0] + (np.arange(self.ncol) + 0.5) * self.dx
        ys = self.origin[1] + (np.arange(self.nrow) + 0.5) * self.dy
        X, Y = np.meshgrid(xs, ys)  # shape (nrow, ncol)

        # For each cell, find the nearest region whose disc contains it.
        # If multiple regions contain the cell, pick the nearest centre.
        best_dist = np.full((self.nrow, self.ncol), np.inf, dtype=np.float64)

        for idx, name in enumerate(self._order):
            reg = self._regions[name]
            cx, cy = reg.center
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            inside = dist <= reg.radius
            # Pick nearest region (smallest distance to centre)
            better = inside & (dist < best_dist)
            best_dist[better] = dist[better]
            labels[better] = idx

        return labels

    def region_at(self, x: float, y: float) -> Optional[int]:
        """Return 0-based region index at point (x, y), or None."""
        best_idx = -1
        best_dist = float('inf')
        for idx, name in enumerate(self._order):
            reg = self._regions[name]
            if reg.contains_point(x, y):
                d = (x - reg.center[0]) ** 2 + (y - reg.center[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
        return best_idx if best_idx >= 0 else None

    # -- overlap detection ---------------------------------------------------

    def detect_overlaps(self) -> List[Tuple[str, str, float]]:
        """Detect pairwise region overlaps.

        Returns:
            List of (name_a, name_b, overlap_fraction) where overlap_fraction
            is the estimated fraction of the smaller region that overlaps the
            larger one (using circle-circle intersection approximation).
        """
        overlaps = []
        names = self._order
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ra = self._regions[names[i]]
                rb = self._regions[names[j]]
                d = math.sqrt(
                    (ra.center[0] - rb.center[0]) ** 2
                    + (ra.center[1] - rb.center[1]) ** 2
                )
                if d < ra.radius + rb.radius:
                    # Estimate overlap: fraction of smaller circle
                    r_min = min(ra.radius, rb.radius)
                    r_max = max(ra.radius, rb.radius)
                    # Simple approximation: overlap fraction ≈ 1 - d/(r_min+r_max)
                    frac = max(0.0, 1.0 - d / (r_min + r_max))
                    overlaps.append((names[i], names[j], round(frac, 4)))
        return overlaps

    # -- well assignment -----------------------------------------------------

    def wells_per_region(self) -> Dict[str, List[int]]:
        """Map region name → list of well IDs."""
        return {name: list(self._regions[name].well_ids) for name in self._order}

    def region_for_well(self, well_id: int) -> Optional[str]:
        """Find the region containing a given well ID (first match)."""
        for name in self._order:
            if well_id in self._regions[name].well_ids:
                return name
        return None

    # -- K field construction ------------------------------------------------

    def build_K_field(self) -> np.ndarray:
        """Build a 3D K field (nrow, ncol, 3) from zone membership.

        Cells not in any region get K = (25, 25, 25) (default homogeneous).

        Returns:
            Array of shape (nrow, ncol, 3) with Kx, Ky, Kz per cell.
        """
        K_default = np.array([25.0, 25.0, 25.0])
        K_field = np.tile(K_default, (self.nrow, self.ncol, 1))
        labels = self.zone_membership()

        for idx, name in enumerate(self._order):
            reg = self._regions[name]
            mask = labels == idx
            if np.any(mask):
                K_field[mask] = np.array(reg.K)

        return K_field

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            'dx': self.dx,
            'dy': self.dy,
            'ncol': self.ncol,
            'nrow': self.nrow,
            'origin': list(self.origin),
            'regions': [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> KRegionManager:
        mgr = cls(
            dx=d.get('dx', 10.0),
            dy=d.get('dy', 10.0),
            ncol=d.get('ncol', 20),
            nrow=d.get('nrow', 15),
            origin=tuple(d.get('origin', (0.0, 0.0))),
        )
        for rd in d.get('regions', []):
            mgr.add_region(KRegion.from_dict(rd))
        return mgr

    def __repr__(self) -> str:
        return (
            f"KRegionManager(n={len(self)}, "
            f"grid={self.nrow}x{self.ncol}, dx={self.dx}, dy={self.dy})"
        )
