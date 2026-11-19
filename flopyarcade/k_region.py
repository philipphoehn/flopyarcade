"""A5: K-region abstraction — generalise helper wells into formal capture zones.

The existing ENVTYPE 4/5 environments use helper wells identified by
``wellX{i}`, ``wellY{i}`, ``wellZ{i}`, ``wellQ{i}`` dict keys. This module
provides a clean, reusable abstraction:

* **KRegion** — a single capture zone (position, radius, flow rate, label).
* **KRegionManager** — a collection of KRegions with spatial queries,
  reward aggregation, and serialization.

The abstraction is backward-compatible: a single KRegion with ``n=1``
reproduces the single-well behaviour, and the manager can wrap the
existing ``helperWells`` dict without modification.

Usage::

    from flopyarcade.k_region import KRegion, KRegionManager

    mgr = KRegionManager()
    mgr.add_region(KRegion(
        label="helper1",
        position=(160.0, 320.0, 20.0),
        radius=2.0,
        flow_rate=300.0,
    ))

    # Query: is a point inside any region?
    region = mgr.nearest_region((160.0, 321.0, 20.0))
    if region and region.contains((160.0, 321.0, 20.0)):
        reward = region.reward_weight

    # Serialize
    data = mgr.to_dict()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Single K-region
# ---------------------------------------------------------------------------

@dataclass
class KRegion:
    """A single capture zone (generalised helper well).

    Attributes:
        label: Human-readable identifier (e.g. ``"helper1"``, ``"main"``).
        position: 3D centre of the capture zone ``(x, y, z)``.
        radius: Capture radius in model units. A particle at distance
            <latex>d \le r</latex> is considered inside the region.
        flow_rate: Pumping rate <latex>Q</latex> (m<sup>3</sup>/d).
            Positive = extraction (capture), negative = recharge.
        reward_weight: Multiplier applied to the base reward when a
            particle is captured by this region. Defaults to 1.0.
        active: Whether this region is currently active. Inactive regions
            are excluded from spatial queries and reward computation.
    """
    label: str
    position: Tuple[float, float, float]
    radius: float
    flow_rate: float = 0.0
    reward_weight: float = 1.0
    active: bool = True

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError(
                f"KRegion '{self.label}': radius must be >= 0, got {self.radius}"
            )
        if len(self.position) != 3:
            raise ValueError(
                f"KRegion '{self.label}': position must be (x, y, z), got {self.position}"
            )

    # -- Spatial queries --

    def distance(self, point: Tuple[float, float, float]) -> float:
        """Euclidean distance from *point* to the region centre."""
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        dz = point[2] - self.position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def contains(self, point: Tuple[float, float, float]) -> bool:
        """True if *point* is within the capture radius."""
        if not self.active:
            return False
        return self.distance(point) <= self.radius

    def in_xy(self, point: Tuple[float, float, float]) -> bool:
        """True if the XY projection is within the capture radius (ignores Z).

        Useful for top-view rendering where Z is irrelevant.
        """
        if not self.active:
            return False
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        return math.sqrt(dx * dx + dy * dy) <= self.radius

    # -- Reward --

    def compute_reward(self, base_reward: float = 1.0) -> float:
        """Reward contribution when a particle is captured by this region."""
        return self.reward_weight * base_reward

    # -- Serialization --

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "label": self.label,
            "position": list(self.position),
            "radius": self.radius,
            "flow_rate": self.flow_rate,
            "reward_weight": self.reward_weight,
            "active": self.active,
        }

    def __repr__(self) -> str:
        return (
            f"KRegion(label='{self.label}', "
            f"pos={tuple(round(v, 2) for v in self.position)}, "
            f"r={self.radius}, Q={self.flow_rate})"
        )


# ---------------------------------------------------------------------------
# KRegionManager
# ---------------------------------------------------------------------------

class KRegionManager:
    """Manages a collection of KRegions for a single environment instance.

    Provides spatial queries, reward aggregation, and serialization.
    Designed to wrap or replace the legacy ``helperWells`` dict pattern.

    Example::

        mgr = KRegionManager()
        mgr.add_region(KRegion(label="main", position=(80, 320, 20), radius=sqrt(8)))
        mgr.add_region(KRegion(label="helper1", position=(160, 200, 20), radius=2.0))

        # Which region does this particle belong to?
        region = mgr.nearest_region((80, 321, 20))
        # (80, 321, 20) is 1 unit from main → main wins

        # Aggregate reward if particles hit both regions
        total = mgr.total_reward(base=1.0)
    """

    def __init__(self) -> None:
        self._regions: List[KRegion] = []
        self._labels: Dict[str, int] = {}  # label → index

    # -- Adding / removing --

    def add_region(self, region: KRegion) -> None:
        """Add a KRegion.

        Raises:
            ValueError: If a region with the same label already exists.
        """
        if region.label in self._labels:
            raise ValueError(
                f"KRegion with label '{region.label}' already exists"
            )
        self._regions.append(region)
        self._labels[region.label] = len(self._regions) - 1

    def remove_region(self, label: str) -> KRegion:
        """Remove and return a region by label.

        Raises:
            KeyError: If label not found.
        """
        if label not in self._labels:
            raise KeyError(f"KRegion '{label}' not found")
        idx = self._labels.pop(label)
        removed = self._regions.pop(idx)
        # Rebuild index map (cheap; region count is small)
        self._labels = {r.label: i for i, r in enumerate(self._regions)}
        return removed

    def get(self, label: str) -> KRegion:
        """Get a region by label.

        Raises:
            KeyError: If label not found.
        """
        if label not in self._labels:
            raise KeyError(f"KRegion '{label}' not found")
        return self._regions[self._labels[label]]

    # -- Collection access --

    @property
    def regions(self) -> List[KRegion]:
        """All regions (including inactive)."""
        return list(self._regions)

    @property
    def active_regions(self) -> List[KRegion]:
        """Only active regions."""
        return [r for r in self._regions if r.active]

    @property
    def region_count(self) -> int:
        return len(self._regions)

    def __len__(self) -> int:
        return len(self._regions)

    def __contains__(self, label: str) -> bool:
        return label in self._labels

    # -- Spatial queries --

    def nearest_region(
        self, point: Tuple[float, float, float]
    ) -> Optional[KRegion]:
        """Return the active region whose centre is closest to *point*.

        Returns None if no active regions exist.
        """
        best: Optional[KRegion] = None
        best_dist = float("inf")
        for r in self.active_regions:
            d = r.distance(point)
            if d < best_dist:
                best_dist = d
                best = r
        return best

    def region_at(
        self, point: Tuple[float, float, float]
    ) -> Optional[KRegion]:
        """Return the first active region that contains *point*, or None.

        Iterates in insertion order; the first match wins.
        """
        for r in self.active_regions:
            if r.contains(point):
                return r
        return None

    def region_at_xy(
        self, point: Tuple[float, float, float]
    ) -> Optional[KRegion]:
        """Like :meth:`region_at` but ignores Z (XY projection)."""
        for r in self.active_regions:
            if r.in_xy(point):
                return r
        return None

    def all_containing(
        self, point: Tuple[float, float, float]
    ) -> List[KRegion]:
        """All active regions that contain *point."""
        return [r for r in self.active_regions if r.contains(point)]

    # -- Reward aggregation --

    def total_reward(self, base_reward: float = 1.0) -> float:
        """Sum of reward weights across all active regions.

        Used when a particle is captured by *any* region (max),
        or when multiple particles hit different regions (sum).
        """
        return sum(r.compute_reward(base_reward) for r in self.active_regions)

    def max_reward(self, base_reward: float = 1.0) -> float:
        """Maximum single-region reward (for the 'best capture' case)."""
        active = self.active_regions
        if not active:
            return 0.0
        return max(r.compute_reward(base_reward) for r in active)

    # -- Flow rate aggregation --

    def total_flow_rate(self) -> float:
        """Net flow rate across all active regions."""
        return sum(r.flow_rate for r in self.active_regions)

    def net_extraction(self) -> float:
        """Total extraction (positive Q) across active regions."""
        return sum(r.flow_rate for r in self.active_regions if r.flow_rate > 0)

    def net_recharge(self) -> float:
        """Total recharge (negative Q, returned as positive magnitude)."""
        return abs(sum(r.flow_rate for r in self.active_regions if r.flow_rate < 0))

    # -- Activation toggling --

    def activate(self, label: str) -> None:
        """Activate a region by label."""
        self.get(label).active = True

    def deactivate(self, label: str) -> None:
        """Deactivate a region by label."""
        self.get(label).active = False

    def set_flow_rate(self, label: str, flow_rate: float) -> None:
        """Update the flow rate of a region."""
        self.get(label).flow_rate = flow_rate

    # -- Bulk operations --

    def clear(self) -> None:
        """Remove all regions."""
        self._regions.clear()
        self._labels.clear()

    @classmethod
    def from_helper_wells_dict(
        cls, helper_wells: dict, base_radius: float = 2.0
    ) -> "KRegionManager":
        """Build a KRegionManager from the legacy ``helperWells`` dict.

        Expected keys: ``wellX{i}`, ``wellY{i}`, ``wellZ{i}`, ``wellQ{i}``
        where *i* is 1-based.

        Args:
            helper_wells: The legacy dict.
            base_radius: Default capture radius (from ``helperWellRadius``).

        Returns:
            A new KRegionManager with one region per helper well.
        """
        mgr = cls()
        i = 1
        while f"wellX{i}" in helper_wells:
            pos = (
                float(helper_wells[f"wellX{i}"]),
                float(helper_wells[f"wellY{i}"]),
                float(helper_wells[f"wellZ{i}"]),
            )
            q = float(helper_wells.get(f"wellQ{i}", 0.0))
            mgr.add_region(
                KRegion(
                    label=f"helper{i}",
                    position=pos,
                    radius=base_radius,
                    flow_rate=q,
                )
            )
            i += 1
        return mgr

    # -- Serialization --

    def to_dict(self) -> dict:
        """Serialize all regions to a plain dict."""
        return {
            "region_count": self.region_count,
            "regions": [r.to_dict() for r in self._regions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KRegionManager":
        """Reconstruct a KRegionManager from :meth:`to_dict` output."""
        mgr = cls()
        for rd in data.get("regions", []):
            mgr.add_region(
                KRegion(
                    label=rd["label"],
                    position=tuple(rd["position"]),  # type: ignore[arg-type]
                    radius=rd["radius"],
                    flow_rate=rd.get("flow_rate", 0.0),
                    reward_weight=rd.get("reward_weight", 1.0),
                    active=rd.get("active", True),
                )
            )
        return mgr

    def __repr__(self) -> str:
        return (
            f"KRegionManager(n={self.region_count}, "
            f"active={len(self.active_regions)}, "
            f"labels={[r.label for r in self._regions]})"
        )

    def __iter__(self):
        return iter(self._regions)

