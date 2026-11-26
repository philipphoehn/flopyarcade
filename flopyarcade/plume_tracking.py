"""V1: Plume front tracking — distance-from-source over time; box-plot across episodes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict


@dataclass
class PlumePoint:
    """A single plume-front measurement."""
    t: int
    distance: float
    episode: int


class PlumeTracker:
    """Track the plume front (furthest particle from source) over time and episodes."""

    def __init__(self, source_pos: tuple[float, float] = (0.0, 0.0)):
        self.source_pos: Tuple[float, float] = tuple(source_pos)
        self._points: List[PlumePoint] = []

    # ------------------------------------------------------------------ recording
    def record(self, t: int, positions: list[tuple[float, float]], episode: int) -> None:
        """Compute distance from source for each particle; store the furthest (plume front)."""
        if not positions:
            raise ValueError("positions list must not be empty")
        max_dist = 0.0
        sx, sy = self.source_pos
        for px, py in positions:
            d = math.hypot(px - sx, py - sy)
            if d > max_dist:
                max_dist = d
        self._points.append(PlumePoint(t=t, distance=max_dist, episode=episode))

    # ------------------------------------------------------------------ accessors
    def front_positions(self) -> list[PlumePoint]:
        """Return all recorded front positions."""
        return list(self._points)

    def max_distance(self) -> float:
        if not self._points:
            return 0.0
        return max(p.distance for p in self._points)

    def min_distance(self) -> float:
        if not self._points:
            return 0.0
        return min(p.distance for p in self._points)

    def mean_distance(self) -> float:
        if not self._points:
            return 0.0
        return sum(p.distance for p in self._points) / len(self._points)

    # ------------------------------------------------------------------ speed
    def front_speed(self) -> dict:
        """Return {mean_speed, max_speed, min_speed} from consecutive points."""
        if len(self._points) < 2:
            return {"mean_speed": 0.0, "max_speed": 0.0, "min_speed": 0.0}
        speeds = []
        for i in range(1, len(self._points)):
            dt = self._points[i].t - self._points[i - 1].t
            if dt == 0:
                continue
            speed = (self._points[i].distance - self._points[i - 1].distance) / dt
            speeds.append(speed)
        if not speeds:
            return {"mean_speed": 0.0, "max_speed": 0.0, "min_speed": 0.0}
        return {
            "mean_speed": sum(speeds) / len(speeds),
            "max_speed": max(speeds),
            "min_speed": min(speeds),
        }

    # ------------------------------------------------------------------ per-episode
    def per_episode_stats(self) -> Dict[int, dict]:
        """Per-episode {min_dist, max_dist, mean_dist, n_points}."""
        episodes: Dict[int, list[float]] = {}
        for p in self._points:
            episodes.setdefault(p.episode, []).append(p.distance)
        result = {}
        for ep in sorted(episodes.keys()):
            dists = episodes[ep]
            result[ep] = {
                "min_dist": min(dists),
                "max_dist": max(dists),
                "mean_dist": sum(dists) / len(dists),
                "n_points": len(dists),
            }
        return result

    # ------------------------------------------------------------------ boxplot
    @staticmethod
    def _quartiles(sorted_vals: list[float]) -> Tuple[float, float, float]:
        """Return (q1, median, q3) using linear interpolation."""
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0], sorted_vals[0], sorted_vals[0]

        def _pct(p: float) -> float:
            idx = (n - 1) * p
            lo = int(idx)
            hi = lo + 1 if lo + 1 < n else lo
            frac = idx - lo
            return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

        return _pct(0.25), _pct(0.50), _pct(0.75)

    def boxplot_data(self) -> dict:
        """Return boxplot-ready dict for matplotlib.

        Keys: episodes, min, q1, median, q3, max, outliers
        """
        episodes: Dict[int, list[float]] = {}
        for p in self._points:
            episodes.setdefault(p.episode, []).append(p.distance)

        ep_ids = sorted(episodes.keys())
        mins, q1s, medians, q3s, maxs = [], [], [], [], []
        outliers: Dict[int, list[float]] = {ep: [] for ep in ep_ids}

        for ep in ep_ids:
            vals = sorted(episodes[ep])
            mn, mx = vals[0], vals[-1]
            q1, med, q3 = self._quartiles(vals)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            in_range = [v for v in vals if lower_fence <= v <= upper_fence]
            if in_range:
                mn = min(in_range)
                mx = max(in_range)
            outliers[ep] = [v for v in vals if v < lower_fence or v > upper_fence]
            mins.append(mn)
            q1s.append(q1)
            medians.append(med)
            q3s.append(q3)
            maxs.append(mx)

        return {
            "episodes": ep_ids,
            "min": mins,
            "q1": q1s,
            "median": medians,
            "q3": q3s,
            "max": maxs,
            "outliers": outliers,
        }

    # ------------------------------------------------------------------ trend
    def trend(self) -> dict:
        """Linear regression on (t, distance) pairs: {slope, intercept, r_squared, n_points}."""
        n = len(self._points)
        if n < 2:
            return {"slope": 0.0,
                    "intercept": self._points[0].distance if n == 1 else 0.0,
                    "r_squared": 0.0, "n_points": n}

        xs = [p.t for p in self._points]
        ys = [p.distance for p in self._points]
        x_bar = sum(xs) / n
        y_bar = sum(ys) / n

        ss_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
        ss_xx = sum((x - x_bar) ** 2 for x in xs)
        ss_yy = sum((y - y_bar) ** 2 for y in ys)

        if ss_xx == 0:
            return {"slope": 0.0, "intercept": y_bar, "r_squared": 0.0, "n_points": n}

        slope = ss_xy / ss_xx
        intercept = y_bar - slope * x_bar
        if ss_xx * ss_yy > 0:
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)
        else:
            r_squared = 0.0

        return {"slope": slope, "intercept": intercept, "r_squared": r_squared, "n_points": n}

    # ------------------------------------------------------------------ serialisation
    def save_json(self, path: str) -> None:
        """Save tracker state to JSON file."""
        data = {
            "source_pos": list(self.source_pos),
            "points": [asdict(p) for p in self._points],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PlumeTracker":
        """Load tracker from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        tracker = cls(source_pos=tuple(data["source_pos"]))
        tracker._points = [PlumePoint(**p) for p in data["points"]]
        return tracker


# ------------------------------------------------------------------ module-level helpers
def compare_plumes(tracker_a: PlumeTracker, tracker_b: PlumeTracker) -> dict:
    """Compare two plume trackers; return summary dict."""
    a_max = tracker_a.max_distance()
    b_max = tracker_b.max_distance()
    a_mean = tracker_a.mean_distance()
    b_mean = tracker_b.mean_distance()
    faster = "A" if a_mean >= b_mean else "B"
    speed_ratio = (a_mean / b_mean) if b_mean != 0 else float("inf")
    return {
        "a_max": a_max,
        "b_max": b_max,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "faster": faster,
        "speed_ratio": speed_ratio,
    }


def format_plume_report(tracker: PlumeTracker) -> str:
    """Human-readable text summary of a PlumeTracker."""
    lines = [
        "=" * 50,
        "  Plume Front Tracking Report",
        "=" * 50,
        f"  Source position : {tracker.source_pos}",
        f"  Points recorded : {len(tracker.front_positions())}",
        f"  Max distance    : {tracker.max_distance():.4f}",
        f"  Min distance    : {tracker.min_distance():.4f}",
        f"  Mean distance   : {tracker.mean_distance():.4f}",
        "",
        "  Speed statistics:",
    ]
    speed = tracker.front_speed()
    lines += [
        f"    Mean speed : {speed['mean_speed']:.6f}",
        f"    Max  speed : {speed['max_speed']:.6f}",
        f"    Min  speed : {speed['min_speed']:.6f}",
    ]
    trend = tracker.trend()
    lines += [
        "",
        "  Trend (linear fit):",
        f"    Slope     : {trend['slope']:.6f}",
        f"    Intercept : {trend['intercept']:.6f}",
        f"    R²        : {trend['r_squared']:.6f}",
    ]
    eps = tracker.per_episode_stats()
    lines += ["", f"  Episodes ({len(eps)}):"]
    for ep, stats in eps.items():
        lines.append(
            f"    Ep {ep}: min={stats['min_dist']:.4f}  "
            f"max={stats['max_dist']:.4f}  mean={stats['mean_dist']:.4f}  "
            f"n={stats['n_points']}"
        )
    lines.append("=" * 50)
    return chr(10).join(lines)
