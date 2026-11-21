"""Benchmark drift tracking (ROADMAP V8).

Provides score-over-time tracking with CI bands, alert detection on
drops exceeding a threshold, trend analysis, and multi-envtype comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class DriftPoint:
    """A single benchmark observation."""
    date: str
    score: float
    ci_low: float = 0.0
    ci_high: float = 0.0
    git_sha: Optional[str] = None
    alert: bool = False


# ─── Tracker ─────────────────────────────────────────────────────────────────

class DriftTracker:
    """Tracks benchmark scores over time with alert and trend analysis."""

    def __init__(self, envtype: str, alert_threshold_pct: float = 5.0) -> None:
        self.envtype = envtype
        self.alert_threshold_pct = alert_threshold_pct
        self._points: list[DriftPoint] = []

    # -- mutation ---------------------------------------------------------

    def add_point(
        self,
        date: str,
        score: float,
        ci_low: Optional[float] = None,
        ci_high: Optional[float] = None,
        git_sha: Optional[str] = None,
    ) -> None:
        """Append a data point; auto-sets alert if score drops > threshold from previous."""
        alert = False
        if self._points:
            prev_score = self._points[-1].score
            if prev_score != 0:
                drop_pct = (prev_score - score) / abs(prev_score) * 100.0
                if drop_pct > self.alert_threshold_pct:
                    alert = True

        point = DriftPoint(
            date=date,
            score=score,
            ci_low=ci_low if ci_low is not None else 0.0,
            ci_high=ci_high if ci_high is not None else 0.0,
            git_sha=git_sha,
            alert=alert,
        )
        self._points.append(point)

    # -- accessors --------------------------------------------------------

    def points(self) -> list[DriftPoint]:
        return list(self._points)

    def n_points(self) -> int:
        return len(self._points)

    def latest_score(self) -> float:
        if not self._points:
            raise ValueError("No data points recorded.")
        return self._points[-1].score

    def best_score(self) -> float:
        if not self._points:
            raise ValueError("No data points recorded.")
        return max(p.score for p in self._points)

    def worst_score(self) -> float:
        if not self._points:
            raise ValueError("No data points recorded.")
        return min(p.score for p in self._points)

    def total_decline_pct(self) -> float:
        if len(self._points) < 2:
            return 0.0
        first = self._points[0].score
        last = self._points[-1].score
        if first == 0:
            return 0.0
        return (first - last) / abs(first) * 100.0

    def detect_alerts(self) -> list[DriftPoint]:
        return [p for p in self._points if p.alert]

    # -- analysis ---------------------------------------------------------

    def trend(self) -> dict:
        """Linear regression on (index, score). Returns slope, intercept, r_squared, declining, improving."""
        n = len(self._points)
        if n < 2:
            return {"slope": 0.0, "intercept": self._points[0].score if n else 0.0,
                    "r_squared": 1.0, "declining": False, "improving": False}

        xs = list(range(n))
        ys = [p.score for p in self._points]

        x_mean = sum(xs) / n
        y_mean = sum(ys) / n

        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        ss_xx = sum((x - x_mean) ** 2 for x in xs)
        ss_yy = sum((y - y_mean) ** 2 for y in ys)

        if ss_xx == 0:
            slope = 0.0
        else:
            slope = ss_xy / ss_xx

        intercept = y_mean - slope * x_mean

        if ss_yy == 0:
            r_squared = 1.0
        else:
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "declining": slope < 0,
            "improving": slope > 0,
        }

    def ci_band_data(self) -> dict:
        """Return data suitable for plotting a CI band chart."""
        return {
            "dates": [p.date for p in self._points],
            "scores": [p.score for p in self._points],
            "ci_low": [p.ci_low for p in self._points],
            "ci_high": [p.ci_high for p in self._points],
            "alerts": [p.date for p in self._points if p.alert],
        }

    def rolling_mean(self, window: int = 5) -> list[float]:
        """Simple rolling mean over scores; returns list of same length as points."""
        if not self._points:
            return []
        scores = [p.score for p in self._points]
        n = len(scores)
        result = []
        for i in range(n):
            start = max(0, i - window + 1)
            window_slice = scores[start:i + 1]
            result.append(sum(window_slice) / len(window_slice))
        return result

    # -- serialization ----------------------------------------------------

    def save_json(self, path: str) -> None:
        data = {
            "envtype": self.envtype,
            "alert_threshold_pct": self.alert_threshold_pct,
            "points": [asdict(p) for p in self._points],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "DriftTracker":
        with open(path, "r") as f:
            data = json.load(f)
        tracker = cls(
            envtype=data["envtype"],
            alert_threshold_pct=data.get("alert_threshold_pct", 5.0),
        )
        for pd in data["points"]:
            point = DriftPoint(
                date=pd["date"],
                score=pd["score"],
                ci_low=pd.get("ci_low", 0.0),
                ci_high=pd.get("ci_high", 0.0),
                git_sha=pd.get("git_sha"),
                alert=pd.get("alert", False),
            )
            tracker._points.append(point)
        return tracker


# ─── Comparison & reporting ──────────────────────────────────────────────────

def compare_drift(trackers: dict[str, DriftTracker]) -> dict:
    """Compare drift across multiple envtypes."""
    result = {}
    for envtype, tracker in trackers.items():
        result[envtype] = {
            "envtype": envtype,
            "latest": tracker.latest_score() if tracker.n_points() > 0 else None,
            "best": tracker.best_score() if tracker.n_points() > 0 else None,
            "decline_pct": tracker.total_decline_pct(),
            "n_alerts": len(tracker.detect_alerts()),
        }
    return result


def format_drift_report(tracker: DriftTracker) -> str:
    """Human-readable text summary."""
    lines = [
        f"Benchmark Drift Report — {tracker.envtype}",
        f"  Points: {tracker.n_points()}",
    ]
    if tracker.n_points() > 0:
        lines.append(f"  Latest score: {tracker.latest_score():.4f}")
        lines.append(f"  Best score: {tracker.best_score():.4f}")
        lines.append(f"  Worst score: {tracker.worst_score():.4f}")
        lines.append(f"  Total decline: {tracker.total_decline_pct():.2f}%")

    trend = tracker.trend()
    direction = "declining" if trend["declining"] else ("improving" if trend["improving"] else "flat")
    lines.append(f"  Trend: {direction} (slope={trend['slope']:.4f}, R²={trend['r_squared']:.4f})")

    alerts = tracker.detect_alerts()
    if alerts:
        lines.append(f"  Alerts: {len(alerts)}")
        for a in alerts:
            lines.append(f"    - {a.date}: score={a.score:.4f}")
    else:
        lines.append("  Alerts: none")

    return "\n".join(lines)


def alert_summary(points: list[DriftPoint]) -> dict:
    """Summarize alerts across a list of DriftPoints."""
    alerts = [p for p in points if p.alert]
    if not alerts:
        return {
            "n_alerts": 0,
            "first_alert_date": None,
            "last_alert_date": None,
            "max_drop_pct": 0.0,
        }
    max_drop = 0.0
    for i, p in enumerate(points):
        if p.alert and i > 0 and points[i - 1].score != 0:
            drop = (points[i - 1].score - p.score) / abs(points[i - 1].score) * 100.0
            if drop > max_drop:
                max_drop = drop
    return {
        "n_alerts": len(alerts),
        "first_alert_date": alerts[0].date,
        "last_alert_date": alerts[-1].date,
        "max_drop_pct": max_drop,
    }

