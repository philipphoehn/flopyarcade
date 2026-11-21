"""Observation coverage heatmap – fraction of action space visited per generation (ROADMAP P9).

Tracks which actions an RL agent uses across training generations,
computes coverage metrics, gap analysis, and export-ready heatmap data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ActionBucket:
    """A single action's usage statistics."""

    action: str
    count: int
    fraction: float


@dataclass
class GenerationCoverage:
    """Coverage snapshot for one generation."""

    generation: int
    buckets: list[ActionBucket]
    total_actions: int
    n_unique_actions: int
    coverage: float  # fraction of action space visited


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CoverageTracker:
    """Track action-space coverage across training generations."""

    def __init__(self, action_space: list[str] = None):
        if action_space is None:
            action_space = ['up', 'down', 'left', 'right', 'keep']
        self.action_space: list[str] = list(action_space)
        self._generations: list[GenerationCoverage] = []

    # -- recording ----------------------------------------------------------

    def record_generation(self, generation: int, actions: list[str]) -> GenerationCoverage:
        """Record the actions taken in one generation and compute coverage."""
        n = len(self.action_space)
        counts: dict[str, int] = {a: 0 for a in self.action_space}
        for act in actions:
            if act in counts:
                counts[act] += 1
            # silently ignore unknown actions

        total = len(actions) if actions else 1
        buckets = [
            ActionBucket(
                action=a,
                count=counts[a],
                fraction=counts[a] / total if total else 0.0,
            )
            for a in self.action_space
        ]
        n_unique = sum(1 for b in buckets if b.count > 0)
        coverage = n_unique / n if n else 0.0

        gc = GenerationCoverage(
            generation=generation,
            buckets=buckets,
            total_actions=len(actions),
            n_unique_actions=n_unique,
            coverage=coverage,
        )
        self._generations.append(gc)
        return gc

    # -- accessors ----------------------------------------------------------

    def generations(self) -> list[GenerationCoverage]:
        return list(self._generations)

    def n_generations(self) -> int:
        return len(self._generations)

    def overall_coverage(self) -> float:
        if not self._generations:
            return 0.0
        return sum(g.coverage for g in self._generations) / len(self._generations)

    # -- trend --------------------------------------------------------------

    def coverage_trend(self) -> dict:
        """Linear-regression slope on per-generation coverage."""
        n = len(self._generations)
        if n < 2:
            return {"first": self._generations[0].coverage if self._generations else 0.0,
                    "last": self._generations[-1].coverage if self._generations else 0.0,
                    "improving": False,
                    "slope": 0.0}

        xs = list(range(n))
        ys = [g.coverage for g in self._generations]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den else 0.0

        return {
            "first": ys[0],
            "last": ys[-1],
            "improving": slope > 0,
            "slope": slope,
        }

    # -- per-action stats ---------------------------------------------------

    def _total_counts(self) -> dict[str, int]:
        totals: dict[str, int] = {a: 0 for a in self.action_space}
        for gen in self._generations:
            for b in gen.buckets:
                totals[b.action] = totals.get(b.action, 0) + b.count
        return totals

    def _total_actions_all_gens(self) -> int:
        return sum(g.total_actions for g in self._generations)

    def most_used_action(self) -> str:
        totals = self._total_counts()
        if not self._generations:
            return self.action_space[0] if self.action_space else ""
        return max(totals, key=totals.get)

    def least_used_action(self) -> str:
        totals = self._total_counts()
        if not self._generations:
            return self.action_space[0] if self.action_space else ""
        return min(totals, key=totals.get)

    def action_distribution(self) -> dict[str, float]:
        total = self._total_actions_all_gens()
        if not total:
            return {a: 0.0 for a in self.action_space}
        totals = self._total_counts()
        return {a: totals[a] / total for a in self.action_space}

    def gap_analysis(self, threshold: float = 0.1) -> list[dict]:
        """Actions whose overall fraction is below *threshold*."""
        if not self._generations:
            return []
        dist = self.action_distribution()
        totals = self._total_counts()
        gaps = []
        for a in self.action_space:
            if dist[a] < threshold:
                gens_used = sum(
                    1 for g in self._generations
                    if any(b.action == a and b.count > 0 for b in g.buckets)
                )
                gaps.append({
                    "action": a,
                    "total_count": totals[a],
                    "fraction": dist[a],
                    "generations_used": gens_used,
                })
        return gaps

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "action_space": self.action_space,
            "generations": [asdict(g) for g in self._generations],
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "CoverageTracker":
        data = json.loads(Path(path).read_text())
        tracker = cls(action_space=data["action_space"])
        for gen_data in data["generations"]:
            buckets = [ActionBucket(**b) for b in gen_data["buckets"]]
            gc = GenerationCoverage(
                generation=gen_data["generation"],
                buckets=buckets,
                total_actions=gen_data["total_actions"],
                n_unique_actions=gen_data["n_unique_actions"],
                coverage=gen_data["coverage"],
            )
            tracker._generations.append(gc)
        return tracker


# ---------------------------------------------------------------------------
# Comparison & reporting
# ---------------------------------------------------------------------------

def compare_coverage(tracker_a: CoverageTracker, tracker_b: CoverageTracker) -> dict:
    """Per-generation coverage difference where both trackers have data."""
    gen_a = {g.generation: g for g in tracker_a.generations()}
    gen_b = {g.generation: g for g in tracker_b.generations()}

    common = sorted(set(gen_a) & set(gen_b))
    per_gen = {
        str(g): {"a": gen_a[g].coverage, "b": gen_b[g].coverage,
                 "diff": gen_a[g].coverage - gen_b[g].coverage}
        for g in common
    }
    overall_a = tracker_a.overall_coverage()
    overall_b = tracker_b.overall_coverage()

    return {
        "per_generation": per_gen,
        "overall_a": overall_a,
        "overall_b": overall_b,
        "overall_diff": overall_a - overall_b,
    }


def format_coverage_report(tracker: CoverageTracker) -> str:
    """Human-readable text table."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("COVERAGE HEATMAP REPORT")
    lines.append("=" * 60)
    lines.append(f"Action space: {', '.join(tracker.action_space)}")
    lines.append(f"Generations:  {tracker.n_generations()}")
    lines.append(f"Overall coverage: {tracker.overall_coverage():.4f}")
    trend = tracker.coverage_trend()
    lines.append(f"Trend: first={trend['first']:.4f} last={trend['last']:.4f} "
                 f"improving={trend['improving']} slope={trend['slope']:.6f}")
    lines.append("")

    # per-generation table
    if tracker.n_generations():
        lines.append(f"{'Gen':>6}  {'Cov':>8}  {'N_uniq':>8}  {'Total':>8}  Actions")
        lines.append("-" * 50)
        for g in tracker.generations():
            acts = ", ".join(f"{b.action}={b.fraction:.2f}" for b in g.buckets)
            lines.append(f"{g.generation:>6}  {g.coverage:>8.4f}  {g.n_unique_actions:>8}  "
                         f"{g.total_actions:>8}  {acts}")
        lines.append("")

    # action distribution
    dist = tracker.action_distribution()
    lines.append("Action distribution (all generations):")
    for a in tracker.action_space:
        lines.append(f"  {a:>8}: {dist[a]:.4f}")
    lines.append("")

    # gaps
    gaps = tracker.gap_analysis()
    if gaps:
        lines.append(f"Gap analysis (coverage < 0.1):")
        for g in gaps:
            lines.append(f"  {g['action']}: count={g['total_count']} "
                         f"fraction={g['fraction']:.4f} gens_used={g['generations_used']}")
    else:
        lines.append("No gaps (all actions above 0.1 threshold).")
    lines.append("=" * 60)
    return "\n".join(lines)


def heatmap_data(tracker: CoverageTracker) -> dict:
    """Matrix ready for matplotlib imshow / plotly heatmap."""
    actions = tracker.action_space
    gens = [g.generation for g in tracker.generations()]
    matrix: list[list[float]] = []
    for g in tracker.generations():
        row = [b.fraction for b in g.buckets]
        matrix.append(row)
    return {
        "generations": gens,
        "actions": actions,
        "matrix": matrix,
    }

