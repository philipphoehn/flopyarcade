"""B19: Seed-sensitivity table — std, IQR, CV per envtype×algorithm cell.

Computes statistical spread of scores across seeds to identify unstable
configurations and enable cross-algorithm comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SeedSensitivityCell:
    """One (envtype, algorithm) cell with aggregate statistics across seeds."""

    envtype: str
    algorithm: str
    n_seeds: int
    mean: float
    std: float
    iqr: float
    cv: float
    min: float
    max: float
    scores: list[float] = field(default_factory=list)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p
    f = int(k)
    c = min(f + 1, n - 1)
    frac = k - f
    return sorted_vals[f] * (1.0 - frac) + sorted_vals[c] * frac


def compute_seed_sensitivity(
    results: dict[str, dict[str, list[float]]],
) -> list[SeedSensitivityCell]:
    """Compute per-cell statistics from raw per-seed scores.

    Args:
        results: {envtype: {algorithm: [score_seed_1, score_seed_2, ...]}}

    Returns:
        List of SeedSensitivityCell, one per (envtype, algorithm) pair.
    """
    cells: list[SeedSensitivityCell] = []
    for envtype in sorted(results):
        for algorithm in sorted(results[envtype]):
            scores = results[envtype][algorithm]
            if not scores:
                continue
            n = len(scores)
            mean = sum(scores) / n
            variance = sum((s - mean) ** 2 for s in scores) / n
            std = variance ** 0.5
            sorted_scores = sorted(scores)
            q1 = _percentile(sorted_scores, 0.25)
            q3 = _percentile(sorted_scores, 0.75)
            iqr = q3 - q1
            cv = std / mean if mean != 0 else 0.0
            cells.append(
                SeedSensitivityCell(
                    envtype=envtype,
                    algorithm=algorithm,
                    n_seeds=n,
                    mean=mean,
                    std=std,
                    iqr=iqr,
                    cv=cv,
                    min=min(scores),
                    max=max(scores),
                    scores=list(scores),
                )
            )
    return cells


def find_unstable_cells(
    cells: list[SeedSensitivityCell],
    cv_threshold: float = 0.5,
) -> list[SeedSensitivityCell]:
    """Return cells whose coefficient of variation exceeds the threshold."""
    return [c for c in cells if c.cv > cv_threshold]


def format_sensitivity_report(cells: list[SeedSensitivityCell]) -> str:
    """Render a human-readable table of sensitivity cells.

    Columns: EnvType, Algorithm, Mean, Std, IQR, CV, Flagged
    """
    header = (f"{'EnvType':<16} {'Algorithm':<14} {'Mean':>10} "
              f"{'Std':>10} {'IQR':>10} {'CV':>8} {'Flagged':>8}")
    sep = "-" * len(header)
    lines = [header, sep]
    for c in cells:
        flagged = "UNSTABLE" if c.cv > 0.5 else ""
        lines.append(
            f"{c.envtype:<16} {c.algorithm:<14} {c.mean:>10.4f} {c.std:>10.4f} "
            f"{c.iqr:>10.4f} {c.cv:>8.4f} {flagged:>8}"
        )
    return "\n".join(lines)


def cross_algorithm_comparison(
    cells: list[SeedSensitivityCell],
) -> dict[str, dict[str, float]]:
    """Build {envtype: {algorithm: cv}} for quick side-by-side comparison."""
    out: dict[str, dict[str, float]] = {}
    for c in cells:
        out.setdefault(c.envtype, {})[c.algorithm] = c.cv
    return out
