"""Cost-reward Pareto front analysis with bootstrap CI envelope.

ROADMAP B22: Non-dominated sort, CI envelope, algorithm ranking.
"""

import json
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ParetoPoint:
    """A single algorithm × environment observation."""

    algorithm: str
    envtype: str
    cost: float
    reward: float


def find_pareto_front(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Return the non-dominated subset (maximize reward, minimize cost).

    A point *a* dominates *b* if a.reward >= b.reward AND a.cost <= b.cost
    and at least one strict inequality holds.

    Parameters
    ----------
    points : list of ParetoPoint
        Candidate points.

    Returns
    -------
    list of ParetoPoint
        Non-dominated points, sorted by cost ascending.
    """
    if not points:
        return []

    # A point is non-dominated if no other point is strictly better in
    # both objectives (or equal in one and strictly better in the other).
    front = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            # q dominates p if q.reward >= p.reward and q.cost <= p.cost
            # with at least one strict inequality
            if (q.reward >= p.reward and q.cost <= p.cost
                    and (q.reward > p.reward or q.cost < p.cost)):
                dominated = True
                break
        if not dominated:
            front.append(p)

    front.sort(key=lambda x: x.cost)
    return front


def compute_ci_envelope(
    points: list[ParetoPoint],
    n_bootstraps: int = 500,
    ci_level: float = 0.95,
) -> list[dict]:
    """Bootstrap CI envelope over the Pareto front, binned by cost.

    Groups points by (algorithm, envtype), resamples rewards within each
    group, and reports mean ± CI at each cost level.

    Parameters
    ----------
    points : list of ParetoPoint
        Points to analyse (typically the Pareto front).
    n_bootstraps : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    list of dict
        Each dict has keys ``cost``, ``reward_mean``, ``ci_low``, ``ci_high``,
        sorted by cost ascending.
    """
    if not points:
        return []

    # Group by cost value (unique cost levels)
    from collections import defaultdict

    cost_groups: dict[float, list[float]] = defaultdict(list)
    for p in points:
        cost_groups[round(p.cost, 10)].append(p.reward)

    rng = np.random.default_rng(42)
    alpha = 1.0 - ci_level

    results = []
    for cost_val in sorted(cost_groups.keys()):
        rewards = np.array(cost_groups[cost_val], dtype=np.float64)
        n = rewards.size

        if n == 1:
            mean = float(rewards[0])
            ci_low = mean
            ci_high = mean
        else:
            idx = rng.integers(0, n, size=(n_bootstraps, n))
            boot_means = rewards[idx].mean(axis=1)
            mean = float(rewards.mean())
            ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
            ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

        results.append({
            "cost": cost_val,
            "reward_mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    return results


def front_efficiency(front: list[ParetoPoint], all_points: list[ParetoPoint]) -> float:
    """Ratio of front size to total points (lower = better concentration).

    Parameters
    ----------
    front : list of ParetoPoint
        The Pareto front.
    all_points : list of ParetoPoint
        All candidate points.

    Returns
    -------
    float
        ``len(front) / len(all_points)``; 1.0 if all are on the front.
    """
    if not all_points:
        return 1.0
    return len(front) / len(all_points)


def rank_algorithms(fronts: dict[str, list[ParetoPoint]]) -> list[str]:
    """Rank algorithms by mean reward on their Pareto front (descending).

    Parameters
    ----------
    fronts : dict mapping algorithm name -> list of ParetoPoint

    Returns
    -------
    list of str
        Algorithm names sorted by mean front reward, highest first.
    """
    scored: list[tuple[str, float]] = []
    for name, pts in fronts.items():
        if pts:
            mean_reward = sum(p.reward for p in pts) / len(pts)
        else:
            mean_reward = 0.0
        scored.append((name, mean_reward))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored]


def format_pareto_report(front: list[ParetoPoint], all_points: list[ParetoPoint]) -> str:
    """Human-readable text summary of the Pareto front.

    Parameters
    ----------
    front : list of ParetoPoint
        The Pareto front.
    all_points : list of ParetoPoint
        All candidate points.

    Returns
    -------
    str
        Multi-line report.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  Pareto Front Report (B22)")
    lines.append("=" * 60)
    lines.append(f"  Total points:   {len(all_points)}")
    lines.append(f"  Front size:     {len(front)}")
    lines.append(f"  Efficiency:     {front_efficiency(front, all_points):.4f}")
    lines.append("-" * 60)
    if front:
        lines.append(f"  {'Algorithm':<20} {'Env':<12} {'Cost':>8} {'Reward':>10}")
        lines.append("  " + "-" * 54)
        for p in front:
            lines.append(f"  {p.algorithm:<20} {p.envtype:<12} {p.cost:>8.4f} {p.reward:>10.4f}")
    else:
        lines.append("  (empty front)")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_front(front: list[ParetoPoint], path: str) -> None:
    """Serialize a Pareto front to a JSON file.

    Parameters
    ----------
    front : list of ParetoPoint
    path : str
        Destination file path.
    """
    data = [asdict(p) for p in front]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_front(path: str) -> list[ParetoPoint]:
    """Load a Pareto front from a JSON file.

    Parameters
    ----------
    path : str
        Source file path.

    Returns
    -------
    list of ParetoPoint
    """
    with open(path, "r") as f:
        data = json.load(f)
    return [ParetoPoint(**d) for d in data]
