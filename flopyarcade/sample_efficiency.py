"""B27: Sample efficiency metric – episodes-to-80%-of-best per algorithm × env."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingCurve:
    """A single training run's reward trajectory."""

    algorithm: str
    envtype: str
    episodes: list[int]
    rewards: list[float]


def episodes_to_threshold(curve: TrainingCurve, threshold_pct: float = 80.0) -> int:
    """Return the episode index at which reward first reaches *threshold_pct* %
    of the best reward in the curve.  Returns -1 if never reached.
    """
    if not curve.rewards:
        return -1
    best = max(curve.rewards)
    if best <= 0:
        # Avoid division by zero / negative thresholds
        target = best * (threshold_pct / 100.0)
        for i, r in enumerate(curve.rewards):
            if r >= target:
                return curve.episodes[i] if i < len(curve.episodes) else i
        return -1
    target = best * (threshold_pct / 100.0)
    for i, r in enumerate(curve.rewards):
        if r >= target:
            return curve.episodes[i] if i < len(curve.episodes) else i
    return -1


def sample_efficiency_table(
    results: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, int]]:
    """Compute episodes-to-80% for every algorithm × env combination.

    Parameters
    ----------
    results : {algorithm: {envtype: [reward per episode]}}

    Returns
    -------
    {algorithm: {envtype: episodes_to_80}}
    """
    table: dict[str, dict[str, int]] = {}
    for algo, envs in results.items():
        table[algo] = {}
        for envtype, rewards in envs.items():
            curve = TrainingCurve(
                algorithm=algo,
                envtype=envtype,
                episodes=list(range(1, len(rewards) + 1)),
                rewards=rewards,
            )
            table[algo][envtype] = episodes_to_threshold(curve, 80.0)
    return table


def rank_by_efficiency(table: dict[str, dict[str, int]]) -> list[str]:
    """Rank algorithms by mean episodes-to-80 (lower = better).

    Algorithms with -1 (never reached) are treated as having a very large
    penalty so they sort last.
    """
    _PENALTY = 10**9

    def _mean_eps(algo: str) -> float:
        values = table[algo]
        scores = [_PENALTY if v < 0 else v for v in values.values()]
        return sum(scores) / len(scores) if scores else _PENALTY

    return sorted(table.keys(), key=_mean_eps)


def efficiency_gap(
    table: dict[str, dict[str, int]],
    algorithm_a: str,
    algorithm_b: str,
    envtype: str,
) -> dict:
    """Compute the gap between two algorithms on a specific env."""
    a_episodes = table[algorithm_a][envtype]
    b_episodes = table[algorithm_b][envtype]
    gap = a_episodes - b_episodes
    if b_episodes == 0 or b_episodes < 0:
        ratio = float("inf") if a_episodes >= 0 else float("nan")
    else:
        ratio = a_episodes / b_episodes
    faster = algorithm_a if a_episodes <= b_episodes else algorithm_b
    return {
        "a_episodes": a_episodes,
        "b_episodes": b_episodes,
        "gap": gap,
        "ratio": ratio,
        "faster": faster,
    }


def format_efficiency_report(
    table: dict[str, dict[str, int]],
    envtypes: list[str],
) -> str:
    """Return a human-readable text table with rankings."""
    rankings = rank_by_efficiency(table)
    algo_width = max((len(a) for a in table), default=10)
    header = "Rank".ljust(6) + "Algorithm".ljust(algo_width + 2)
    for e in envtypes:
        header += e.ljust(14)
    sep = "-" * len(header)

    lines = ["Sample Efficiency Report (episodes-to-80%-of-best)", sep]
    lines.append(header)
    lines.append(sep)

    for rank, algo in enumerate(rankings, start=1):
        row = str(rank).ljust(6) + algo.ljust(algo_width + 2)
        for e in envtypes:
            val = table[algo].get(e, -1)
            display = "N/A" if val < 0 else str(val)
            row += display.ljust(14)
        lines.append(row)

    lines.append(sep)
    lines.append("N/A = threshold never reached")
    return "\n".join(lines)


def save_table(table: dict[str, dict[str, int]], path: str | Path) -> None:
    """Save the efficiency table as JSON."""
    Path(path).write_text(json.dumps(table, indent=2))


def load_table(path: str | Path) -> dict[str, dict[str, int]]:
    """Load the efficiency table from JSON."""
    return json.loads(Path(path).read_text())
