"""B21: Cross-env transfer heatmap.

Builds a train-X eval-Y score matrix with per-pair significance testing,
ranks best/worst transfers, and provides text/JSON persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class TransferCell:
    train_env: str
    eval_env: str
    score: float
    p_value: float
    significant: bool


def _t_test_against_zero(scores: list[float]) -> float:
    """One-sample t-test of scores against null hypothesis mean=0.

    Returns p-value clamped to [0, 1]. If fewer than 2 observations, return 1.0.
    """
    if len(scores) < 2:
        return 1.0
    arr = np.array(scores, dtype=float)
    t_stat, p_val = stats.ttest_1samp(arr, 0.0)
    return float(np.clip(p_val, 0.0, 1.0))


def build_transfer_matrix(
    results: dict[str, dict[str, list[float]]],
) -> list[list[TransferCell]]:
    """Build the 2D transfer matrix.

    Parameters
    ----------
    results : {train_env: {eval_env: [score per run]}}

    Returns
    -------
    list[list[TransferCell]] indexed as [train_idx][eval_idx]
    """
    if not results:
        return []

    train_envs = sorted(results.keys())
    eval_envs_set: set[str] = set()
    for sub in results.values():
        eval_envs_set.update(sub.keys())
    eval_envs = sorted(eval_envs_set)

    matrix: list[list[TransferCell]] = []
    for tr in train_envs:
        row: list[TransferCell] = []
        for ev in eval_envs:
            scores = results.get(tr, {}).get(ev, [])
            if scores:
                mean_score = float(np.mean(scores))
                p_val = _t_test_against_zero(scores)
            else:
                mean_score = 0.0
                p_val = 1.0
            significant = p_val < 0.05
            row.append(
                TransferCell(
                    train_env=tr,
                    eval_env=ev,
                    score=mean_score,
                    p_value=p_val,
                    significant=significant,
                )
            )
        matrix.append(row)
    return matrix


def find_best_transfers(
    matrix: list[list[TransferCell]],
    envtypes: list[str],
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Return top-k (train, eval, score) pairs sorted by score descending."""
    cells = [c for row in matrix for c in row]
    cells.sort(key=lambda c: c.score, reverse=True)
    return [(c.train_env, c.eval_env, c.score) for c in cells[:top_k]]


def find_worst_transfers(
    matrix: list[list[TransferCell]],
    envtypes: list[str],
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Return bottom-k (train, eval, score) pairs sorted by score ascending."""
    cells = [c for row in matrix for c in row]
    cells.sort(key=lambda c: c.score)
    return [(c.train_env, c.eval_env, c.score) for c in cells[:top_k]]


def _stars(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_heatmap(
    matrix: list[list[TransferCell]],
    envtypes: list[str],
) -> str:
    """Return a text table with scores and significance stars (*, **)."""
    if not matrix or not envtypes:
        return ""

    n = len(envtypes)
    header = "train \\ eval" + "".join(f"  {ev:>12}" for ev in envtypes)
    lines = [header, "-" * len(header)]

    for i, row in enumerate(matrix):
        if i >= n:
            break
        label = f"{envtypes[i]:<12}"
        cells_str = ""
        for j in range(n):
            if j < len(row):
                c = row[j]
                star = _stars(c.p_value)
                cells_str += f"  {c.score:8.4f}{star}"
            else:
                cells_str += f"  {'\u2014':>12}"
        lines.append(label + cells_str)

    lines.append("")
    lines.append("* p<0.05  ** p<0.01")
    return "\n".join(lines)


def save_matrix(
    matrix: list[list[TransferCell]],
    envtypes: list[str],
    path: str | Path,
) -> None:
    """Serialize matrix to JSON."""
    data = {
        "envtypes": envtypes,
        "matrix": [[asdict(c) for c in row] for row in matrix],
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_matrix(path: str | Path) -> tuple[list[list[TransferCell]], list[str]]:
    """Load matrix from JSON. Returns (matrix, envtypes)."""
    data = json.loads(Path(path).read_text())
    envtypes: list[str] = data["envtypes"]
    matrix = [[TransferCell(**c) for c in row] for row in data["matrix"]]
    return matrix, envtypes
