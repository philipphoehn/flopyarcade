"""Cross-env transfer 6×6 heatmap with significance stars (ROADMAP V6).

Provides a transfer score matrix, Welch's and paired t-tests (no scipy),
significance star annotations, heatmap data for matplotlib imshow,
a demo matrix generator, matrix comparison, and a formatted text report.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TransferScore:
    """A single transfer score entry between two environments."""

    source_env: str
    target_env: str
    score: float
    p_value: float | None = None
    n_trials: int = 10


# ---------------------------------------------------------------------------
# Transfer matrix
# ---------------------------------------------------------------------------

class TransferMatrix:
    """Square transfer score matrix with p-values and significance stars."""

    def __init__(self, envtypes: list[str]):
        self.envtypes: list[str] = list(envtypes)
        n = len(self.envtypes)
        self._scores: list[list[float]] = [[0.0] * n for _ in range(n)]
        self._p_values: list[list[float]] = [[float("nan")] * n for _ in range(n)]
        self._n_trials: list[list[int]] = [[10] * n for _ in range(n)]

    # -- accessors ----------------------------------------------------------

    def n_envtypes(self) -> int:
        return len(self.envtypes)

    def set_score(
        self,
        source: str,
        target: str,
        score: float,
        p_value: float | None = None,
        n_trials: int = 10,
    ) -> None:
        si = self.envtypes.index(source)
        ti = self.envtypes.index(target)
        self._scores[si][ti] = score
        self._p_values[si][ti] = p_value if p_value is not None else float("nan")
        self._n_trials[si][ti] = n_trials

    def get_score(self, source: str, target: str) -> TransferScore:
        si = self.envtypes.index(source)
        ti = self.envtypes.index(target)
        return TransferScore(
            source_env=source,
            target_env=target,
            score=self._scores[si][ti],
            p_value=self._p_values[si][ti] if not math.isnan(self._p_values[si][ti]) else None,
            n_trials=self._n_trials[si][ti],
        )

    def matrix(self) -> list[list[float]]:
        """Return the 2-D score matrix (rows=source, cols=target)."""
        return [row[:] for row in self._scores]

    def p_matrix(self) -> list[list[float]]:
        """Return the 2-D p-value matrix; NaN where missing."""
        return [row[:] for row in self._p_values]

    # -- analysis -----------------------------------------------------------

    def best_transfer(self) -> TransferScore:
        """Highest off-diagonal score."""
        best = None
        for i, s in enumerate(self.envtypes):
            for j, t in enumerate(self.envtypes):
                if i == j:
                    continue
                if best is None or self._scores[i][j] > best.score:
                    best = TransferScore(
                        source_env=s, target_env=t,
                        score=self._scores[i][j],
                        p_value=self._p_values[i][j] if not math.isnan(self._p_values[i][j]) else None,
                        n_trials=self._n_trials[i][j],
                    )
        if best is None:
            s = self.envtypes[0]
            best = TransferScore(source_env=s, target_env=s, score=self._scores[0][0],
                                 p_value=None, n_trials=self._n_trials[0][0])
        return best

    def worst_transfer(self) -> TransferScore:
        """Lowest off-diagonal score."""
        worst = None
        for i, s in enumerate(self.envtypes):
            for j, t in enumerate(self.envtypes):
                if i == j:
                    continue
                if worst is None or self._scores[i][j] < worst.score:
                    worst = TransferScore(
                        source_env=s, target_env=t,
                        score=self._scores[i][j],
                        p_value=self._p_values[i][j] if not math.isnan(self._p_values[i][j]) else None,
                        n_trials=self._n_trials[i][j],
                    )
        if worst is None:
            s = self.envtypes[0]
            worst = TransferScore(source_env=s, target_env=s, score=self._scores[0][0],
                                  p_value=None, n_trials=self._n_trials[0][0])
        return worst

    def significance_stars(self, alpha: float = 0.05) -> list[list[str]]:
        """2-D grid of stars: '***' p<0.001, '**' p<0.01, '*' p<0.05, '' else; diagonal '—'."""
        n = self.n_envtypes()
        grid: list[list[str]] = []
        for i in range(n):
            row: list[str] = []
            for j in range(n):
                if i == j:
                    row.append("—")
                else:
                    p = self._p_values[i][j]
                    if math.isnan(p):
                        row.append("")
                    elif p < 0.001:
                        row.append("***")
                    elif p < 0.01:
                        row.append("**")
                    elif p < alpha:
                        row.append("*")
                    else:
                        row.append("")
            grid.append(row)
        return grid

    def row_best(self, source: str) -> dict:
        """Best transfer FROM this source (highest off-diagonal score in that row)."""
        si = self.envtypes.index(source)
        best_target = None
        best_score = float("-inf")
        for j, t in enumerate(self.envtypes):
            if j == si:
                continue
            if self._scores[si][j] > best_score:
                best_score = self._scores[si][j]
                best_target = t
        if best_target is None:
            return {"target": source, "score": self._scores[si][si]}
        return {"target": best_target, "score": best_score}

    def col_best(self, target: str) -> dict:
        """Best transfer TO this target (highest off-diagonal score in that column)."""
        ti = self.envtypes.index(target)
        best_source = None
        best_score = float("-inf")
        for i, s in enumerate(self.envtypes):
            if i == ti:
                continue
            if self._scores[i][ti] > best_score:
                best_score = self._scores[i][ti]
                best_source = s
        if best_source is None:
            return {"source": target, "score": self._scores[ti][ti]}
        return {"source": best_source, "score": best_score}

    # -- export -------------------------------------------------------------

    def heatmap_data(self) -> dict:
        """Ready-for-matplotlib-imshow dict: labels, values, stars, vmin, vmax."""
        vals = self.matrix()
        all_vals = [v for row in vals for v in row]
        vmin = min(all_vals) if all_vals else 0.0
        vmax = max(all_vals) if all_vals else 1.0
        return {
            "labels": self.envtypes[:],
            "values": vals,
            "stars": self.significance_stars(),
            "vmin": vmin,
            "vmax": vmax,
        }

    # -- statistical tests (no scipy) ---------------------------------------

    @staticmethod
    def t_test(scores_a: list[float], scores_b: list[float]) -> dict:
        """Welch's t-test for two independent samples (p approx via normal CDF)."""
        n_a = len(scores_a)
        n_b = len(scores_b)
        if n_a < 2 or n_b < 2:
            return {"t_stat": float("nan"), "df": 0, "p_value": 1.0, "significant": False}
        mean_a = sum(scores_a) / n_a
        mean_b = sum(scores_b) / n_b
        var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1)
        var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1)
        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return {"t_stat": 0.0, "df": n_a + n_b - 2, "p_value": 1.0, "significant": False}
        t_stat = (mean_a - mean_b) / se
        # Welch–Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else n_a + n_b - 2
        # p-value approx via normal CDF (two-tailed)
        p_value = 1.0 - math.erf(abs(t_stat) / math.sqrt(2.0))
        return {
            "t_stat": t_stat,
            "df": df,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def paired_t_test(paired: list[tuple[float, float]]) -> dict:
        """Paired t-test (p approx via normal CDF, two-tailed)."""
        n = len(paired)
        if n < 2:
            return {"t_stat": float("nan"), "df": 0, "p_value": 1.0, "significant": False}
        diffs = [a - b for a, b in paired]
        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var_d / n) if var_d > 0 else 0.0
        if se == 0:
            return {"t_stat": 0.0, "df": n - 1, "p_value": 1.0, "significant": False}
        t_stat = mean_d / se
        df = n - 1
        p_value = 1.0 - math.erf(abs(t_stat) / math.sqrt(2.0))
        return {
            "t_stat": t_stat,
            "df": df,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    # -- serialisation ------------------------------------------------------

    def save_json(self, path: str) -> None:
        """Serialise to JSON."""
        data = {
            "envtypes": self.envtypes,
            "scores": self._scores,
            "p_values": [[(v if not math.isnan(v) else None) for v in row] for row in self._p_values],
            "n_trials": self._n_trials,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> TransferMatrix:
        """Load from JSON."""
        with open(path) as f:
            data = json.load(f)
        tm = cls(data["envtypes"])
        for i in range(len(data["envtypes"])):
            for j in range(len(data["envtypes"])):
                p = data["p_values"][i][j]
                tm._scores[i][j] = data["scores"][i][j]
                tm._p_values[i][j] = float(p) if p is not None else float("nan")
                tm._n_trials[i][j] = data["n_trials"][i][j]
        return tm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def generate_demo_matrix(n_envs: int = 6, seed: int = 42) -> TransferMatrix:
    """Create a synthetic n×n transfer matrix with realistic scores.

    Diagonal = 1.0, off-diagonal scores in [0.3, 0.8],
    p-values drawn to produce a mix of significance levels.
    """
    import random
    rng = random.Random(seed)
    envs = [f"env_{i}" for i in range(n_envs)]
    tm = TransferMatrix(envs)
    for i, s in enumerate(envs):
        for j, t in enumerate(envs):
            if i == j:
                tm.set_score(s, t, 1.0, p_value=None, n_trials=10)
            else:
                score = round(rng.uniform(0.3, 0.8), 4)
                # Mix of p-values to exercise all star levels
                p_pool = [0.0003, 0.0007, 0.004, 0.008, 0.03, 0.04, 0.12, 0.35, None]
                p = rng.choice(p_pool)
                tm.set_score(s, t, score, p_value=p, n_trials=rng.randint(5, 20))
    return tm


def compare_matrices(m1: TransferMatrix, m2: TransferMatrix) -> dict:
    """Per-cell score differences between two matrices.

    Returns: {diffs: 2-D list, best_improved: (s,t,delta), worst_degraded: (s,t,delta)}
    """
    n = m1.n_envtypes()
    diffs: list[list[float]] = []
    best_improved = None
    worst_degraded = None
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            d = m2.matrix()[i][j] - m1.matrix()[i][j]
            row.append(d)
            if i != j:
                if best_improved is None or d > best_improved[2]:
                    best_improved = (m1.envtypes[i], m1.envtypes[j], d)
                if worst_degraded is None or d < worst_degraded[2]:
                    worst_degraded = (m1.envtypes[i], m1.envtypes[j], d)
        diffs.append(row)
    return {
        "diffs": diffs,
        "best_improved": best_improved,
        "worst_degraded": worst_degraded,
    }


def format_transfer_report(m: TransferMatrix) -> str:
    """Pretty text table with scores and significance stars."""
    n = m.n_envtypes()
    labels = m.envtypes
    scores = m.matrix()
    stars = m.significance_stars()
    # column widths
    col_w = max(max(len(l) for l in labels), 6) + 2
    sep = "-" * (col_w * n + 1)
    lines = ["Transfer Score Matrix (rows=source, cols=target)", sep]
    header = "".join(f"{l:<{col_w}}" for l in labels)
    lines.append(header)
    lines.append(sep)
    for i, s in enumerate(labels):
        cells = []
        for j in range(n):
            score_str = f"{scores[i][j]:.3f}"
            star = stars[i][j]
            display = f"{score_str} {star}".strip()
            cells.append(f"{display:<{col_w}}")
        lines.append("".join(cells))
    lines.append(sep)
    # summary
    best = m.best_transfer()
    worst = m.worst_transfer()
    lines.append(f"Best transfer:  {best.source_env} → {best.target_env}  score={best.score:.4f}")
    lines.append(f"Worst transfer: {worst.source_env} → {worst.target_env}  score={worst.score:.4f}")
    return "\n".join(lines)
