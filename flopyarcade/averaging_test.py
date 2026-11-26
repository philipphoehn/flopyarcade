"""C11: Test averaging behaviour with same model.

Verification/testing module that ensures when the same model (same weights,
same seed) is evaluated multiple times, results are consistent.
Tests determinism and averaging logic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class RunResult:
    """A single evaluation run result."""
    run_id: int
    seed: int
    score: float
    n_episodes: int
    final_epsilon: Optional[float] = None
    model_hash: Optional[str] = None


class AveragingValidator:
    """Validates averaging behaviour across multiple runs of the same model."""

    def __init__(self, model_hash: str = 'default', tolerance: float = 0.0):
        self.model_hash = model_hash
        self.tolerance = tolerance
        self._runs: list[RunResult] = []

    # ------------------------------------------------------------------ data
    def add_run(self, run_id: int, seed: int, score: float,
                n_episodes: int, final_epsilon=None) -> None:
        """Append a run result."""
        self._runs.append(RunResult(
            run_id=run_id,
            seed=seed,
            score=score,
            n_episodes=n_episodes,
            final_epsilon=final_epsilon,
            model_hash=self.model_hash,
        ))

    def runs(self) -> list[RunResult]:
        return list(self._runs)

    def n_runs(self) -> int:
        return len(self._runs)

    # ------------------------------------------------------------- statistics
    def mean_score(self) -> float:
        if not self._runs:
            return 0.0
        return sum(r.score for r in self._runs) / len(self._runs)

    def std_score(self) -> float:
        """Population standard deviation of scores."""
        if not self._runs:
            return 0.0
        m = self.mean_score()
        variance = sum((r.score - m) ** 2 for r in self._runs) / len(self._runs)
        return math.sqrt(variance)

    def cv(self) -> float:
        """Coefficient of variation = std / mean."""
        m = self.mean_score()
        if m == 0.0:
            return 0.0
        return self.std_score() / m

    def deterministic(self) -> bool:
        """True if all scores are identical (within 1e-12)."""
        if len(self._runs) <= 1:
            return True
        first = self._runs[0].score
        return all(abs(r.score - first) < 1e-12 for r in self._runs[1:])

    def max_deviation(self) -> float:
        """Max |score - mean| across all runs."""
        if not self._runs:
            return 0.0
        m = self.mean_score()
        return max(abs(r.score - m) for r in self._runs)

    # ------------------------------------------------------------- checks
    def consistency_check(self, threshold: float = 0.01) -> dict:
        m = self.mean_score()
        std = self.std_score()
        c = self.cv()
        det = self.deterministic()
        max_dev = self.max_deviation()
        passed = max_dev < threshold * abs(m) if m != 0 else max_dev < threshold
        return {
            'deterministic': det,
            'max_deviation': max_dev,
            'mean': m,
            'std': std,
            'cv': c,
            'passed': passed,
        }

    def pairwise_diffs(self) -> list[tuple[int, int, float]]:
        """All (i, j, |score_i - score_j|) for i < j."""
        diffs = []
        n = len(self._runs)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append((i, j, abs(self._runs[i].score - self._runs[j].score)))
        return diffs

    # ---------------------------------------------------------- seed grouping
    def same_seed_scores(self, seed: int) -> list[float]:
        return [r.score for r in self._runs if r.seed == seed]

    def different_seed_scores(self) -> dict[int, float]:
        """seed -> score (last run per seed)."""
        result: dict[int, float] = {}
        for r in self._runs:
            result[r.seed] = r.score
        return result

    def variance_by_seed(self) -> dict[int, float]:
        """seed -> std of its scores (if multiple runs per seed)."""
        from collections import defaultdict
        groups: dict[int, list[float]] = defaultdict(list)
        for r in self._runs:
            groups[r.seed].append(r.score)
        result = {}
        for seed, scores in groups.items():
            if len(scores) < 2:
                result[seed] = 0.0
            else:
                m = sum(scores) / len(scores)
                var = sum((s - m) ** 2 for s in scores) / len(scores)
                result[seed] = math.sqrt(var)
        return result

    # ------------------------------------------------------------- persistence
    def save_json(self, path: str) -> None:
        data = {
            'model_hash': self.model_hash,
            'tolerance': self.tolerance,
            'runs': [asdict(r) for r in self._runs],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> AveragingValidator:
        with open(path, 'r') as f:
            data = json.load(f)
        v = cls(model_hash=data['model_hash'], tolerance=data['tolerance'])
        for rd in data['runs']:
            v._runs.append(RunResult(**rd))
        return v


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_identical_runs(n_runs: int, seed: int,
                             base_score: float = 0.75,
                             noise: float = 0.0) -> list[RunResult]:
    """Generate synthetic runs with optional noise (default zero => identical)."""
    import random
    rng = random.Random(seed)
    results = []
    for i in range(n_runs):
        score = base_score + (rng.gauss(0, noise) if noise > 0 else 0.0)
        results.append(RunResult(
            run_id=i,
            seed=seed,
            score=score,
            n_episodes=100,
            final_epsilon=0.01,
            model_hash='sim',
        ))
    return results


def simulate_noisy_runs(n_runs: int, seed: int,
                        base_score: float = 0.75,
                        noise_std: float = 0.05) -> list[RunResult]:
    """Generate synthetic runs with Gaussian noise."""
    import random
    rng = random.Random(seed)
    results = []
    for i in range(n_runs):
        score = base_score + rng.gauss(0, noise_std)
        results.append(RunResult(
            run_id=i,
            seed=seed,
            score=score,
            n_episodes=100,
            final_epsilon=0.01,
            model_hash='sim',
        ))
    return results


# ---------------------------------------------------------------------------
# Assertion / reporting
# ---------------------------------------------------------------------------

def assert_consistent(results: list[RunResult], tolerance: float = 0.01) -> dict:
    """Raise AssertionError if inconsistency exceeds tolerance. Returns check dict."""
    v = AveragingValidator()
    for r in results:
        v.add_run(r.run_id, r.seed, r.score, r.n_episodes, r.final_epsilon)
    check = v.consistency_check(threshold=tolerance)
    if not check['passed']:
        raise AssertionError(
            f"Inconsistency detected: max_deviation={check['max_deviation']:.6f} "
            f">= threshold*mean={tolerance * abs(check['mean']):.6f}"
        )
    return check


def format_averaging_report(v: AveragingValidator) -> str:
    """Produce a human-readable text summary."""
    check = v.consistency_check()
    lines = [
        "=" * 50,
        "  Averaging Validation Report",
        "=" * 50,
        f"  Model hash      : {v.model_hash}",
        f"  Number of runs  : {v.n_runs()}",
        f"  Mean score      : {check['mean']:.6f}",
        f"  Std deviation   : {check['std']:.6f}",
        f"  Coeff. variation: {check['cv']:.6f}",
        f"  Max deviation   : {check['max_deviation']:.6f}",
        f"  Deterministic   : {check['deterministic']}",
        f"  Consistency     : {'PASSED' if check['passed'] else 'FAILED'}",
        "-" * 50,
    ]
    if v.n_runs() > 0:
        lines.append("  Per-run scores:")
        for r in v.runs():
            eps_str = f", eps={r.final_epsilon:.4f}" if r.final_epsilon is not None else ""
            lines.append(f"    run {r.run_id:3d} | seed={r.seed:5d} | score={r.score:.6f}{eps_str}")
    lines.append("=" * 50)
    return "\n".join(lines)
