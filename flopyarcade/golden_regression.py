"""Q2 Golden-file regression: reference records and CI comparison with tolerance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class GoldenRecord:
    envtype: str
    seed: int
    episode: int
    final_reward: float
    n_steps: int
    obs_final: Optional[list[float]] = None
    action_sequence: Optional[list[int]] = None


@dataclass
class RegressionResult:
    envtype: str
    seed: int
    episode: int
    expected_reward: float
    actual_reward: float
    diff_pct: float
    passed: bool
    expected_steps: int
    actual_steps: int


class GoldenReference:
    """Collection of golden reference records for regression comparison."""

    def __init__(self, records: list[GoldenRecord]):
        self._records = records
        self._index = {
            (r.envtype, r.seed, r.episode): r for r in records
        }

    def get(self, envtype: str, seed: int, episode: int) -> GoldenRecord:
        key = (envtype, seed, episode)
        if key not in self._index:
            raise KeyError(
                f"No golden record for envtype={envtype!r}, seed={seed}, episode={episode}"
            )
        return self._index[key]

    def all(self) -> list[GoldenRecord]:
        return list(self._records)

    def n_records(self) -> int:
        return len(self._records)

    def save_json(self, path: str) -> None:
        data = [asdict(r) for r in self._records]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> GoldenReference:
        with open(path, "r") as f:
            data = json.load(f)
        records = [GoldenRecord(**d) for d in data]
        return cls(records)


def compare_to_golden(
    reference: GoldenReference,
    actual: list[GoldenRecord],
    tolerance_pct: float = 0.1,
) -> list[RegressionResult]:
    results = []
    for rec in actual:
        try:
            golden = reference.get(rec.envtype, rec.seed, rec.episode)
        except KeyError:
            results.append(
                RegressionResult(
                    envtype=rec.envtype,
                    seed=rec.seed,
                    episode=rec.episode,
                    expected_reward=0.0,
                    actual_reward=rec.final_reward,
                    diff_pct=float("inf"),
                    passed=False,
                    expected_steps=0,
                    actual_steps=rec.n_steps,
                )
            )
            continue

        expected = golden.final_reward
        actual_val = rec.final_reward
        diff_pct = abs(actual_val - expected) / (abs(expected) + 1e-8) * 100.0
        passed = diff_pct <= tolerance_pct
        results.append(
            RegressionResult(
                envtype=rec.envtype,
                seed=rec.seed,
                episode=rec.episode,
                expected_reward=expected,
                actual_reward=actual_val,
                diff_pct=diff_pct,
                passed=passed,
                expected_steps=golden.n_steps,
                actual_steps=rec.n_steps,
            )
        )
    return results


def all_passed(results: list[RegressionResult]) -> bool:
    return all(r.passed for r in results)


def failure_summary(results: list[RegressionResult]) -> dict:
    failures = [
        {"envtype": r.envtype, "episode": r.episode, "diff_pct": r.diff_pct}
        for r in results
        if not r.passed
    ]
    n_passed = sum(1 for r in results if r.passed)
    return {
        "n_total": len(results),
        "n_passed": n_passed,
        "n_failed": len(failures),
        "failures": failures,
    }


def generate_golden_from_run(
    envtype: str,
    seed: int,
    episodes: list[dict],
    n_episodes: int = 3,
) -> list[GoldenRecord]:
    records = []
    for i, ep in enumerate(episodes[:n_episodes]):
        records.append(
            GoldenRecord(
                envtype=envtype,
                seed=seed,
                episode=i,
                final_reward=float(ep.get("final_reward", 0.0)),
                n_steps=int(ep.get("n_steps", 0)),
                obs_final=ep.get("obs_final"),
                action_sequence=ep.get("action_sequence"),
            )
        )
    return records


def ci_check(
    golden_path: str,
    actual_records: list[GoldenRecord],
    tolerance_pct: float = 0.1,
) -> dict:
    reference = GoldenReference.load_json(golden_path)
    results = compare_to_golden(reference, actual_records, tolerance_pct)
    summary = failure_summary(results)
    return {
        "passed": all_passed(results),
        "results": results,
        "summary": summary,
    }


def format_regression_report(results: list[RegressionResult]) -> str:
    lines = []
    header = f"{'ENVTYPE':<12} {'SEED':>6} {'EP':>4} {'EXPECTED':>12} {'ACTUAL':>12} {'DIFF%':>10} {'STATUS':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.envtype:<12} {r.seed:>6} {r.episode:>4} "
            f"{r.expected_reward:>12.4f} {r.actual_reward:>12.4f} "
            f"{r.diff_pct:>10.4f} {status:>6}"
        )
    return "\n".join(lines)
