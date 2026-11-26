"""Failure taxonomy auto-classifier (ROADMAP P6).

Labels episodes as: boundary_hit | convergence_fail | no_progress | NaN | timeout | success.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class FailureType(Enum):
    BOUNDARY_HIT = "boundary_hit"
    CONVERGENCE_FAIL = "convergence_fail"
    NO_PROGRESS = "no_progress"
    NAN = "nan"
    TIMEOUT = "timeout"
    SUCCESS = "success"


@dataclass
class FailureRecord:
    episode: int
    step: int
    failure_type: FailureType
    reward_at_fail: float
    details: str
    obs_norm: Optional[float] = None


def classify_failure(
    reward_history: list[float],
    obs_history: list[float],
    step: int,
    max_steps: int,
    convergence_iters: int = 100,
    nan_detected: bool = False,
) -> FailureType:
    """Classify a single episode into a FailureType.

    Priority (highest first):
      1. NaN
      2. NO_PROGRESS (reached max_steps with flat reward)
      3. CONVERGENCE_FAIL (convergence_iters > 500)
      4. BOUNDARY_HIT (obs_norm > 0.99)
      5. TIMEOUT (reached max_steps with some progress)
      6. SUCCESS
    """
    if nan_detected:
        return FailureType.NAN

    if step >= max_steps:
        # Check no-progress first
        if len(reward_history) >= 2:
            delta = reward_history[-1] - reward_history[0]
            if delta < 0.01:
                return FailureType.NO_PROGRESS
        # If only one reward value or empty, treat as no progress
        if len(reward_history) < 2:
            return FailureType.NO_PROGRESS

    if convergence_iters > 500:
        return FailureType.CONVERGENCE_FAIL

    # Check boundary hit via obs_history (last observation norm)
    if obs_history:
        last_obs = abs(obs_history[-1])
        if last_obs > 0.99:
            return FailureType.BOUNDARY_HIT

    if step >= max_steps:
        return FailureType.TIMEOUT

    return FailureType.SUCCESS


class FailureClassifier:
    """Accumulate and aggregate failure records across episodes."""

    def __init__(
        self,
        progress_threshold: float = 0.01,
        convergence_threshold: int = 500,
        boundary_threshold: float = 0.99,
    ):
        self.progress_threshold = progress_threshold
        self.convergence_threshold = convergence_threshold
        self.boundary_threshold = boundary_threshold
        self.records: list[FailureRecord] = []

    def classify(
        self,
        episode: int,
        step: int,
        reward_history: list[float],
        obs_history: list[float],
        max_steps: int,
        convergence_iters: int = 100,
        nan_detected: bool = False,
    ) -> FailureRecord:
        """Classify an episode and store the record."""
        if nan_detected:
            ftype = FailureType.NAN
            details = "NaN detected in reward or observation"
        elif step >= max_steps:
            delta = (reward_history[-1] - reward_history[0]) if len(reward_history) >= 2 else 0.0
            if delta < self.progress_threshold:
                ftype = FailureType.NO_PROGRESS
                details = f"No progress: reward delta {delta:.4f} < {self.progress_threshold}"
            else:
                ftype = FailureType.TIMEOUT
                details = f"Reached max_steps={max_steps} with progress (delta={delta:.4f})"
        elif convergence_iters > self.convergence_threshold:
            ftype = FailureType.CONVERGENCE_FAIL
            details = f"convergence_iters={convergence_iters} > {self.convergence_threshold}"
        elif obs_history and abs(obs_history[-1]) > self.boundary_threshold:
            ftype = FailureType.BOUNDARY_HIT
            details = f"obs_norm={abs(obs_history[-1]):.4f} > {self.boundary_threshold}"
        else:
            ftype = FailureType.SUCCESS
            details = "Episode completed successfully"

        reward_at_fail = reward_history[-1] if reward_history else 0.0
        obs_norm = abs(obs_history[-1]) if obs_history else None

        record = FailureRecord(
            episode=episode,
            step=step,
            failure_type=ftype,
            reward_at_fail=reward_at_fail,
            details=details,
            obs_norm=obs_norm,
        )
        self.records.append(record)
        return record

    def summary(self) -> dict:
        """Return counts per FailureType, total_episodes, and failure_rate."""
        counts = {ft.value: 0 for ft in FailureType}
        for rec in self.records:
            counts[rec.failure_type.value] += 1

        total = len(self.records)
        failures = total - counts.get("success", 0)
        failure_rate = failures / total if total > 0 else 0.0

        return {
            "counts": counts,
            "total_episodes": total,
            "failure_rate": failure_rate,
        }

    def most_common_failure(self) -> FailureType:
        """Return the most frequent non-success failure type."""
        if not self.records:
            raise ValueError("No records to analyse")

        counts: dict[FailureType, int] = {}
        for rec in self.records:
            if rec.failure_type == FailureType.SUCCESS:
                continue
            counts[rec.failure_type] = counts.get(rec.failure_type, 0) + 1

        if not counts:
            return FailureType.SUCCESS

        return max(counts, key=counts.get)

    def failure_rate(self, failure_type: FailureType) -> float:
        """Fraction of all episodes that match the given failure type."""
        if not self.records:
            return 0.0
        count = sum(1 for r in self.records if r.failure_type == failure_type)
        return count / len(self.records)

    def save(self, path: str) -> None:
        """Save records as a JSON list."""
        data = []
        for rec in self.records:
            d = asdict(rec)
            d["failure_type"] = rec.failure_type.value
            data.append(d)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FailureClassifier":
        """Load records from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        classifier = cls()
        for item in data:
            item["failure_type"] = FailureType(item["failure_type"])
            classifier.records.append(FailureRecord(**item))
        return classifier


def format_failure_report(classifier: FailureClassifier) -> str:
    """Generate a human-readable text summary of the classifier state."""
    s = classifier.summary()
    lines = [
        "=" * 60,
        "FAILURE TAXONOMY REPORT",
        "=" * 60,
        f"Total episodes: {s['total_episodes']}",
        f"Failure rate:   {s['failure_rate']:.2%}",
        "-" * 60,
    ]

    for ft in FailureType:
        count = s["counts"].get(ft.value, 0)
        rate = count / s["total_episodes"] if s["total_episodes"] > 0 else 0.0
        bar = "#" * min(count, 40)
        lines.append(f"  {ft.value:<20s} {count:>5d}  ({rate:.1%})  {bar}")

    lines.append("-" * 60)

    if classifier.records:
        try:
            mc = classifier.most_common_failure()
            lines.append(f"Most common failure: {mc.value}")
        except ValueError:
            pass

    lines.append("=" * 60)
    return "\n".join(lines)