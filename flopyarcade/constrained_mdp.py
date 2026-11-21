"""Constrained MDP with Lagrangian dual (ROADMAP R8).

Enforces hard constraint h_capture >= h_safety via adaptive Lagrangian penalty.
Reports violation rate over episodes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ConstraintSpec:
    """Safety constraint specification.

    Attributes:
        min_head: The safety threshold (h_safety). Constraint is violated when
            the constraint value falls below this.
        penalty_weight: Initial Lagrange multiplier (lambda).
        violation_log: Accumulated per-step violation magnitudes.
    """

    min_head: float = 0.5
    penalty_weight: float = 1.0
    violation_log: list[float] = field(default_factory=list)


class LagrangianDual:
    """Adaptive Lagrange multiplier for a single constraint.

    Update rule: lambda += lr * violation, clamped to [0, max_lambda].
    Penalty:     lambda * max(0, min_head - constraint_value).
    """

    def __init__(
        self,
        initial_lambda: float = 1.0,
        learning_rate: float = 0.1,
        max_lambda: float = 10.0,
    ) -> None:
        self._initial_lambda = initial_lambda
        self.learning_rate = learning_rate
        self.max_lambda = max_lambda
        self._lambda: float = initial_lambda
        self._history: list[float] = [initial_lambda]
        self.min_head: float = 0.0  # set by ConstrainedReward from ConstraintSpec

    def update(self, violation: float) -> None:
        """Increase lambda proportionally to the violation magnitude."""
        self._lambda += self.learning_rate * max(0.0, violation)
        self._lambda = max(0.0, min(self._lambda, self.max_lambda))
        self._history.append(self._lambda)

    def get_penalty(self, constraint_value: float) -> float:
        """Return lambda * max(0, min_head - constraint_value)."""
        shortfall = max(0.0, self.min_head - constraint_value)
        return self._lambda * shortfall

    def get_lambda(self) -> float:
        """Return current Lagrange multiplier."""
        return self._lambda

    def history(self) -> list[float]:
        """Return lambda values over time (including initial)."""
        return list(self._history)

    def reset(self) -> None:
        """Reset to initial lambda and clear history."""
        self._lambda = self._initial_lambda
        self._history = [self._initial_lambda]


class ConstrainedReward:
    """Wraps a base reward function with a Lagrangian safety penalty."""

    def __init__(
        self,
        base_reward_fn: Callable[[dict], float],
        constraint_spec: ConstraintSpec,
        dual: LagrangianDual,
    ) -> None:
        self.base_reward_fn = base_reward_fn
        self.constraint_spec = constraint_spec
        self.dual = dual
        # Bind the safety threshold to the dual for penalty computation
        self.dual.min_head = constraint_spec.min_head

    def compute(self, state: dict) -> dict:
        """Compute total reward, penalty, and constraint status for a single step.

        Expects ``state`` to contain ``constraint_value`` (or ``h_capture``).
        Returns a dict with keys:
            total_reward, base_reward, penalty, constraint_value, violated
        """
        base_reward = self.base_reward_fn(state)
        constraint_value = state.get(
            "constraint_value",
            state.get("h_capture", self.constraint_spec.min_head),
        )
        violated = constraint_value < self.constraint_spec.min_head

        if violated:
            penalty = self.dual.get_penalty(constraint_value)
            violation_magnitude = self.constraint_spec.min_head - constraint_value
            self.constraint_spec.violation_log.append(violation_magnitude)
            self.dual.update(violation_magnitude)
        else:
            penalty = 0.0

        total_reward = base_reward - penalty

        return {
            "total_reward": total_reward,
            "base_reward": base_reward,
            "penalty": penalty,
            "constraint_value": constraint_value,
            "violated": violated,
        }

    def accumulate(self, episode_rewards: list[dict]) -> dict:
        """Aggregate per-step results into episode-level statistics."""
        if not episode_rewards:
            return {
                "total": 0.0,
                "mean_base": 0.0,
                "mean_penalty": 0.0,
                "violation_rate": 0.0,
                "final_lambda": self.dual.get_lambda(),
            }
        n = len(episode_rewards)
        total = sum(r["total_reward"] for r in episode_rewards)
        mean_base = sum(r["base_reward"] for r in episode_rewards) / n
        mean_penalty = sum(r["penalty"] for r in episode_rewards) / n
        n_violations = sum(1 for r in episode_rewards if r["violated"])
        violation_rate = n_violations / n
        return {
            "total": total,
            "mean_base": mean_base,
            "mean_penalty": mean_penalty,
            "violation_rate": violation_rate,
            "final_lambda": self.dual.get_lambda(),
        }


class ViolationTracker:
    """Tracks constraint violations across steps and computes safety metrics."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._values: list[float] = []

    def record(self, constraint_value: float) -> None:
        """Record a single constraint value observation."""
        self._values.append(constraint_value)

    def _violations(self) -> list[float]:
        return [self.threshold - v for v in self._values if v < self.threshold]

    def violation_rate(self) -> float:
        """Fraction of recorded values that are below the threshold."""
        if not self._values:
            return 0.0
        return len(self._violations()) / len(self._values)

    def max_violation(self) -> float:
        """Largest shortfall below the threshold, or 0.0 if none."""
        viols = self._violations()
        return max(viols) if viols else 0.0

    def n_violations(self) -> int:
        """Count of values below the threshold."""
        return len(self._violations())

    def rolling_rate(self, window: int = 10) -> float:
        """Violation rate over the most recent *window* observations."""
        if not self._values:
            return 0.0
        recent = self._values[-window:]
        n_v = sum(1 for v in recent if v < self.threshold)
        return n_v / len(recent)

    def is_safe(self) -> bool:
        """True when violation rate is below 1 %."""
        return self.violation_rate() < 0.01

    def save(self, path: str) -> None:
        """Serialize to JSON at *path*."""
        with open(path, "w") as f:
            json.dump({"threshold": self.threshold, "values": self._values}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ViolationTracker":
        """Deserialize from a JSON file produced by :meth:`save`."""
        with open(path) as f:
            data = json.load(f)
        tracker = cls(threshold=data["threshold"])
        tracker._values = data["values"]
        return tracker


def format_constraint_report(episode_results: list[dict]) -> str:
    """Generate a human-readable constraint / safe-RL summary.

    Each element of *episode_results* should be a dict produced by
    :meth:`ConstrainedReward.accumulate`.
    """
    if not episode_results:
        return "No episodes recorded."

    n = len(episode_results)
    mean_violation_rate = sum(r["violation_rate"] for r in episode_results) / n
    mean_penalty = sum(r["mean_penalty"] for r in episode_results) / n
    final_lambda = episode_results[-1]["final_lambda"]
    total_reward = sum(r["total"] for r in episode_results)

    lines = [
        "=" * 50,
        "  Constraint / Safe RL Report (R8)",
        "=" * 50,
        f"  Episodes:            {n}",
        f"  Total reward:        {total_reward:.4f}",
        f"  Mean violation rate: {mean_violation_rate:.4f}",
        f"  Mean penalty:        {mean_penalty:.4f}",
        f"  Final lambda:        {final_lambda:.4f}",
        "=" * 50,
    ]
    return "\n".join(lines)
