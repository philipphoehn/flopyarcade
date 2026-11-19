"""Penalty for consecutive 'keep' actions to prevent keep-winning.

When the agent repeatedly chooses 'keep' (no well movement), the reward
for not moving should decrease to discourage a degenerate strategy.

The penalty escalates linearly with consecutive keep count:
    penalty = base_penalty * keep_count

A maximum cap prevents the penalty from overwhelming the game signal.

Usage::

    from flopyarcade.keep_penalty import KeepPenaltyTracker

    tracker = KeepPenaltyTracker(base=0.5, max_penalty=10.0)
    # Each step:
    penalty = tracker.on_action('keep')   # 0.5
    penalty = tracker.on_action('keep')   # 1.0
    penalty = tracker.on_action('up')     # 0.0 (reset)
"""

from typing import Optional


class KeepPenaltyTracker:
    """Track consecutive 'keep' actions and compute an escalating penalty.

    Args:
        base: Penalty per consecutive keep (default 0.5).
        max_penalty: Cap on total penalty (default 10.0).
        keep_actions: Set of action names considered as 'keep' (default {'keep', 'keepQ'})."""

    def __init__(self,
                 base: float = 0.5,
                 max_penalty: float = 10.0,
                 keep_actions: Optional[set] = None):
        self.base = max(0.0, base)
        self.max_penalty = max(0.0, max_penalty)
        self.keep_actions = keep_actions or {'keep', 'keepQ'}
        self.count = 0

    def on_action(self, action: str) -> float:
        """Register an action and return the penalty for this step.

        If the action is a 'keep' action, increment the counter and
        return the scaled penalty. Otherwise, reset and return 0.0.

        Args:
            action: The action string chosen this step.

        Returns:
            Penalty value to subtract from reward (0.0 if not keep).
        """
        if action in self.keep_actions:
            self.count += 1
            return min(self.base * self.count, self.max_penalty)
        else:
            self.count = 0
            return 0.0

    def reset(self) -> None:
        """Reset the consecutive-keep counter (call on episode reset)."""
        self.count = 0

    @property
    def penalty(self) -> float:
        """Current penalty value without registering a new action."""
        return min(self.base * self.count, self.max_penalty)

    def __repr__(self) -> str:
        return (f"KeepPenaltyTracker(base={self.base}, "
                f"max={self.max_penalty}, count={self.count})")


def compute_distance_cap_reward(
    length_shortest: float,
    length_actual: float,
    distance_max: float,
    reward_max: float,
    deviation_factor: float,
    keep_count: int = 0,
    keep_base: float = 0.5,
    keep_max: float = 10.0,
) -> float:
    """Compute the final reward with an optional keep-action penalty.

    This is a pure function combining the forward-reward formula with
    the keep penalty, useful for testing and benchmarking.

    Args:
        length_shortest: Straight-line distance from start to end.
        length_actual: Actual path length taken.
        distance_max: Maximum possible distance in the environment.
        reward_max: Maximum reward for a perfect trajectory.
        deviation_factor: Exponent for deviation penalty.
        keep_count: Consecutive keep actions (0 = normal movement).
        keep_base: Base penalty per keep.
        keep_max: Maximum keep penalty.

    Returns:
        Final reward value (may be negative if keep penalty dominates).
    """
    if length_shortest > 0:
        dist_frac = length_shortest / distance_max
        reward_max_segment = dist_frac * reward_max
        path_ratio = length_shortest / length_actual if length_actual > 0 else 1.0
        reward = reward_max_segment * (path_ratio ** deviation_factor)
    elif length_shortest == 0:
        reward = 0.0
    else:
        dist_frac = length_shortest / distance_max
        reward_max_segment = dist_frac * reward_max
        reward = -5.0 * abs(reward_max_segment)

    # Apply keep penalty
    if keep_count > 0:
        penalty = min(keep_base * keep_count, keep_max)
        reward -= penalty

    return reward
