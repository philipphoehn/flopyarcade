"""Population pruning and ensemble validation for FloPyArcade GA agents.

Roadmap C9: Prune underperforming agents from the population and validate
the best agent using an ensemble of multiple test episodes to ensure
robustness (not just one lucky run).

Usage::

    from flopyarcade.pruning_ensemble import PopulationPruner, EnsembleValidator

    pruner = PopulationPruner(keep_fraction=0.7)
    surviving = pruner.prune(agents, rewards)

    validator = EnsembleValidator(n_episodes=10, seed=42)
    report = validator.validate(env, best_agent)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Population Pruner
# ---------------------------------------------------------------------------

@dataclass
class PruneResult:
    """Result of a pruning operation.

    Attributes:
        surviving_indices: Indices of agents that survived pruning.
        removed_indices: Indices of agents that were removed.
        threshold: The reward threshold used for pruning.
        kept_count: Number of surviving agents.
        removed_count: Number of removed agents.
    """
    surviving_indices: List[int] = field(default_factory=list)
    removed_indices: List[int] = field(default_factory=list)
    threshold: float = 0.0
    kept_count: int = 0
    removed_count: int = 0


class PopulationPruner:
    """Prune the bottom fraction of agents based on reward.

    Args:
        keep_fraction: Fraction of population to keep (0 < keep_fraction <= 1).
            e.g. 0.7 keeps the top 70 % of agents.
        min_keep: Minimum number of agents to always keep (default 2).
    """

    def __init__(self, keep_fraction: float = 0.7, min_keep: int = 2):
        if not (0 < keep_fraction <= 1):
            raise ValueError("keep_fraction must be in (0, 1]")
        if min_keep < 1:
            raise ValueError("min_keep must be >= 1")
        self.keep_fraction = keep_fraction
        self.min_keep = min_keep

    def prune(
        self,
        agents: Sequence[Any],
        rewards: Sequence[float],
    ) -> PruneResult:
        """Prune agents, keeping the top ``keep_fraction`` by reward.

        Args:
            agents: Sequence of agent objects (order defines indices).
            rewards: Reward for each agent (same length as agents).

        Returns:
            PruneResult with surviving/removed indices and threshold.

        Raises:
            ValueError: If agents and rewards have different lengths.
        """
        if len(agents) != len(rewards):
            raise ValueError(
                f"agents ({len(agents)}) and rewards ({len(rewards)}) "
                "must have the same length"
            )

        n = len(agents)
        if n == 0:
            return PruneResult(threshold=0.0, kept_count=0, removed_count=0)

        # Determine how many to keep
        keep_count = max(self.min_keep, int(n * self.keep_fraction))
        keep_count = min(keep_count, n)

        # Rank agents by reward (descending)
        indexed_rewards = sorted(
            enumerate(rewards), key=lambda x: x[1], reverse=True
        )
        surviving_indices = [idx for idx, _ in indexed_rewards[:keep_count]]
        removed_indices = [idx for idx, _ in indexed_rewards[keep_count:]]

        # Threshold is the worst reward among survivors
        threshold = min(rewards[i] for i in surviving_indices) if surviving_indices else 0.0

        return PruneResult(
            surviving_indices=sorted(surviving_indices),
            removed_indices=sorted(removed_indices),
            threshold=threshold,
            kept_count=keep_count,
            removed_count=len(removed_indices),
        )

    def prune_agents(
        self,
        agents: Sequence[Any],
        rewards: Sequence[float],
    ) -> Tuple[List[Any], PruneResult]:
        """Prune and return surviving agents plus the result.

        Args:
            agents: Sequence of agent objects.
            rewards: Reward for each agent.

        Returns:
            Tuple of (surviving_agents_list, PruneResult).
        """
        result = self.prune(agents, rewards)
        surviving = [agents[i] for i in result.surviving_indices]
        return surviving, result


# ---------------------------------------------------------------------------
# Ensemble Validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Ensemble validation results for a single agent.

    Attributes:
        n_episodes: Number of validation episodes run.
        rewards: List of per-episode rewards.
        mean_reward: Mean reward across episodes.
        std_reward: Standard deviation (0 if n < 2).
        min_reward: Minimum reward.
        max_reward: Maximum reward.
        median_reward: Median reward.
        pass_rate: Fraction of episodes with reward > 0.
        passed: Whether the agent meets a basic viability threshold.
    """
    n_episodes: int = 0
    rewards: List[float] = field(default_factory=list)
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    median_reward: float = 0.0
    pass_rate: float = 0.0
    passed: bool = False


class EnsembleValidator:
    """Validate an agent over multiple episodes for robustness.

    Args:
        n_episodes: Number of validation episodes (default 10).
        seed: Optional random seed for reproducibility.
        viability_threshold: Minimum mean reward for 'passed' (default 0.0).
    """

    def __init__(
        self,
        n_episodes: int = 10,
        seed: Optional[int] = None,
        viability_threshold: float = 0.0,
    ):
        if n_episodes < 1:
            raise ValueError("n_episodes must be >= 1")
        self.n_episodes = n_episodes
        self.seed = seed
        self.viability_threshold = viability_threshold

    def validate(self, env: Any, agent: Any) -> ValidationReport:
        """Run ensemble validation episodes and compute statistics.

        Args:
            env: FloPyArcade environment (must support reset/step).
            agent: Agent with an ``act(obs)`` method.

        Returns:
            ValidationReport with per-episode rewards and aggregate stats.
        """
        rewards = []
        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=self.seed + ep if self.seed is not None else None)
            episode_reward = 0.0
            done = False
            while not done:
                action = agent.act(obs)
                obs, reward, done, _, _ = env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)

        return self._build_report(rewards)

    def validate_with_rewards(self, rewards: Sequence[float]) -> ValidationReport:
        """Build a validation report from pre-computed rewards.

        Useful for testing or when rewards are already available from
        a benchmark run.

        Args:
            rewards: List of per-episode reward values.

        Returns:
            ValidationReport.
        """
        return self._build_report(list(rewards))

    def _build_report(self, rewards: List[float]) -> ValidationReport:
        """Compute aggregate statistics from a list of rewards."""
        n = len(rewards)
        if n == 0:
            return ValidationReport(n_episodes=0)

        mean_r = statistics.mean(rewards)
        std_r = statistics.stdev(rewards) if n >= 2 else 0.0
        min_r = min(rewards)
        max_r = max(rewards)
        med_r = statistics.median(rewards)
        pass_rate = sum(1 for r in rewards if r > 0) / n
        passed = mean_r >= self.viability_threshold

        return ValidationReport(
            n_episodes=n,
            rewards=rewards,
            mean_reward=mean_r,
            std_reward=std_r,
            min_reward=min_r,
            max_reward=max_r,
            median_reward=med_r,
            pass_rate=pass_rate,
            passed=passed,
        )


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def prune_and_validate(
    agents: Sequence[Any],
    rewards: Sequence[float],
    env: Optional[Any] = None,
    pruner: Optional[PopulationPruner] = None,
    validator: Optional[EnsembleValidator] = None,
) -> Dict[str, Any]:
    """Prune population then validate the best survivor via ensemble.

    Args:
        agents: Sequence of agent objects.
        rewards: Reward for each agent.
        env: Environment for ensemble validation (optional).
        pruner: PopulationPruner instance (uses defaults if None).
        validator: EnsembleValidator instance (uses defaults if None).

    Returns:
        Dict with 'prune_result', 'best_index', 'best_reward',
        and 'validation' (if env provided).
    """
    if pruner is None:
        pruner = PopulationPruner()
    if validator is None:
        validator = EnsembleValidator()

    prune_result = pruner.prune(agents, rewards)

    if prune_result.surviving_indices:
        best_idx = max(
            prune_result.surviving_indices,
            key=lambda i: rewards[i],
        )
        best_reward = rewards[best_idx]
    else:
        best_idx = -1
        best_reward = 0.0

    result: Dict[str, Any] = {
        "prune_result": prune_result,
        "best_index": best_idx,
        "best_reward": best_reward,
    }

    if env is not None and best_idx >= 0:
        report = validator.validate(env, agents[best_idx])
        result["validation"] = report

    return result
