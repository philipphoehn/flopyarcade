"""C3: Diagnostic utilities for best-reward agent consistency.

Roadmap C3: 'Why is best-reward agent inconsistent?'
Roadmap C11: 'Test averaging behaviour with same model'

This module provides diagnostic functions to verify that:
  1. The rewards list is correctly aligned with agent indices.
  2. bestAgentReward matches max(rewards).
  3. The sorted parent index points to the true maximum.
  4. Averaging over repeated runs produces consistent results.

Usage::

    from flopyarcade.best_agent_diag import (
        verify_reward_alignment,
        verify_best_agent_consistency,
        verify_averaging,
        run_full_diagnostic,
    )

    report = run_full_diagnostic(rewards, sorted_idx, best_reward)
    if report['ok']:
        print('All consistency checks passed')
    else:
        for issue in report['issues']:
            print(f'  ISSUE: {issue}')
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticReport:
    """Result of a consistency diagnostic run.

    Attributes:
        ok: True if all checks passed.
        issues: List of human-readable issue descriptions.
        checks: Number of checks performed.
        details: Raw check results.
    """
    ok: bool = True
    issues: List[str] = field(default_factory=list)
    checks: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, msg: str) -> None:
        self.ok = False
        self.issues.append(msg)


@dataclass
class AveragingReport:
    """Result of averaging behaviour verification.

    Attributes:
        n_runs: Number of repeated runs.
        rewards_per_run: List of total rewards per run.
        mean_reward: Mean across runs.
        std_reward: Standard deviation (0 if n < 2).
        cv: Coefficient of variation (std/mean), 0 if mean is 0.
        consistent: True if CV < threshold.
        max_deviation_pct: Maximum |reward - mean| / |mean| * 100.
    """
    n_runs: int = 0
    rewards_per_run: List[float] = field(default_factory=list)
    mean_reward: float = 0.0
    std_reward: float = 0.0
    cv: float = 0.0
    consistent: bool = False
    max_deviation_pct: float = 0.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def verify_reward_alignment(
    rewards: Sequence[float],
    agent_indices: Sequence[int],
    n_agents: int,
) -> DiagnosticReport:
    """Verify that rewards list is correctly aligned with agent count.

    Checks:
      - len(rewards) == n_agents
      - All agent_indices are in [0, n_agents)
      - No duplicate indices
      - No NaN or Inf values

    Args:
        rewards: Reward for each agent.
        agent_indices: Indices referenced by the caller.
        n_agents: Total number of agents.

    Returns:
        DiagnosticReport.
    """
    report = DiagnosticReport()
    report.checks = 4

    # Check 1: length matches
    if len(rewards) != n_agents:
        report.add_issue(
            f"rewards length ({len(rewards)}) != n_agents ({n_agents})"
        )

    # Check 2: all indices in range
    for idx in agent_indices:
        if idx < 0 or idx >= n_agents:
            report.add_issue(f"agent index {idx} out of range [0, {n_agents})")
            break

    # Check 3: no duplicates
    if len(agent_indices) != len(set(agent_indices)):
        report.add_issue("duplicate indices in agent_indices")

    # Check 4: no NaN/Inf
    import math
    for i, r in enumerate(rewards):
        if math.isnan(r) or math.isinf(r):
            report.add_issue(f"rewards[{i}] = {r} is NaN or Inf")
            break

    report.details = {
        'n_rewards': len(rewards),
        'n_agents': n_agents,
        'n_indices': len(agent_indices),
    }
    return report


def verify_best_agent_consistency(
    rewards: Sequence[float],
    best_index: int,
    best_reward: float,
) -> DiagnosticReport:
    """Verify that the claimed best agent/reward is actually the maximum.

    Args:
        rewards: Full rewards list.
        best_index: Index claimed to be the best.
        best_reward: Reward claimed for the best agent.

    Returns:
        DiagnosticReport.
    """
    report = DiagnosticReport()
    report.checks = 3

    actual_max = max(rewards)
    actual_max_idx = rewards.index(actual_max)

    # Check 1: best_reward matches max
    if abs(best_reward - actual_max) > 1e-9:
        report.add_issue(
            f"best_reward ({best_reward}) != max(rewards) ({actual_max})"
        )

    # Check 2: best_index points to max
    if rewards[best_index] != actual_max:
        report.add_issue(
            f"rewards[best_index={best_index}] = {rewards[best_index]} "
            f"!= max ({actual_max} at index {actual_max_idx})"
        )

    # Check 3: no ties that could cause ambiguity
    max_count = sum(1 for r in rewards if abs(r - actual_max) < 1e-9)

    report.details = {
        'best_index': best_index,
        'best_reward': best_reward,
        'actual_max': actual_max,
        'actual_max_idx': actual_max_idx,
    }
    if max_count > 1:
        report.details['tie_count'] = max_count
    return report


def verify_sorted_indices(
    rewards: Sequence[float],
    sorted_indices: Sequence[int],
    n_elites: int,
) -> DiagnosticReport:
    """Verify that sorted_indices are correctly ordered by reward (descending).

    Args:
        rewards: Full rewards list.
        sorted_indices: Indices in claimed sorted order.
        n_elites: Expected number of elite indices.

    Returns:
        DiagnosticReport.
    """
    report = DiagnosticReport()
    report.checks = 3

    # Check 1: correct length
    if len(sorted_indices) != n_elites:
        report.add_issue(
            f"len(sorted_indices)={len(sorted_indices)} != n_elites={n_elites}"
        )

    # Check 3: all indices valid (must check before building rewards_at_idx)
    n = len(rewards)
    out_of_range = False
    for idx in sorted_indices:
        if idx < 0 or idx >= n:
            report.add_issue(f"index {idx} out of range [0, {n})")
            out_of_range = True
            break

    # Check 2: rewards in descending order (only if indices are valid)
    rewards_at_idx = []
    if not out_of_range:
        rewards_at_idx = [rewards[i] for i in sorted_indices]
        for i in range(len(rewards_at_idx) - 1):
            if rewards_at_idx[i] < rewards_at_idx[i + 1] - 1e-9:
                report.add_issue(
                    f"sorted order violated at position {i}: "
                    f"{rewards_at_idx[i]} < {rewards_at_idx[i+1]}"
                )
                break

    report.details = {
        'n_sorted': len(sorted_indices),
        'n_elites': n_elites,
        'rewards_at_idx': rewards_at_idx,
    }
    return report


# ---------------------------------------------------------------------------
# Averaging verification (C11)
# ---------------------------------------------------------------------------

def verify_averaging(
    rewards_per_run: Sequence[float],
    cv_threshold: float = 0.1,
) -> AveragingReport:
    """Verify averaging behaviour across repeated runs with the same model.

    If the same model is run multiple times on the same environment, the
    rewards should be approximately consistent (low CV).

    Args:
        rewards_per_run: Total reward for each repeated run.
        cv_threshold: Maximum acceptable coefficient of variation.

    Returns:
        AveragingReport.
    """
    n = len(rewards_per_run)
    if n == 0:
        return AveragingReport(n_runs=0)

    mean_r = statistics.mean(rewards_per_run)
    std_r = statistics.stdev(rewards_per_run) if n >= 2 else 0.0

    if abs(mean_r) > 1e-12:
        cv = std_r / abs(mean_r)
    else:
        cv = 0.0 if std_r == 0 else float('inf')

    # Max deviation percentage
    if abs(mean_r) > 1e-12:
        max_dev = max(abs(r - mean_r) for r in rewards_per_run) / abs(mean_r) * 100.0
    else:
        max_dev = 0.0

    consistent = cv < cv_threshold

    return AveragingReport(
        n_runs=n,
        rewards_per_run=list(rewards_per_run),
        mean_reward=mean_r,
        std_reward=std_r,
        cv=cv,
        consistent=consistent,
        max_deviation_pct=max_dev,
    )


# ---------------------------------------------------------------------------
# Full diagnostic pipeline
# ---------------------------------------------------------------------------

def run_full_diagnostic(
    rewards: Sequence[float],
    best_index: int,
    best_reward: float,
    sorted_indices: Optional[Sequence[int]] = None,
    n_elites: int = 1,
    n_agents: Optional[int] = None,
) -> DiagnosticReport:
    """Run all consistency checks and return a combined report.

    Args:
        rewards: Full rewards list from runAgentsRepeatedlyGenetic.
        best_index: The index used for bestAgentReward.
        best_reward: The bestAgentReward value.
        sorted_indices: The sorted parent indices (if available).
        n_elites: Number of elite agents.
        n_agents: Total agent count (defaults to len(rewards)).

    Returns:
        DiagnosticReport with all issues collected.
    """
    if n_agents is None:
        n_agents = len(rewards)

    combined = DiagnosticReport()

    # Run alignment check
    r1 = verify_reward_alignment(rewards, sorted_indices or [best_index], n_agents)
    combined.checks += r1.checks
    combined.issues.extend(r1.issues)
    combined.details['alignment'] = r1.details

    # Run best-agent consistency
    r2 = verify_best_agent_consistency(rewards, best_index, best_reward)
    combined.checks += r2.checks
    combined.issues.extend(r2.issues)
    combined.details['best_agent'] = r2.details

    # Run sorted indices check (if provided)
    if sorted_indices is not None:
        r3 = verify_sorted_indices(rewards, sorted_indices, n_elites)
        combined.checks += r3.checks
        combined.issues.extend(r3.issues)
        combined.details['sorted'] = r3.details

    combined.ok = len(combined.issues) == 0
    return combined


def diagnose_generation(
    rewards: List[float],
    sorted_parent_idxs: List[int],
    best_agent_reward: float,
    n_agents: int,
    n_elites: int = 1,
) -> Dict[str, Any]:
    """High-level diagnostic for a single GA generation.

    Mirrors the data available in FloPyArcade's agent loop after
    runAgentsRepeatedlyGenetic + _sort_and_save_top_agents.

    Args:
        rewards: rewards list from the generation.
        sorted_parent_idxs: Output of _sort_and_save_top_agents.
        best_agent_reward: Value of self.bestAgentReward.
        n_agents: Total agents (NAGENTS).
        n_elites: NAGENTELITES.

    Returns:
        Dict with 'ok', 'issues', 'averaging', 'root_cause_hints'.
    """
    report = run_full_diagnostic(
        rewards=rewards,
        best_index=sorted_parent_idxs[0] if sorted_parent_idxs else 0,
        best_reward=best_agent_reward,
        sorted_indices=sorted_parent_idxs,
        n_elites=n_elites,
        n_agents=n_agents,
    )

    # Generate root cause hints
    hints: List[str] = []
    if not report.ok:
        if any('length' in i for i in report.issues):
            hints.append(
                "Reward list length mismatch: possible multiprocess race "
                "in runAgentsRepeatedlyGenetic where not all agents "
                "returned rewards."
            )
        if any('!= max' in i for i in report.issues):
            hints.append(
                "bestAgentReward does not match max(rewards): the sort "
                "or index selection in _sort_and_save_top_agents may be "
                "using a stale rewards list."
            )
        if any('out of range' in i for i in report.issues):
            hints.append(
                "Index out of bounds: agent count changed between "
                "generation start and sort (e.g., dynamic population)."
            )
        if any('NaN' in i or 'Inf' in i for i in report.issues):
            hints.append(
                "Non-finite reward: MODFLOW convergence failure or "
                "division by zero in reward calculation."
            )

    return {
        'ok': report.ok,
        'issues': report.issues,
        'checks_run': report.checks,
        'details': report.details,
        'root_cause_hints': hints,
    }
