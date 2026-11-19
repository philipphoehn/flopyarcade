"""A2: Novelty metric length-invariance fix + threshold stop + elite separation.

This module provides the missing A2 features as a standalone API that can be
integrated into the existing ``FloPyAgent`` novelty pipeline without modifying
``agent.py`` directly.

Features
--------
- **Length-invariant scoring**: already present in ``calculateNoveltyPerPair``
  (C2 fix).  Re-exported here as ``length_invariant_score`` for clarity.
- **Threshold-based stopping**: ``NoveltyThreshold`` checks whether the mean
  novelty score across the population has dropped below a configurable
  threshold, signalling that the search should terminate.
- **Elite / child separation**: ``NoveltyElitism`` partitions the population
  into *elites* (top-N by novelty score) and *novelty-children* (the rest),
  matching the ``NNOVELTYELITES`` / ``NNOVELTYAGENTS`` hyper-parameters.
- **NoveltyController**: high-level facade that ties the three pieces together
  and returns a decision dict suitable for the GA loop.

Usage::

    from flopyarcade.novelty_fix import NoveltyController

    ctrl = NoveltyController(
        threshold=0.05,
        n_elites=10,
    )

    # after each generation's novelty scores are computed:
    decision = ctrl.evaluate(novelty_scores)
    # decision = {
    #     'stop': False,
    #     'elites': [indices...],
    #     'children': [indices...],
    #     'mean_novelty': 0.12,
    #     'min_novelty': 0.03,
    # }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Length-invariant scoring (C2 fix, re-exported)
# ---------------------------------------------------------------------------

def length_invariant_score(
    pair_novelties: Sequence[float],
    episode_counts: Optional[Sequence[int]] = None,
) -> float:
    """Compute a length-invariant novelty score from per-episode pair scores.

    This is the C2 fix: divide the summed novelty by the number of episodes so
    that agents with longer action sequences are not systematically favoured.

    Args:
        pair_novelties: Per-episode novelty contributions (already computed
            by ``actionNoveltyMetric``).
        episode_counts: Optional per-episode counts (e.g. number of actions
            per episode). If provided, the score is normalised by the total
            number of actions rather than the number of episodes.

    Returns:
        Mean novelty per episode (or per action if counts given).

    Raises:
        ValueError: If inputs are empty.
    """
    if len(pair_novelties) == 0:
        raise ValueError("pair_novelties must not be empty")

    total = float(np.sum(pair_novelties))

    if episode_counts is not None:
        total_actions = int(np.sum(episode_counts))
        if total_actions == 0:
            raise ValueError("total episode actions must be > 0")
        return total / total_actions

    return total / len(pair_novelties)


# ---------------------------------------------------------------------------
# Threshold-based stopping
# ---------------------------------------------------------------------------

@dataclass
class NoveltyThreshold:
    """Configurable threshold for stopping novelty search.

    When the mean novelty score across the population drops below ``threshold``
    for ``patience`` consecutive generations, the search should terminate.

    Attributes:
        threshold: Minimum mean novelty below which we consider the search
            stalled. Typical range 0.01–0.1 depending on action space.
        patience: Number of consecutive low-novelty generations to tolerate
            before signalling stop. Default 3.
        min_population: Minimum unique agents required before the threshold
            applies (avoids stopping with < 2 agents). Default 4.
    """
    threshold: float = 0.05
    patience: int = 3
    min_population: int = 4

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.min_population < 2:
            raise ValueError("min_population must be >= 2")

    def check(self, mean_novelty: float, n_unique: int, low_count: int) -> bool:
        """Return True if the novelty search should stop.

        Args:
            mean_novelty: Current generation's mean novelty score.
            n_unique: Number of unique agents in the archive.
            low_count: Consecutive generations where mean < threshold.

        Returns:
            True → stop the novelty search.
        """
        if n_unique < self.min_population:
            return False
        return mean_novelty < self.threshold and low_count >= self.patience

    def update_low_count(self, mean_novelty: float, n_unique: int, low_count: int) -> int:
        """Update the consecutive-low counter.

        Returns:
            New low_count value.
        """
        if n_unique < self.min_population:
            return 0
        if mean_novelty < self.threshold:
            return low_count + 1
        return 0


# ---------------------------------------------------------------------------
# Elite / novelty-child separation
# ---------------------------------------------------------------------------

@dataclass
class NoveltyElitism:
    """Partition the population into elites and novelty-children.

    Mirrors the existing ``NNOVELTYELITES`` and ``NNOVELTYAGENTS`` hyper-params
    but as an explicit, testable object.

    Attributes:
        n_elites: Number of top-novelty agents to treat as elites.
            These survive to the next generation unconditionally.
        n_children: Number of remaining agents treated as novelty-children.
            These are subject to selection / replacement.
    """
    n_elites: int = 10
    n_children: int = 20

    def __post_init__(self) -> None:
        if self.n_elites < 1:
            raise ValueError("n_elites must be >= 1")
        if self.n_children < 0:
            raise ValueError("n_children must be >= 0")

    def partition(self, novelty_scores: Sequence[float]) -> dict:
        """Split indices into elites and children by novelty score.

        Args:
            novelty_scores: Novelty score per agent (higher = more novel).

        Returns:
            dict with keys:
                'elites': sorted list of indices (top n_elites)
                'children': sorted list of remaining indices (up to n_children)
                'discarded': indices beyond elites + children
        """
        scores = np.asarray(novelty_scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            return {'elites': [], 'children': [], 'discarded': []}

        # Sort indices by descending novelty
        order = np.argsort(scores)[::-1].tolist()

        n_el = min(self.n_elites, n)
        n_chi = min(self.n_children, n - n_el)

        elites = sorted(order[:n_el])
        children = sorted(order[n_el:n_el + n_chi])
        discarded = sorted(order[n_el + n_chi:])

        return {
            'elites': elites,
            'children': children,
            'discarded': discarded,
        }

    def is_elite(self, idx: int, novelty_scores: Sequence[float]) -> bool:
        """Check whether agent ``idx`` is in the elite set."""
        result = self.partition(novelty_scores)
        return idx in result['elites']


# ---------------------------------------------------------------------------
# High-level controller
# ---------------------------------------------------------------------------

@dataclass
class NoveltyControllerConfig:
    """Combined configuration for the NoveltyController."""
    threshold: float = 0.05
    patience: int = 3
    min_population: int = 4
    n_elites: int = 10
    n_children: int = 20


class NoveltyController:
    """High-level facade tying together threshold, elitism, and scoring.

    Call :meth:`evaluate` after each generation's novelty scores are computed.
    The returned dict drives the GA loop's stop/continue and elite/child split.

    Example::

        ctrl = NoveltyController(threshold=0.05, n_elites=10)

        # generation 1
        d1 = ctrl.evaluate([0.12, 0.08, 0.15, 0.03, 0.09])
        # d1['stop'] → False (patience not reached)

        # generation 2 (still low)
        d2 = ctrl.evaluate([0.04, 0.03, 0.05, 0.02, 0.04])
        # d2['stop'] → False (patience = 3, only 2 consecutive)

        # generation 3 (still low)
        d3 = ctrl.evaluate([0.04, 0.03, 0.05, 0.02, 0.04])
        # d3['stop'] → True
    """

    def __init__(self, config: Optional[NoveltyControllerConfig] = None) -> None:
        cfg = config or NoveltyControllerConfig()
        self.threshold = NoveltyThreshold(
            threshold=cfg.threshold,
            patience=cfg.patience,
            min_population=cfg.min_population,
        )
        self.elitism = NoveltyElitism(
            n_elites=cfg.n_elites,
            n_children=cfg.n_children,
        )
        self._low_count = 0

    def reset(self) -> None:
        """Reset the consecutive-low counter (e.g. at the start of a new search)."""
        self._low_count = 0

    def evaluate(self, novelty_scores: Sequence[float]) -> dict:
        """Evaluate the current generation's novelty scores.

        Args:
            novelty_scores: Novelty score per unique agent (higher = more novel).

        Returns:
            dict with keys:
                'stop' (bool): whether to terminate the novelty search
                'elites' (list[int]): indices of elite agents
                'children' (list[int]): indices of novelty-child agents
                'discarded' (list[int]): indices beyond elites + children
                'mean_novelty' (float)
                'min_novelty' (float)
                'max_novelty' (float)
                'n_unique' (int)
                'low_count' (int): consecutive low-novelty generations
        """
        scores = np.asarray(novelty_scores, dtype=np.float64)
        n_unique = len(scores)

        if n_unique == 0:
            return {
                'stop': True,
                'elites': [], 'children': [], 'discarded': [],
                'mean_novelty': 0.0, 'min_novelty': 0.0, 'max_novelty': 0.0,
                'n_unique': 0, 'low_count': 0,
            }

        mean_nov = float(np.mean(scores))
        min_nov = float(np.min(scores))
        max_nov = float(np.max(scores))

        # Update low counter
        self._low_count = self.threshold.update_low_count(
            mean_nov, n_unique, self._low_count
        )

        # Check stop condition
        stop = self.threshold.check(mean_nov, n_unique, self._low_count)

        # Partition elites / children
        parts = self.elitism.partition(scores.tolist())

        return {
            'stop': stop,
            'elites': parts['elites'],
            'children': parts['children'],
            'discarded': parts['discarded'],
            'mean_novelty': mean_nov,
            'min_novelty': min_nov,
            'max_novelty': max_nov,
            'n_unique': n_unique,
            'low_count': self._low_count,
        }

    @property
    def low_count(self) -> int:
        return self._low_count

    def to_dict(self) -> dict:
        """Serialize configuration."""
        return {
            'threshold': self.threshold.threshold,
            'patience': self.threshold.patience,
            'min_population': self.threshold.min_population,
            'n_elites': self.elitism.n_elites,
            'n_children': self.elitism.n_children,
        }

    def __repr__(self) -> str:
        t = self.threshold
        e = self.elitism
        return (
            f"NoveltyController("
            f"threshold={t.threshold}, patience={t.patience}, "
            f"min_pop={t.min_population}, "
            f"elites={e.n_elites}, children={e.n_children})"
        )
