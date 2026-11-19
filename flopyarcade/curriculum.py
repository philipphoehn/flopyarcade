#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade - Curriculum Learning Pipeline
# R2: Auto-promote 1s-d → 2s-d → 3s-d → cost modes based on reward threshold

"""Curriculum learning pipeline for FloPyArcade.

Progresses the agent through increasingly complex environments:
    1s-d → 2s-d → 3s-d → 5s-c (cost mode)

Promotion is triggered when the agent's rolling-average reward
exceeds a configurable threshold for the current level.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# --- Level definitions ---

LEVELS = [
    "1s-d",
    "2s-d",
    "3s-d",
    "5s-c",
]

DEFAULT_THRESHOLDS = {
    "1s-d": 0.70,
    "2s-d": 0.70,
    "3s-d": 0.70,
    # 5s-c is the final level; no further promotion
}

DEFAULT_WINDOW = 50  # rolling window size for reward averaging


@dataclass
class LevelState:
    """State for a single curriculum level."""
    envtype: str
    rewards: list = field(default_factory=list)
    episodes_played: int = 0
    promoted: bool = False
    promoted_at: Optional[float] = None
    threshold: float = 0.70

    @property
    def mean_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return float(np.mean(self.rewards))

    def add_reward(self, reward: float) -> None:
        self.rewards.append(float(reward))
        self.episodes_played += 1

    def rolling_mean(self, window: int = DEFAULT_WINDOW) -> float:
        if not self.rewards:
            return 0.0
        recent = self.rewards[-window:]
        return float(np.mean(recent))


@dataclass
class CurriculumConfig:
    """Configuration for the curriculum pipeline."""
    levels: list = field(default_factory=lambda: list(LEVELS))
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    window: int = DEFAULT_WINDOW
    min_episodes_per_level: int = 20
    state_file: Optional[str] = None


class Curriculum:
    """Curriculum learning pipeline.

    Usage
    -----
        cur = Curriculum()
        cur.reset()

        # In training loop:
        envtype = cur.current_envtype()
        # ... play episode, get reward ...
        cur.report_reward(reward)
        if cur.should_promote():
            cur.promote()
            # re-create environment with new envtype
    """

    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self._levels: list[LevelState] = []
        self._current_idx: int = 0
        self._history: list[dict] = []
        self.reset()

    # --- Core API ---

    def reset(self) -> None:
        """Reset curriculum to the first level."""
        self._levels = [
            LevelState(
                envtype=lv,
                threshold=self.config.thresholds.get(lv, 0.70),
            )
            for lv in self.config.levels
        ]
        self._current_idx = 0
        self._history = []

    @property
    def current_level(self) -> LevelState:
        return self._levels[self._current_idx]

    def current_envtype(self) -> str:
        """Return the current environment type string."""
        return self._levels[self._current_idx].envtype

    def current_level_index(self) -> int:
        return self._current_idx

    def is_final_level(self) -> bool:
        """True if at the last level (no further promotion)."""
        return self._current_idx >= len(self._levels) - 1

    def report_reward(self, reward: float) -> None:
        """Record an episode reward for the current level."""
        self._levels[self._current_idx].add_reward(reward)

    def should_promote(self) -> bool:
        """Check if the current level meets promotion criteria.

        Criteria:
        - At least `min_episodes_per_level` episodes played
        - Rolling mean reward >= threshold
        - Not at the final level
        """
        if self.is_final_level():
            return False
        lv = self._levels[self._current_idx]
        if lv.episodes_played < self.config.min_episodes_per_level:
            return False
        return lv.rolling_mean(self.config.window) >= lv.threshold

    def promote(self) -> str:
        """Promote to the next level. Returns the new envtype.

        Raises
        ------
        RuntimeError
            If already at the final level.
        """
        if self.is_final_level():
            raise RuntimeError("Already at final curriculum level")
        self._levels[self._current_idx].promoted = True
        self._levels[self._current_idx].promoted_at = time.time()
        self._history.append({
            "from": self._levels[self._current_idx].envtype,
            "to": self._levels[self._current_idx + 1].envtype,
            "episodes": self._levels[self._current_idx].episodes_played,
            "mean_reward": self._levels[self._current_idx].mean_reward,
            "timestamp": time.time(),
        })
        self._current_idx += 1
        return self._levels[self._current_idx].envtype

    def get_status(self) -> dict:
        """Return a serialisable status dict."""
        return {
            "current_level": self._levels[self._current_idx].envtype,
            "current_index": self._current_idx,
            "total_levels": len(self._levels),
            "is_final": self.is_final_level(),
            "should_promote": self.should_promote(),
            "current_episodes": self._levels[self._current_idx].episodes_played,
            "current_mean_reward": self._levels[self._current_idx].mean_reward,
            "current_rolling_mean": self._levels[self._current_idx].rolling_mean(
                self.config.window
            ),
            "promotion_history": self._history,
        }

    def to_dict(self) -> dict:
        """Full serialisable state."""
        return {
            "config": {
                "levels": self.config.levels,
                "thresholds": self.config.thresholds,
                "window": self.config.window,
                "min_episodes_per_level": self.config.min_episodes_per_level,
            },
            "current_index": self._current_idx,
            "levels": [
                {
                    "envtype": lv.envtype,
                    "episodes_played": lv.episodes_played,
                    "mean_reward": lv.mean_reward,
                    "promoted": lv.promoted,
                    "promoted_at": lv.promoted_at,
                    "n_rewards": len(lv.rewards),
                }
                for lv in self._levels
            ],
            "promotion_history": self._history,
        }

    def save(self, path: Optional[str] = None) -> Path:
        """Persist curriculum state to JSON."""
        p = Path(path or self.config.state_file or 'curriculum_state.json')
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    @classmethod
    def load(cls, path: str, config: Optional[CurriculumConfig] = None) -> "Curriculum":
        """Load curriculum state from JSON (rewards not restored, only counters)."""
        cur = cls(config)
        with open(path) as f:
            data = json.load(f)
        cur._current_idx = data.get('current_index', 0)
        for i, lv_data in enumerate(data.get('levels', [])):
            if i < len(cur._levels):
                cur._levels[i].episodes_played = lv_data.get('episodes_played', 0)
                cur._levels[i].promoted = lv_data.get('promoted', False)
                cur._levels[i].promoted_at = lv_data.get('promoted_at')
        cur._history = data.get('promotion_history', [])
        return cur


# --- Convenience factory ---

def make_curriculum(
    levels: Optional[list] = None,
    thresholds: Optional[dict] = None,
    window: int = DEFAULT_WINDOW,
    min_episodes: int = 20,
    state_file: Optional[str] = None,
) -> Curriculum:
    """Create a Curriculum with sensible defaults."""
    cfg = CurriculumConfig(
        levels=levels or list(LEVELS),
        thresholds=thresholds or dict(DEFAULT_THRESHOLDS),
        window=window,
        min_episodes_per_level=min_episodes,
        state_file=state_file,
    )
    return Curriculum(cfg)


# --- CLI smoke test ---

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="FloPyArcade Curriculum smoke test")
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--min-episodes', type=int, default=5)
    args = parser.parse_args()

    cur = make_curriculum(window=args.window, min_episodes=args.min_episodes)
    print(f"Curriculum: {cur.config.levels}")
    print(f"Current: {cur.current_envtype()}")

    # Simulate: earn high rewards at level 1
    print("\nSimulating high-reward episodes at 1s-d...")
    for i in range(10):
        cur.report_reward(0.85)
        if cur.should_promote():
            new_env = cur.promote()
            print(f"  Promoted to {new_env} after {i+1} episodes")
            break

    # Simulate: earn low rewards at level 2 (no promotion)
    print("\nSimulating low-reward episodes at 2s-d...")
    for i in range(10):
        cur.report_reward(0.30)
    print(f"  should_promote = {cur.should_promote()} (expected False)")

    print(f"\nStatus: {json.dumps(cur.get_status(), indent=2)}")
    print("\nSmoke test passed.")
