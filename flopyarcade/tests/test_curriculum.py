"""Tests for R2: Curriculum learning pipeline.

Validates level progression, promotion logic, reward tracking,
serialisation, and edge cases.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from flopyarcade.curriculum import (
    LEVELS,
    Curriculum,
    CurriculumConfig,
    LevelState,
    make_curriculum,
)


# --- LevelState unit tests ---

class TestLevelState:
    def test_default_state(self):
        lv = LevelState(envtype="1s-d")
        assert lv.envtype == "1s-d"
        assert lv.rewards == []
        assert lv.episodes_played == 0
        assert lv.promoted is False
        assert lv.promoted_at is None

    def test_mean_reward_empty(self):
        lv = LevelState(envtype="1s-d")
        assert lv.mean_reward == 0.0

    def test_mean_reward_values(self):
        lv = LevelState(envtype="1s-d")
        for r in [0.5, 0.8, 0.6]:
            lv.add_reward(r)
        assert abs(lv.mean_reward - 0.6333) < 0.001

    def test_rolling_mean_short_window(self):
        lv = LevelState(envtype="1s-d")
        for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
            lv.add_reward(r)
        # window=3: last 3 are 0.3, 0.4, 0.5
        assert abs(lv.rolling_mean(window=3) - 0.4) < 0.001

    def test_rolling_mean_empty(self):
        lv = LevelState(envtype="1s-d")
        assert lv.rolling_mean(window=10) == 0.0

    def test_episodes_counter(self):
        lv = LevelState(envtype="1s-d")
        for _ in range(5):
            lv.add_reward(0.5)
        assert lv.episodes_played == 5


# --- Curriculum core tests ---

class TestCurriculumInit:
    def test_default_levels(self):
        cur = Curriculum()
        assert cur.config.levels == LEVELS

    def test_starts_at_first_level(self):
        cur = Curriculum()
        assert cur.current_envtype() == "1s-d"
        assert cur.current_level_index() == 0

    def test_not_final_initially(self):
        cur = Curriculum()
        assert cur.is_final_level() is False

    def test_custom_levels(self):
        cfg = CurriculumConfig(levels=["1s-d", "3s-d"])
        cur = Curriculum(cfg)
        assert cur.config.levels == ["1s-d", "3s-d"]
        assert len(cur._levels) == 2


class TestPromotionLogic:
    def test_no_promote_below_min_episodes(self):
        cfg = CurriculumConfig(min_episodes_per_level=10, window=5)
        cur = Curriculum(cfg)
        for _ in range(5):
            cur.report_reward(0.99)
        assert cur.should_promote() is False

    def test_no_promote_below_threshold(self):
        cfg = CurriculumConfig(min_episodes_per_level=5, window=10)
        cur = Curriculum(cfg)
        for _ in range(10):
            cur.report_reward(0.30)
        assert cur.should_promote() is False

    def test_promote_above_threshold(self):
        cfg = CurriculumConfig(min_episodes_per_level=5, window=10)
        cur = Curriculum(cfg)
        for _ in range(10):
            cur.report_reward(0.85)
        assert cur.should_promote() is True

    def test_promote_advances_level(self):
        cfg = CurriculumConfig(min_episodes_per_level=5, window=10)
        cur = Curriculum(cfg)
        for _ in range(10):
            cur.report_reward(0.85)
        new_env = cur.promote()
        assert new_env == "2s-d"
        assert cur.current_envtype() == "2s-d"
        assert cur.current_level_index() == 1

    def test_no_promote_at_final_level(self):
        cfg = CurriculumConfig(min_episodes_per_level=5, window=10)
        cur = Curriculum(cfg)
        # Jump to final level
        cur._current_idx = len(cur._levels) - 1
        for _ in range(10):
            cur.report_reward(0.99)
        assert cur.should_promote() is False

    def test_promote_raises_at_final(self):
        cfg = CurriculumConfig(min_episodes_per_level=5, window=10)
        cur = Curriculum(cfg)
        cur._current_idx = len(cur._levels) - 1
        with pytest.raises(RuntimeError, match="final curriculum level"):
            cur.promote()

    def test_promotion_history_recorded(self):
        cfg = CurriculumConfig(min_episodes_per_level=5, window=10)
        cur = Curriculum(cfg)
        for _ in range(10):
            cur.report_reward(0.85)
        cur.promote()
        assert len(cur._history) == 1
        assert cur._history[0]["from"] == "1s-d"
        assert cur._history[0]["to"] == "2s-d"
        assert cur._history[0]["episodes"] == 10

    def test_multiple_promotions(self):
        cfg = CurriculumConfig(min_episodes_per_level=5, window=10)
        cur = Curriculum(cfg)
        for level in range(2):  # promote twice
            for _ in range(10):
                cur.report_reward(0.85)
            cur.promote()
        assert cur.current_envtype() == "3s-d"
        assert len(cur._history) == 2


class TestStatusAndSerialisation:
    def test_get_status_keys(self):
        cur = Curriculum()
        status = cur.get_status()
        assert "current_level" in status
        assert "current_index" in status
        assert "total_levels" in status
        assert "is_final" in status
        assert "should_promote" in status
        assert "promotion_history" in status

    def test_to_dict_structure(self):
        cur = Curriculum()
        d = cur.to_dict()
        assert "config" in d
        assert "current_index" in d
        assert "levels" in d
        assert len(d["levels"]) == len(LEVELS)

    def test_save_and_load(self):
        cur = Curriculum(CurriculumConfig(min_episodes_per_level=5, window=10))
        for _ in range(10):
            cur.report_reward(0.85)
        cur.promote()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        cur.save(path)
        cur2 = Curriculum.load(path)
        assert cur2.current_envtype() == "2s-d"
        assert cur2.current_level_index() == 1

        os.unlink(path)

    def test_save_creates_parent_dirs(self):
        cur = Curriculum()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "dir", "state.json")
            cur.save(path)
            assert os.path.exists(path)


class TestMakeCurriculum:
    def test_returns_curriculum_instance(self):
        cur = make_curriculum()
        assert isinstance(cur, Curriculum)

    def test_custom_params(self):
        cur = make_curriculum(
            levels=["1s-d", "2s-d"],
            thresholds={"1s-d": 0.5},
            window=20,
            min_episodes=10,
        )
        assert cur.config.levels == ["1s-d", "2s-d"]
        assert cur.config.thresholds["1s-d"] == 0.5
        assert cur.config.window == 20
        assert cur.config.min_episodes_per_level == 10


class TestEdgeCases:
    def test_zero_rewards(self):
        cur = Curriculum(CurriculumConfig(min_episodes_per_level=5, window=10))
        for _ in range(10):
            cur.report_reward(0.0)
        assert cur.should_promote() is False

    def test_negative_rewards(self):
        cur = Curriculum(CurriculumConfig(min_episodes_per_level=5, window=10))
        for _ in range(10):
            cur.report_reward(-0.5)
        assert cur.should_promote() is False

    def test_single_level_config(self):
        cfg = CurriculumConfig(levels=["1s-d"])
        cur = Curriculum(cfg)
        assert cur.is_final_level() is True
        assert cur.should_promote() is False

    def test_reset_clears_state(self):
        cur = Curriculum(CurriculumConfig(min_episodes_per_level=5, window=10))
        for _ in range(10):
            cur.report_reward(0.85)
        cur.promote()
        cur.reset()
        assert cur.current_envtype() == "1s-d"
        assert cur.current_level_index() == 0
        assert cur._history == []
