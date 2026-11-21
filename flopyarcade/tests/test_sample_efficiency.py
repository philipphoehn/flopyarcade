"""Tests for flopyarcade.sample_efficiency (B27)."""

import json
import os
import tempfile

import pytest

from flopyarcade.sample_efficiency import (
    TrainingCurve,
    episodes_to_threshold,
    sample_efficiency_table,
    rank_by_efficiency,
    efficiency_gap,
    format_efficiency_report,
    save_table,
    load_table,
)


# ---------- episodes_to_threshold ----------


def test_episodes_to_threshold_basic():
    """Linearly increasing rewards: threshold hit at a known episode."""
    rewards = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    episodes = list(range(1, 11))
    curve = TrainingCurve("algo", "env", episodes, rewards)
    # best = 100, 80% = 80, first hit at reward 80 -> episode 8
    assert episodes_to_threshold(curve, 80.0) == 8


def test_episodes_to_threshold_never_reached():
    """All rewards below threshold -> -1."""
    rewards = [10, 20, 30]
    episodes = [1, 2, 3]
    curve = TrainingCurve("algo", "env", episodes, rewards)
    # best = 30, 80% = 24, first hit at 30 -> episode 3
    # Actually 20 >= 24? No. 30 >= 24? Yes. So it IS reached.
    # Let's use a case where it truly is never reached:
    rewards2 = [10, 20, 30]
    curve2 = TrainingCurve("algo", "env", episodes, rewards2)
    # best=30, 80%=24 -> 30>=24 at episode 3. It is reached.
    # To test never reached, use 100% threshold where only last reaches it:
    # Actually let's use a decreasing curve where early episodes are high
    # but we want threshold that is never met. That's impossible with max.
    # The only way to get -1 is if rewards list is empty OR best is negative and threshold can't be met.
    # Actually if best > 0, target = best * pct/100 < best, and max(rewards) = best >= target always.
    # So -1 only happens with empty rewards or all-negative rewards where target is still negative.
    # With negative rewards: best = -5, target = -5*0.8 = -4. No reward >= -4 if all are <= -5.
    rewards_neg = [-10, -8, -6, -5]
    curve_neg = TrainingCurve("algo", "env", [1, 2, 3, 4], rewards_neg)
    # best=-5, target=-4, no reward >= -4 -> -1
    assert episodes_to_threshold(curve_neg, 80.0) == -1


def test_episodes_to_threshold_immediate():
    """First episode already at threshold."""
    rewards = [100, 50, 30]
    episodes = [1, 2, 3]
    curve = TrainingCurve("algo", "env", episodes, rewards)
    # best=100, 80%=80, first episode 100>=80 -> episode 1
    assert episodes_to_threshold(curve, 80.0) == 1


def test_episodes_to_threshold_100_percent():
    """100% threshold means the best reward itself."""
    rewards = [50, 100, 75, 100]
    episodes = [1, 2, 3, 4]
    curve = TrainingCurve("algo", "env", episodes, rewards)
    # best=100, 100%=100, first hit at episode 2
    assert episodes_to_threshold(curve, 100.0) == 2


def test_episodes_to_threshold_empty_curve():
    """Empty rewards -> -1."""
    curve = TrainingCurve("algo", "env", [], [])
    assert episodes_to_threshold(curve, 80.0) == -1


def test_episodes_to_threshold_single_episode():
    """Single episode: threshold is met immediately."""
    rewards = [42.0]
    episodes = [1]
    curve = TrainingCurve("algo", "env", episodes, rewards)
    # best=42, 80%=33.6, 42>=33.6 -> episode 1
    assert episodes_to_threshold(curve, 80.0) == 1


# ---------- sample_efficiency_table ----------


def test_sample_efficiency_table_structure():
    """Table has correct keys and structure."""
    results = {
        "DQN": {
            "env_a": [10, 20, 30, 40, 50],
        }
    }
    table = sample_efficiency_table(results)
    assert "DQN" in table
    assert "env_a" in table["DQN"]
    assert isinstance(table["DQN"]["env_a"], int)


def test_sample_efficiency_table_multiple_algorithms():
    """Multiple algorithms and envs produce correct entries."""
    results = {
        "DQN": {"env_a": [10, 50, 100], "env_b": [5, 20, 50]},
        "PPO": {"env_a": [100, 200, 300], "env_b": [10, 30, 50]},
    }
    table = sample_efficiency_table(results)
    assert set(table.keys()) == {"DQN", "PPO"}
    assert set(table["DQN"].keys()) == {"env_a", "env_b"}
    assert set(table["PPO"].keys()) == {"env_a", "env_b"}
    # DQN env_a: best=100, 80%=80, first hit at 100 -> ep 3
    assert table["DQN"]["env_a"] == 3
    # PPO env_a: best=300, 80%=240, first hit at 300 -> ep 3
    assert table["PPO"]["env_a"] == 3
    # DQN env_b: best=50, 80%=40, first hit at 50 -> ep 3
    assert table["DQN"]["env_b"] == 3
    # PPO env_b: best=50, 80%=40, first hit at 50 -> ep 3
    assert table["PPO"]["env_b"] == 3


# ---------- rank_by_efficiency ----------


def test_rank_by_efficiency_ordering():
    """Faster algorithms (lower episodes) rank first."""
    table = {
        "Fast": {"env_a": 5, "env_b": 10},
        "Slow": {"env_a": 50, "env_b": 100},
        "Mid": {"env_a": 20, "env_b": 30},
    }
    ranking = rank_by_efficiency(table)
    assert ranking == ["Fast", "Mid", "Slow"]


def test_rank_by_efficiency_ties():
    """Ties are broken deterministically (sorted by key)."""
    table = {
        "Alpha": {"env_a": 10},
        "Beta": {"env_a": 10},
    }
    ranking = rank_by_efficiency(table)
    assert set(ranking) == {"Alpha", "Beta"}
    assert len(ranking) == 2


# ---------- efficiency_gap ----------


def test_efficiency_gap_calculation():
    """Gap, ratio, and values are correct."""
    table = {
        "A": {"env_a": 20},
        "B": {"env_a": 10},
    }
    result = efficiency_gap(table, "A", "B", "env_a")
    assert result["a_episodes"] == 20
    assert result["b_episodes"] == 10
    assert result["gap"] == 10
    assert result["ratio"] == 2.0


def test_efficiency_gap_faster_flag():
    """Faster algorithm is the one with fewer episodes."""
    table = {
        "A": {"env_a": 20},
        "B": {"env_a": 10},
    }
    result = efficiency_gap(table, "A", "B", "env_a")
    assert result["faster"] == "B"

    # Swap: A is now faster
    table2 = {
        "A": {"env_a": 5},
        "B": {"env_a": 50},
    }
    result2 = efficiency_gap(table2, "A", "B", "env_a")
    assert result2["faster"] == "A"


# ---------- format_efficiency_report ----------


def test_format_efficiency_report_contains_names():
    """Report includes algorithm names and env types."""
    table = {
        "DQN": {"env_a": 5, "env_b": 10},
        "PPO": {"env_a": 15, "env_b": 20},
    }
    report = format_efficiency_report(table, ["env_a", "env_b"])
    assert "DQN" in report
    assert "PPO" in report
    assert "env_a" in report
    assert "env_b" in report
    assert "Sample Efficiency Report" in report


# ---------- save / load roundtrip ----------


def test_save_load_roundtrip():
    """Save to JSON and load back gives identical data."""
    table = {
        "DQN": {"env_a": 5, "env_b": 10},
        "PPO": {"env_a": 15, "env_b": 20},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_table(table, path)
        loaded = load_table(path)
        assert loaded == table
    finally:
        os.unlink(path)


# ---------- edge cases ----------


def test_empty_results_table():
    """Empty results produce empty table."""
    table = sample_efficiency_table({})
    assert table == {}


def test_rank_empty_table():
    """Ranking an empty table returns empty list."""
    assert rank_by_efficiency({}) == []