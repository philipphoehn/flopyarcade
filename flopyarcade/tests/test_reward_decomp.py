"""Tests for flopyarcade.reward_decomp (ROADMAP P7)."""

import json
import tempfile
from pathlib import Path

import pytest

from flopyarcade.reward_decomp import (
    EpisodeReward,
    RewardComponent,
    RewardDecomposer,
    compare_decompositions,
    format_decomp_report,
    stacked_bar_data,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_decomposer(n_episodes: int = 3, components: list[str] | None = None) -> RewardDecomposer:
    """Create a decomposer with deterministic test data."""
    components = components or ["accuracy", "efficiency"]
    d = RewardDecomposer(envtype="test-env", component_names=components)
    for ep in range(n_episodes):
        d.record_episode(ep, {"accuracy": float(ep + 1), "efficiency": float(10 - ep)})
    return d


# ---------------------------------------------------------------------------
# dataclass tests
# ---------------------------------------------------------------------------

def test_reward_component_creation():
    c = RewardComponent(name="accuracy", value=3.5, weight=2.0)
    assert c.name == "accuracy"
    assert c.value == 3.5
    assert c.weight == 2.0


def test_reward_component_default_weight():
    c = RewardComponent(name="x", value=1.0)
    assert c.weight == 1.0


def test_episode_reward_creation():
    comps = [RewardComponent(name="a", value=1.0), RewardComponent(name="b", value=2.0)]
    ep = EpisodeReward(episode=0, components=comps, total=3.0)
    assert ep.episode == 0
    assert len(ep.components) == 2
    assert ep.total == 3.0


# ---------------------------------------------------------------------------
# RewardDecomposer core
# ---------------------------------------------------------------------------

def test_init():
    d = RewardDecomposer(envtype="mf6", component_names=["head", "flow"])
    assert d.envtype == "mf6"
    assert d.component_names == ["head", "flow"]
    assert d.n_episodes() == 0


def test_record_episode_total():
    d = RewardDecomposer(envtype="t", component_names=["a", "b"])
    ep = d.record_episode(0, {"a": 2.0, "b": 3.0})
    assert ep.total == 5.0


def test_record_episode_multiple():
    d = RewardDecomposer(envtype="t", component_names=["x", "y", "z"])
    ep1 = d.record_episode(0, {"x": 1.0, "y": 2.0, "z": 3.0})
    ep2 = d.record_episode(1, {"x": 4.0, "y": 5.0, "z": 6.0})
    assert ep1.total == 6.0
    assert ep2.total == 15.0
    assert d.n_episodes() == 2
    assert d.episodes()[1].episode == 1


def test_component_means():
    d = make_decomposer(n_episodes=3)
    means = d.component_means()
    # accuracy: (1+2+3)/3 = 2.0
    assert abs(means["accuracy"] - 2.0) < 1e-9
    # efficiency: (10+9+8)/3 = 9.0
    assert abs(means["efficiency"] - 9.0) < 1e-9


def test_component_dominance_sums_to_one():
    d = make_decomposer(n_episodes=3)
    dom = d.component_dominance()
    total = sum(dom.values())
    assert abs(total - 1.0) < 1e-9


def test_n_episodes():
    d = RewardDecomposer(envtype="t", component_names=["a"])
    assert d.n_episodes() == 0
    d.record_episode(0, {"a": 1.0})
    assert d.n_episodes() == 1
    d.record_episode(1, {"a": 2.0})
    assert d.n_episodes() == 2


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

def test_save_json_creates_file():
    d = make_decomposer(n_episodes=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "out" / "decomp.json")
        d.save_json(path)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["envtype"] == "test-env"
        assert len(data["episodes"]) == 2


def test_load_json_roundtrip():
    d = make_decomposer(n_episodes=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "rt.json")
        d.save_json(path)
        loaded = RewardDecomposer.load_json(path)
        assert loaded.envtype == d.envtype
        assert loaded.component_names == d.component_names
        assert loaded.n_episodes() == d.n_episodes()
        for i in range(d.n_episodes()):
            assert loaded.episodes()[i].total == d.episodes()[i].total
            assert loaded.episodes()[i].components == d.episodes()[i].components


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_structure():
    d = make_decomposer(n_episodes=2)
    s = d.summary()
    assert "n_episodes" in s
    assert "n_components" in s
    assert "total_mean" in s
    assert "most_important_component" in s
    assert "least_important_component" in s
    assert s["n_episodes"] == 2
    assert s["n_components"] == 2
    # efficiency means: (10+9)/2=9.5, accuracy: (1+2)/2=1.5
    # dominance: 9.5/11 vs 1.5/11 → efficiency is most important
    assert s["most_important_component"] == "efficiency"
    assert s["least_important_component"] == "accuracy"


# ---------------------------------------------------------------------------
# stacked_bar_data
# ---------------------------------------------------------------------------

def test_stacked_bar_data_structure():
    d = make_decomposer(n_episodes=3)
    data = stacked_bar_data(d)
    assert "episodes" in data
    assert "components" in data
    assert "totals" in data
    assert data["episodes"] == [0, 1, 2]
    assert set(data["components"].keys()) == {"accuracy", "efficiency"}
    assert len(data["totals"]) == 3
    assert data["components"]["accuracy"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# compare_decompositions
# ---------------------------------------------------------------------------

def test_compare_decompositions():
    d1 = RewardDecomposer(envtype="a", component_names=["x", "y"])
    d1.record_episode(0, {"x": 10.0, "y": 5.0})

    d2 = RewardDecomposer(envtype="b", component_names=["x", "y"])
    d2.record_episode(0, {"x": 3.0, "y": 5.0})

    diff = compare_decompositions(d1, d2)
    assert abs(diff["x"] - 7.0) < 1e-9
    assert abs(diff["y"]) < 1e-9


# ---------------------------------------------------------------------------
# format_decomp_report
# ---------------------------------------------------------------------------

def test_format_decomp_report_contains_names():
    d = make_decomposer(n_episodes=2)
    report = format_decomp_report(d)
    assert "accuracy" in report
    assert "efficiency" in report
    assert "test-env" in report
    assert "Component means" in report
    assert "Dominance" in report


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_single_episode_single_component():
    d = RewardDecomposer(envtype="t", component_names=["only"])
    d.record_episode(0, {"only": 42.0})
    assert d.n_episodes() == 1
    assert d.episodes()[0].total == 42.0
    assert d.component_means()["only"] == 42.0
    assert d.component_dominance()["only"] == 1.0
    s = d.summary()
    assert s["most_important_component"] == "only"
    assert s["least_important_component"] == "only"


def test_zero_values_no_division_error():
    d = RewardDecomposer(envtype="t", component_names=["a", "b"])
    d.record_episode(0, {"a": 0.0, "b": 0.0})
    assert d.episodes()[0].total == 0.0
    means = d.component_means()
    assert means["a"] == 0.0
    assert means["b"] == 0.0
    dom = d.component_dominance()
    assert dom["a"] == 0.0
    assert dom["b"] == 0.0
    s = d.summary()
    assert s["total_mean"] == 0.0


# ---------------------------------------------------------------------------
# missing component in record_episode
# ---------------------------------------------------------------------------

def test_record_episode_missing_component_defaults_zero():
    d = RewardDecomposer(envtype="t", component_names=["a", "b"])
    ep = d.record_episode(0, {"a": 5.0})
    assert ep.total == 5.0
    b_comp = next(c for c in ep.components if c.name == "b")
    assert b_comp.value == 0.0
