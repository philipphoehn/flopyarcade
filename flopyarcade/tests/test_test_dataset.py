"""C19: Tests for test dataset (structured episode dataset for evaluation)."""

import json
import os
import tempfile

import pytest
import numpy as np

from flopyarcade.test_dataset import (
    EpisodeRecord,
    DatasetConfig,
    TestDataset,
    generate_synthetic_dataset,
    evaluate_agent,
    format_dataset_report,
    dataset_diff,
)


# -- helpers ----------------------------------------------------------------

def _make_episode(ep_id=0, envtype="1s-d", seed=42, n_steps=100,
                  reward=100.0, state="capture"):
    return EpisodeRecord(
        episode_id=ep_id,
        envtype=envtype,
        seed=seed,
        n_steps=n_steps,
        total_reward=reward,
        terminal_state=state,
    )


def _make_dataset(n=10):
    ds = TestDataset()
    envtypes = ["1s-d", "2s-d", "3s-d", "1s-c", "2s-c", "3s-c"]
    states = ["capture", "fail", "timeout"]
    for i in range(n):
        ds.add_episode(_make_episode(
            ep_id=i,
            envtype=envtypes[i % 6],
            seed=42 + i,
            n_steps=50 + i * 10,
            reward=100.0 + i * 10,
            state=states[i % 3],
        ))
    return ds


# -- tests ------------------------------------------------------------------

class TestEpisodeRecord:
    def test_creation(self):
        ep = _make_episode()
        assert ep.episode_id == 0
        assert ep.envtype == "1s-d"
        assert ep.seed == 42
        assert ep.n_steps == 100
        assert ep.total_reward == 100.0
        assert ep.terminal_state == "capture"
        assert ep.action_sequence is None
        assert ep.reward_sequence is None

    def test_invalid_terminal_state(self):
        with pytest.raises(ValueError, match="terminal_state"):
            EpisodeRecord(
                episode_id=0, envtype="1s-d", seed=1,
                n_steps=10, total_reward=1.0, terminal_state="win",
            )


class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.name == "flopyarcade-eval"
        assert cfg.version == "1.0.0"
        assert cfg.n_episodes_target == 50
        assert len(cfg.envtypes) == 6
        assert cfg.seeds == [42, 123, 456, 789, 1024]
        assert cfg.metadata == {}


class TestTestDataset:
    def test_init_default(self):
        ds = TestDataset()
        assert ds.n_episodes() == 0
        assert ds.config.name == "flopyarcade-eval"

    def test_add_episode_and_count(self):
        ds = TestDataset()
        ds.add_episode(_make_episode())
        ds.add_episode(_make_episode(ep_id=1, envtype="2s-d"))
        assert ds.n_episodes() == 2

    def test_n_envtypes(self):
        ds = TestDataset()
        ds.add_episode(_make_episode(envtype="1s-d"))
        ds.add_episode(_make_episode(ep_id=1, envtype="2s-d"))
        ds.add_episode(_make_episode(ep_id=2, envtype="1s-d"))
        assert ds.n_envtypes() == 2

    def test_episodes_by_envtype(self):
        ds = _make_dataset(12)
        eps = ds.episodes_by_envtype("1s-d")
        assert len(eps) > 0
        assert all(e.envtype == "1s-d" for e in eps)

    def test_episodes_by_seed(self):
        ds = _make_dataset(12)
        eps = ds.episodes_by_seed(42)
        assert len(eps) >= 1
        assert all(e.seed == 42 for e in eps)

    def test_episodes_by_outcome(self):
        ds = _make_dataset(12)
        eps = ds.episodes_by_outcome("capture")
        assert len(eps) > 0
        assert all(e.terminal_state == "capture" for e in eps)

    def test_capture_rate(self):
        ds = TestDataset()
        ds.add_episode(_make_episode(state="capture"))
        ds.add_episode(_make_episode(ep_id=1, state="fail"))
        ds.add_episode(_make_episode(ep_id=2, state="capture"))
        ds.add_episode(_make_episode(ep_id=3, state="timeout"))
        assert ds.capture_rate() == pytest.approx(0.5)

    def test_mean_reward_overall(self):
        ds = TestDataset()
        ds.add_episode(_make_episode(reward=100.0))
        ds.add_episode(_make_episode(ep_id=1, reward=200.0))
        assert ds.mean_reward() == pytest.approx(150.0)

    def test_mean_reward_per_envtype(self):
        ds = TestDataset()
        ds.add_episode(_make_episode(envtype="1s-d", reward=100.0))
        ds.add_episode(_make_episode(ep_id=1, envtype="2s-d", reward=300.0))
        assert ds.mean_reward("1s-d") == pytest.approx(100.0)
        assert ds.mean_reward("2s-d") == pytest.approx(300.0)

    def test_reward_stats(self):
        ds = TestDataset()
        ds.add_episode(_make_episode(reward=100.0))
        ds.add_episode(_make_episode(ep_id=1, reward=200.0))
        ds.add_episode(_make_episode(ep_id=2, reward=300.0))
        stats = ds.reward_stats()
        assert stats["n"] == 3
        assert stats["mean"] == pytest.approx(200.0)
        assert stats["min"] == pytest.approx(100.0)
        assert stats["max"] == pytest.approx(300.0)
        assert stats["std"] > 0

    def test_duration_stats(self):
        ds = TestDataset()
        ds.add_episode(_make_episode(n_steps=100))
        ds.add_episode(_make_episode(ep_id=1, n_steps=200))
        ds.add_episode(_make_episode(ep_id=2, n_steps=300))
        stats = ds.duration_stats()
        assert stats["mean_steps"] == pytest.approx(200.0)
        assert stats["min_steps"] == 100
        assert stats["max_steps"] == 300

    def test_split(self):
        ds = _make_dataset(20)
        train, test = ds.split(train_frac=0.8)
        assert len(train) + len(test) == 20
        assert len(train) == 16
        assert len(test) == 4

    def test_stratified_split(self):
        ds = _make_dataset(12)  # 2 per envtype
        train, test = ds.stratified_split(train_frac=0.5)
        # Each envtype has 2 episodes -> 1 train, 1 test
        for et in {"1s-d", "2s-d", "3s-d", "1s-c", "2s-c", "3s-c"}:
            assert len([e for e in train if e.envtype == et]) == 1
            assert len([e for e in test if e.envtype == et]) == 1

    def test_save_json_creates_file(self):
        ds = _make_dataset(10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ds.save_json(path)
            assert os.path.exists(path)
            with open(path) as fh:
                data = json.load(fh)
            assert "config" in data
            assert "episodes" in data
            assert "checksum" in data
            assert len(data["episodes"]) == 10
        finally:
            os.unlink(path)

    def test_load_json_roundtrip(self):
        ds = _make_dataset(10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ds.save_json(path)
            ds2 = TestDataset.load_json(path)
            assert ds2.n_episodes() == 10
            assert ds2.config.name == ds.config.name
            assert ds2.checksum() == ds.checksum()
        finally:
            os.unlink(path)

    def test_checksum_deterministic(self):
        ds = _make_dataset(10)
        c1 = ds.checksum()
        c2 = ds.checksum()
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex

    def test_validate_valid_dataset(self):
        cfg = DatasetConfig(n_episodes_target=5)
        ds = TestDataset(cfg)
        for i in range(6):
            ds.add_episode(_make_episode(ep_id=i))
        issues = ds.validate()
        assert issues == []

    def test_validate_catches_issues(self):
        cfg = DatasetConfig(n_episodes_target=50)
        ds = TestDataset(cfg)
        ds.add_episode(_make_episode(ep_id=0))
        issues = ds.validate()
        assert len(issues) > 0
        assert any("below target" in i for i in issues)

    def test_summary_structure(self):
        ds = _make_dataset(10)
        s = ds.summary()
        assert "name" in s
        assert "version" in s
        assert "n_episodes" in s
        assert "n_envtypes" in s
        assert "capture_rate" in s
        assert "mean_reward" in s
        assert "checksum" in s
        assert s["n_episodes"] == 10


class TestGenerateSynthetic:
    def test_count(self):
        ds = generate_synthetic_dataset()
        assert ds.n_episodes() == 50

    def test_varied_outcomes(self):
        ds = generate_synthetic_dataset()
        outcomes = {ep.terminal_state for ep in ds.episodes()}
        assert len(outcomes) >= 2  # at least two different outcomes

    def test_varied_envtypes(self):
        ds = generate_synthetic_dataset()
        assert ds.n_envtypes() == 6


class TestEvaluateAgent:
    def test_structure(self):
        ds = generate_synthetic_dataset()
        scores = {"1s-d": 150.0, "2s-d": 200.0, "3s-d": 100.0}
        result = evaluate_agent(ds, scores)
        assert "envtypes" in result
        assert "overall" in result
        assert "1s-d" in result["envtypes"]
        assert "baseline" in result["envtypes"]["1s-d"]
        assert "agent" in result["envtypes"]["1s-d"]
        assert "delta" in result["envtypes"]["1s-d"]
        assert "n_improved" in result["overall"]


class TestFormatReport:
    def test_report_content(self):
        ds = generate_synthetic_dataset()
        report = format_dataset_report(ds)
        assert "flopyarcade-eval" in report
        assert "1s-d" in report
        assert "Capture" in report
        assert "Checksum" in report


class TestDatasetDiff:
    def test_structure(self):
        ds_a = generate_synthetic_dataset(seed=1)
        ds_b = generate_synthetic_dataset(seed=2)
        diff = dataset_diff(ds_a, ds_b)
        assert "n_episodes_a" in diff
        assert "n_episodes_b" in diff
        assert "checksum_a" in diff
        assert "checksum_b" in diff
        assert "identical" in diff
        assert diff["identical"] is False

    def test_identical_datasets(self):
        ds_a = generate_synthetic_dataset(seed=42)
        ds_b = generate_synthetic_dataset(seed=42)
        diff = dataset_diff(ds_a, ds_b)
        assert diff["identical"] is True
