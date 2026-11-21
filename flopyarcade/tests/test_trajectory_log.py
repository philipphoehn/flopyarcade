"""Tests for P1: Per-step trajectory logging (HDF5/JSON)."""

import json
import os
import tempfile

import pytest

from flopyarcade.trajectory_log import (
    StepRecord,
    TrajectoryLogger,
    compare_trajectories,
)


# ------------------------------------------------------------------ #
# StepRecord
# ------------------------------------------------------------------ #

class TestStepRecord:
    def test_creation_minimal(self):
        rec = StepRecord(step=0, action=[0.0, 1.0], reward=0.5)
        assert rec.step == 0
        assert rec.action == [0.0, 1.0]
        assert rec.reward == 0.5
        assert rec.obs_sha is None
        assert rec.pos_2d is None
        assert rec.head_mean is None

    def test_creation_full(self):
        rec = StepRecord(
            step=5,
            action=[1.0, 0.5, -0.2],
            reward=-1.3,
            obs_sha="abc123",
            pos_2d=(3.14, 2.71),
            head_mean=0.42,
        )
        assert rec.step == 5
        assert rec.action == [1.0, 0.5, -0.2]
        assert rec.reward == -1.3
        assert rec.obs_sha == "abc123"
        assert rec.pos_2d == (3.14, 2.71)
        assert rec.head_mean == 0.42


# ------------------------------------------------------------------ #
# TrajectoryLogger init
# ------------------------------------------------------------------ #

class TestTrajectoryLoggerInit:
    def test_defaults(self):
        logger = TrajectoryLogger(envtype="1s-d", seed=42)
        assert logger.envtype == "1s-d"
        assert logger.seed == 42
        assert logger.agent_type == "random"
        assert logger.records == []

    def test_custom_agent_type(self):
        logger = TrajectoryLogger(envtype="2s-c", seed=7, agent_type="dqn")
        assert logger.agent_type == "dqn"


# ------------------------------------------------------------------ #
# log_step
# ------------------------------------------------------------------ #

class TestLogStep:
    def test_appends_record(self):
        logger = TrajectoryLogger(envtype="1s-d", seed=1)
        assert len(logger.records) == 0
        logger.log_step(step=0, action=[1.0, 0.0], reward=0.5)
        assert len(logger.records) == 1
        assert logger.records[0].step == 0
        assert logger.records[0].action == [1.0, 0.0]
        assert logger.records[0].reward == 0.5

    def test_with_all_optional_fields(self):
        logger = TrajectoryLogger(envtype="1s-d", seed=1)
        logger.log_step(
            step=3,
            action=[0.1, 0.2, 0.3],
            reward=1.5,
            obs_sha="deadbeef",
            pos_2d=(1.0, 2.0),
            head_mean=0.77,
        )
        rec = logger.records[0]
        assert rec.obs_sha == "deadbeef"
        assert rec.pos_2d == (1.0, 2.0)
        assert rec.head_mean == 0.77

    def test_multiple_log_steps_in_sequence(self):
        logger = TrajectoryLogger(envtype="1s-d", seed=99)
        for i in range(5):
            logger.log_step(step=i, action=[float(i)], reward=float(i))
        assert len(logger.records) == 5
        assert logger.records[4].step == 4
        assert logger.records[4].reward == 4.0


# ------------------------------------------------------------------ #
# save_json / load roundtrip
# ------------------------------------------------------------------ #

class TestSaveJson:
    def test_save_json_creates_valid_file(self, tmp_path):
        logger = TrajectoryLogger(envtype="1s-d", seed=0)
        logger.log_step(step=0, action=[1.0, 2.0], reward=0.1)
        logger.log_step(step=1, action=[0.0, 3.0], reward=-0.2)

        path = str(tmp_path / "traj.json")
        logger.save_json(path)

        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["step"] == 0
        assert data[0]["action"] == [1.0, 2.0]
        assert data[1]["reward"] == -0.2

    def test_save_load_roundtrip(self, tmp_path):
        logger = TrajectoryLogger(envtype="3s-a", seed=7, agent_type="neuro")
        logger.log_step(
            step=0, action=[0.5], reward=1.0, obs_sha="aaa", pos_2d=(1.0, 2.0), head_mean=0.3
        )
        logger.log_step(step=1, action=[0.5, 1.5], reward=-2.0)

        path = str(tmp_path / "traj.json")
        logger.save_json(path)

        loaded = TrajectoryLogger.load(path)
        assert len(loaded) == 2
        assert loaded[0].step == 0
        assert loaded[0].action == [0.5]
        assert loaded[0].reward == 1.0
        assert loaded[0].obs_sha == "aaa"
        assert loaded[0].pos_2d == (1.0, 2.0)
        assert loaded[0].head_mean == 0.3
        assert loaded[1].step == 1
        assert loaded[1].action == [0.5, 1.5]
        assert loaded[1].reward == -2.0
        assert loaded[1].obs_sha is None


# ------------------------------------------------------------------ #
# save_hdf5 / load HDF5
# ------------------------------------------------------------------ #

class TestSaveHdf5:
    def test_save_hdf5_roundtrip(self, tmp_path):
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not available")

        logger = TrajectoryLogger(envtype="1s-d", seed=42, agent_type="random")
        logger.log_step(
            step=0, action=[1.0, 0.0], reward=0.5, obs_sha="xyz", pos_2d=(3.0, 4.0), head_mean=0.9
        )
        logger.log_step(step=1, action=[0.0, 1.0], reward=-0.5)

        path = str(tmp_path / "traj.h5")
        result_path = logger.save_hdf5(path)
        assert os.path.exists(result_path)

        loaded = TrajectoryLogger.load(path)
        assert len(loaded) == 2
        assert loaded[0].step == 0
        assert loaded[0].action == [1.0, 0.0]
        assert loaded[0].reward == 0.5
        assert loaded[0].obs_sha == "xyz"
        assert loaded[0].pos_2d == (3.0, 4.0)
        assert loaded[0].head_mean == 0.9
        assert loaded[1].step == 1
        assert loaded[1].action == [0.0, 1.0]
        assert loaded[1].reward == -0.5
        assert loaded[1].obs_sha is None
        assert loaded[1].pos_2d is None
        assert loaded[1].head_mean is None

    def test_save_hdf5_empty(self, tmp_path):
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not available")

        logger = TrajectoryLogger(envtype="1s-d", seed=0)
        path = str(tmp_path / "empty.h5")
        logger.save_hdf5(path)
        assert os.path.exists(path)

        loaded = TrajectoryLogger.load(path)
        assert loaded == []


# ------------------------------------------------------------------ #
# summary
# ------------------------------------------------------------------ #

class TestSummary:
    def test_computes_correct_stats(self):
        logger = TrajectoryLogger(envtype="1s-d", seed=1)
        logger.log_step(step=0, action=[1.0], reward=2.0)
        logger.log_step(step=1, action=[1.0], reward=4.0)
        logger.log_step(step=2, action=[2.0], reward=6.0)

        s = logger.summary()
        assert s["n_steps"] == 3
        assert s["mean_reward"] == pytest.approx(4.0)
        assert s["min_reward"] == 2.0
        assert s["max_reward"] == 6.0
        assert s["total_reward"] == 12.0
        assert s["n_unique_actions"] == 2  # [1.0] and [2.0]

    def test_empty_logger_summary(self):
        logger = TrajectoryLogger(envtype="1s-d", seed=0)
        s = logger.summary()
        assert s["n_steps"] == 0
        assert s["mean_reward"] == 0.0
        assert s["min_reward"] == 0.0
        assert s["max_reward"] == 0.0
        assert s["total_reward"] == 0.0
        assert s["n_unique_actions"] == 0


# ------------------------------------------------------------------ #
# to_dataframe
# ------------------------------------------------------------------ #

class TestToDataFrame:
    def test_returns_list_of_dicts_with_correct_keys(self):
        logger = TrajectoryLogger(envtype="1s-d", seed=1)
        logger.log_step(step=0, action=[0.5, 1.5], reward=0.1, pos_2d=(1.0, 2.0))

        df = logger.to_dataframe()
        assert isinstance(df, list)
        assert len(df) == 1
        expected_keys = {"step", "action", "reward", "obs_sha", "pos_2d", "head_mean"}
        assert set(df[0].keys()) == expected_keys
        assert df[0]["step"] == 0
        assert df[0]["action"] == [0.5, 1.5]
        assert df[0]["reward"] == 0.1
        assert df[0]["pos_2d"] == [1.0, 2.0]


# ------------------------------------------------------------------ #
# compare_trajectories
# ------------------------------------------------------------------ #

class TestCompareTrajectories:
    def test_same_length(self):
        traj_a = [
            StepRecord(step=0, action=[1.0, 0.0], reward=1.0),
            StepRecord(step=1, action=[0.0, 1.0], reward=3.0),
        ]
        traj_b = [
            StepRecord(step=0, action=[1.0, 0.0], reward=2.0),
            StepRecord(step=1, action=[1.0, 1.0], reward=4.0),
        ]
        result = compare_trajectories(traj_a, traj_b)
        assert result["len_a"] == 2
        assert result["len_b"] == 2
        assert result["mean_reward_a"] == pytest.approx(2.0)
        assert result["mean_reward_b"] == pytest.approx(3.0)
        assert result["reward_diff"] == pytest.approx(-1.0)
        # 1 out of 2 actions match
        assert result["action_similarity"] == pytest.approx(0.5)

    def test_different_lengths(self):
        traj_a = [
            StepRecord(step=0, action=[1.0], reward=1.0),
            StepRecord(step=1, action=[2.0], reward=2.0),
            StepRecord(step=2, action=[3.0], reward=3.0),
        ]
        traj_b = [
            StepRecord(step=0, action=[1.0], reward=10.0),
        ]
        result = compare_trajectories(traj_a, traj_b)
        assert result["len_a"] == 3
        assert result["len_b"] == 1
        assert result["mean_reward_a"] == pytest.approx(2.0)
        assert result["mean_reward_b"] == pytest.approx(10.0)
        # aligned to min length = 1, 1 match
        assert result["action_similarity"] == pytest.approx(1.0)

    def test_action_similarity_all_same(self):
        traj_a = [
            StepRecord(step=0, action=[5.0, 3.0], reward=1.0),
            StepRecord(step=1, action=[5.0, 3.0], reward=2.0),
        ]
        traj_b = [
            StepRecord(step=0, action=[5.0, 3.0], reward=99.0),
            StepRecord(step=1, action=[5.0, 3.0], reward=88.0),
        ]
        result = compare_trajectories(traj_a, traj_b)
        assert result["action_similarity"] == pytest.approx(1.0)

    def test_empty_trajectories(self):
        result = compare_trajectories([], [])
        assert result["len_a"] == 0
        assert result["len_b"] == 0
        assert result["mean_reward_a"] == 0.0
        assert result["mean_reward_b"] == 0.0
        assert result["reward_diff"] == 0.0
        assert result["action_similarity"] == 0.0
