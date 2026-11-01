"""Real Ray/RLlib CPU training tests (minimal 1-iteration runs).

These tests exercise the same code paths as train_rllib_*.py main blocks:
  ray.init → register_env → config → tune.run(1 iter) → verify metrics

They use FloPyEnv (1s-d) with MODFLOW on CPU. Each test runs ~40-60s.
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_session():
    """Start Ray once for the whole module, shutdown at end."""
    import ray
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        logging_level=40,
    )
    yield ray
    ray.shutdown()


@pytest.fixture
def registered_env():
    """Register FloPyEnv 1s-d (safe to re-register)."""
    from ray.tune.registry import register_env
    from flopyarcade.flopyarcade import FloPyEnv

    def env_creator(config):
        return FloPyEnv(ENVTYPE="1s-d", flagRender=False, _seed=42)

    register_env("flopy_1s_d", env_creator)
    return "flopy_1s_d"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_training_complete(result):
    """Verify tune.run completed 1 iteration with env interaction."""
    trial = result.trials[0]
    assert trial is not None
    assert trial.last_result["training_iteration"] == 1
    assert trial.last_result["num_env_steps_sampled_lifetime"] > 0
    assert trial.last_result["done"] is True


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------


def test_dqn_one_iteration(ray_session, registered_env):
    """tune.run DQN for 1 iteration — env created, stepped, rewarded."""
    from ray import tune

    result = tune.run(
        "DQN",
        stop={"training_iteration": 1},
        num_samples=1,
        config={
            "env": registered_env,
            "num_workers": 0,
            "learning_rate": 0.0005,
            "train_batch_size": 32,
            "tune_logger": None,
        },
        verbose=0,
    )
    _assert_training_complete(result)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


def test_ppo_one_iteration(ray_session, registered_env):
    """tune.run PPO for 1 iteration."""
    from ray import tune

    result = tune.run(
        "PPO",
        stop={"training_iteration": 1},
        num_samples=1,
        config={
            "env": registered_env,
            "num_workers": 0,
            "learning_rate": 0.001,
            "train_batch_size": 32,
            "minibatch_size": 32,
            "rollout_fragment_length": 8,
            "tune_logger": None,
        },
        verbose=0,
    )
    _assert_training_complete(result)


# ---------------------------------------------------------------------------
# APPO
# ---------------------------------------------------------------------------


def test_appo_one_iteration(ray_session, registered_env):
    """tune.run APPO for 1 iteration."""
    from ray import tune

    result = tune.run(
        "APPO",
        stop={"training_iteration": 1},
        num_samples=1,
        config={
            "env": registered_env,
            "num_workers": 0,
            "learning_rate": 0.001,
            "train_batch_size": 32,
            "tune_logger": None,
        },
        verbose=0,
    )
    _assert_training_complete(result)


# ---------------------------------------------------------------------------
# IMPALA
# ---------------------------------------------------------------------------


def test_impala_one_iteration(ray_session, registered_env):
    """tune.run IMPALA for 1 iteration."""
    from ray import tune

    result = tune.run(
        "IMPALA",
        stop={"training_iteration": 1},
        num_samples=1,
        config={
            "env": registered_env,
            "num_workers": 0,
            "learning_rate": 0.01,
            "train_batch_size": 32,
            "tune_logger": None,
        },
        verbose=0,
    )
    _assert_training_complete(result)


# ---------------------------------------------------------------------------
# play_againstai: benchmark loop (env directly, 3 episodes)
# ---------------------------------------------------------------------------


def test_play_againstai_benchmark(ray_session):
    """Simulate play_againstai benchmark: 3 episodes × 5 steps on 1s-d."""
    from flopyarcade.flopyarcade import FloPyEnv

    env = FloPyEnv(ENVTYPE="1s-d", flagRender=False, _seed=42)

    scores = []
    for _ in range(3):
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        terminated = False
        for step in range(5):
            # Random action (simulates untrained agent in benchmark)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        scores.append(total_reward)

    assert len(scores) == 3
    assert all(np.isfinite(s) for s in scores)
    env.close()


# ---------------------------------------------------------------------------
# Direct env registration & stepping (covers register_env + env_creator path)
# ---------------------------------------------------------------------------


def test_register_env_and_config(ray_session):
    """Verify register_env + env instantiation + step works end-to-end."""
    from ray.tune.registry import register_env
    from flopyarcade.flopyarcade import FloPyEnv

    def creator(config):
        return FloPyEnv(ENVTYPE="2s-d", flagRender=False, _seed=42)

    register_env("flopy_2s_d", creator)

    # Instantiate and step
    env = creator({})
    obs, _ = env.reset(seed=42)
    assert obs.shape[0] > 0
    obs, reward, term, trunc, _ = env.step(0)
    assert np.isfinite(reward)
    env.close()
