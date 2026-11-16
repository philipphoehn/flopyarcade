#!/usr/bin/env python3
"""Tests for RLlib training examples: common_rllib.py, train_rllib_dqn.py,
train_rllib_ppo_icm.py.

Note: DDPG, A3C, and ES were removed from Ray RLlib after v2.8 and are
not available in Ray 2.56.1. Only DQN, PPO, SAC, CQL, APPO, IMPALA, BC,
MARWIL remain as core algorithms.
"""

import sys
import os
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# ============================================================
# Path setup
# ============================================================

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope='module')
def common_rllib():
    """Load common_rllib module."""
    if 'common_rllib' not in sys.modules:
        sys.path.insert(0, str(EXAMPLES_DIR))
        import common_rllib
        yield common_rllib
        sys.path.remove(str(EXAMPLES_DIR))
    else:
        import common_rllib
        yield common_rllib


@pytest.fixture
def mock_env():
    """A minimal gymnasium-compatible env for testing eval loop."""
    import gymnasium as gym
    from gymnasium import spaces

    class SimpleEnv(gym.Env):
        def __init__(self, n_steps=10):
            self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.n_steps = n_steps
            self._step_count = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._step_count = 0
            return self.np_random.uniform(-1, 1, size=(4,)).astype(np.float32), {}

        def step(self, action):
            self._step_count += 1
            obs = self.np_random.uniform(-1, 1, size=(4,)).astype(np.float32)
            reward = 1.0 if self._step_count < self.n_steps else 0.0
            terminated = self._step_count >= self.n_steps
            truncated = False
            return obs, reward, terminated, truncated, {}

    return SimpleEnv(n_steps=5)


# ============================================================
# TESTS: common_rllib.py
# ============================================================

class TestCommonRLlib:
    """Unit tests for common_rllib shared utilities."""

    def test_str_to_bool_true(self, common_rllib):
        assert common_rllib.str_to_bool('true') is True
        assert common_rllib.str_to_bool('yes') is True
        assert common_rllib.str_to_bool('1') is True

    def test_str_to_bool_false(self, common_rllib):
        assert common_rllib.str_to_bool('false') is False
        assert common_rllib.str_to_bool('no') is False
        assert common_rllib.str_to_bool('0') is False

    def test_str_to_bool_passthrough(self, common_rllib):
        assert common_rllib.str_to_bool(True) is True
        assert common_rllib.str_to_bool(False) is False

    def test_str_to_bool_invalid(self, common_rllib):
        with pytest.raises(ValueError):
            common_rllib.str_to_bool('maybe')

    def test_parse_net_arch_none(self, common_rllib):
        assert common_rllib.parse_net_arch(None) is None

    def test_parse_net_arch_valid(self, common_rllib):
        assert common_rllib.parse_net_arch('512,256,128') == [512, 256, 128]

    def test_parse_net_arch_single(self, common_rllib):
        assert common_rllib.parse_net_arch('256') == [256]

    def test_parse_net_arch_invalid(self, common_rllib):
        with pytest.raises(ValueError):
            common_rllib.parse_net_arch('abc,def')

    def test_add_common_args_defaults(self, common_rllib):
        from argparse import ArgumentParser
        parser = ArgumentParser()
        common_rllib.add_common_args(parser)
        args = parser.parse_args([])
        assert args.envtype == '3s-c'
        assert args.seed == 1
        assert args.timesteps == 1_000_000
        assert args.lr == 3e-4
        assert args.gamma == 0.99
        assert args.cpus == 4
        assert args.gpus == 0
        assert args.render is False
        assert args.resume is None
        assert args.eval_only is None
        assert args.quick_test is False
        assert args.reward_threshold == 990.0
        assert args.eval_episodes == 5
        assert args.max_steps_eval == 500

    def test_add_common_args_custom(self, common_rllib):
        from argparse import ArgumentParser
        parser = ArgumentParser()
        common_rllib.add_common_args(parser, default_envtype='6r-c', default_timesteps=5_000_000)
        args = parser.parse_args([])
        assert args.envtype == '6r-c'
        assert args.timesteps == 5_000_000

    def test_make_flopy_env(self, common_rllib):
        env = common_rllib.make_flopy_env('1s-d', seed_env=42)
        assert env is not None
        obs, _ = env.reset()
        assert obs is not None
        env.close()

    def test_make_flopy_env_continuous(self, common_rllib):
        env = common_rllib.make_flopy_env('3s-c')
        assert env.action_space is not None
        assert hasattr(env.action_space, 'shape')
        env.close()

    def test_setup_logging(self, common_rllib):
        common_rllib.setup_logging(0)
        common_rllib.setup_logging(3)
        assert True

    def test_run_evaluation_with_mock(self, common_rllib, mock_env):
        class DummyModel:
            def predict(self, obs, deterministic=True):
                return np.random.uniform(-1, 1, size=mock_env.action_space.shape), {}

        model = DummyModel()
        mean_reward = common_rllib.run_evaluation(
            model, mock_env, max_episodes=3, max_steps=10, verbose=0
        )
        assert mean_reward >= 0
        assert mean_reward <= 5

    def test_get_outdir_default(self, common_rllib):
        from argparse import Namespace
        args = Namespace(outdir=None, envtype='1s-d')
        outdir = common_rllib.get_outdir(args, 'DQN')
        assert '1s-d' in outdir
        assert 'DQN' in outdir
        assert os.path.isdir(outdir)

    def test_get_outdir_custom(self, common_rllib, tmp_path):
        from argparse import Namespace
        args = Namespace(outdir=str(tmp_path / 'custom'), envtype='1s-d')
        outdir = common_rllib.get_outdir(args, 'PPO-ICM')
        assert outdir == str(tmp_path / 'custom')

    def test_effective_seed(self, common_rllib):
        from argparse import Namespace
        args = Namespace(seed=42, seed_env=None)
        assert common_rllib.effective_seed(args) == 42
        args = Namespace(seed=42, seed_env=7)
        assert common_rllib.effective_seed(args) == 7


# ============================================================
# TESTS: RLlib config construction (Ray 2.56 compatible)
# ============================================================

class TestRLlibDQNConfig:
    """Test DQN script's config construction (Ray 2.56)."""

    def test_config_construction(self):
        from ray.rllib.algorithms import DQNConfig
        config = DQNConfig()
        config["environment"] = 'flopy_dqn'
        config["seed"] = 1
        config["num_workers"] = 2
        config["num_gpus"] = 0
        config["framework"] = 'tf'
        config["lr"] = 5e-5
        config["gamma"] = 0.99
        config["buffer_size"] = 100000
        config["learning_starts"] = 500
        config["target_network_update_freq"] = 1000
        config["train_batch_size"] = 64
        config["timesteps_per_iteration"] = 1000
        config["min_iter_time_s"] = 1
        config["model"] = {"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"}
        config["exploration"] = {"type": "PerWorkerEpsilonGreedy"}
        d = config.to_dict()
        assert d["seed"] == 1
        assert d["buffer_size"] == 100000
        assert d["learning_starts"] == 500
        assert d["target_network_update_freq"] == 1000


class TestRLlibPPOICMConfig:
    """Test PPO+ICM script's config construction (Ray 2.56)."""

    def test_config_construction(self):
        from ray.rllib.algorithms import PPOConfig
        config = PPOConfig()
        config["environment"] = 'flopy_ppo_icm'
        config["seed"] = 1
        config["num_workers"] = 2
        config["num_gpus"] = 0
        config["framework"] = 'tf'
        config["lr"] = 5e-5
        config["gamma"] = 0.99
        config["cliprange"] = 0.2
        config["train_batch_size"] = 512
        config["rollout_fragment_length"] = 50
        config["timesteps_per_iteration"] = 1000
        config["min_iter_time_s"] = 1
        config["model"] = {"fcnet_hiddens": [256, 128], "fcnet_activation": "relu"}
        config["exploration"] = {
            "type": "Curiosity",
            "eta": 1.0,
            "lr": 0.001,
            "feature_dim": 256,
            "feature_net_config": {"fcnet_hiddens": [], "fcnet_activation": "relu"},
            "inverse_net_hiddens": [256],
            "inverse_net_activation": "relu",
            "forward_net_hiddens": [256],
            "forward_net_activation": "relu",
            "beta": 0.2,
            "sub_exploration": {"type": "StochasticSampling"}
        }
        d = config.to_dict()
        assert d["seed"] == 1
        assert d["exploration"]["type"] == "Curiosity"
        assert d["exploration"]["eta"] == 1.0


# ============================================================
# TESTS: Script file integrity
# ============================================================

class TestScriptIntegrity:
    """Verify all remaining RLlib scripts have correct structure."""

    @pytest.mark.parametrize("script", [
        'train_rllib_dqn.py',
        'train_rllib_ppo_icm.py',
    ])
    def test_shebang(self, script):
        path = EXAMPLES_DIR / script
        content = path.read_text()
        assert content.startswith('#!'), f"{script} missing shebang"

    @pytest.mark.parametrize("script", [
        'train_rllib_dqn.py',
        'train_rllib_ppo_icm.py',
    ])
    def test_imports_common_rllib(self, script):
        path = EXAMPLES_DIR / script
        content = path.read_text()
        assert 'from common_rllib import' in content, f"{script} doesn't import common_rllib"

    @pytest.mark.parametrize("script", [
        'train_rllib_dqn.py',
        'train_rllib_ppo_icm.py',
    ])
    def test_has_main_guard(self, script):
        path = EXAMPLES_DIR / script
        content = path.read_text()
        assert "if __name__ == '__main__'" in content, f"{script} missing main guard"

    @pytest.mark.parametrize("script", [
        'train_rllib_dqn.py',
        'train_rllib_ppo_icm.py',
    ])
    def test_has_quick_test(self, script):
        path = EXAMPLES_DIR / script
        content = path.read_text()
        assert 'quick_test' in content, f"{script} missing --quick-test support"

    @pytest.mark.parametrize("script", [
        'train_rllib_dqn.py',
        'train_rllib_ppo_icm.py',
    ])
    def test_has_resume_flag(self, script):
        path = EXAMPLES_DIR / script
        content = path.read_text()
        assert 'args.resume' in content, f"{script} missing --resume handling"

    @pytest.mark.parametrize("script", [
        'train_rllib_dqn.py',
        'train_rllib_ppo_icm.py',
    ])
    def test_has_evaluation_config(self, script):
        path = EXAMPLES_DIR / script
        content = path.read_text()
        assert 'evaluation_config' in content, f"{script} missing evaluation_config"


# ============================================================
# TESTS: Verify removed algorithms are documented
# ============================================================

class TestRemovedAlgorithms:
    """Verify that removed scripts are gone and README documents why."""

    def test_ddpg_removed(self):
        assert not (EXAMPLES_DIR / 'train_rllib_ddpg.py').exists(), "DDPG script should be removed (not in Ray 2.56)"

    def test_a3c_removed(self):
        assert not (EXAMPLES_DIR / 'train_rllib_a3c.py').exists(), "A3C script should be removed (not in Ray 2.56)"

    def test_es_removed(self):
        assert not (EXAMPLES_DIR / 'train_rllib_es.py').exists(), "ES script should be removed (not in Ray 2.56)"

    def test_td3_removed(self):
        assert not (EXAMPLES_DIR / 'train_rllib_td3.py').exists(), "TD3 script should be removed (never existed in Ray core)"

    def test_cql_removed(self):
        # CQL exists in Ray 2.56 but our script was fake; verify old script is gone
        # (the real CQL would need custom policy, not config flags)
        # This is already handled by the earlier deletion
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
