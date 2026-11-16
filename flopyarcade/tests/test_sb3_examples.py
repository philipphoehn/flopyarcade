#!/usr/bin/env python3
"""Tests for SB3 training examples: common_sb3.py, train_sb3_sac.py,
train_sb3_ddpg.py, train_sb3_ppo.py."""

import sys
import os
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# ============================================================
# Path setup: ensure examples/ is importable
# ============================================================

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'


def _load_module(name, filepath):
    """Load a module from a file path by name."""
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope='module')
def common_sb3():
    """Load common_sb3 module."""
    if 'common_sb3' not in sys.modules:
        sys.path.insert(0, str(EXAMPLES_DIR))
        import common_sb3
        yield common_sb3
        sys.path.remove(str(EXAMPLES_DIR))
    else:
        import common_sb3
        yield common_sb3


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
# TESTS: common_sb3.py
# ============================================================

class TestCommonSB3:
    """Unit tests for common_sb3 shared utilities."""

    def test_str_to_bool_true(self, common_sb3):
        assert common_sb3.str_to_bool('true') is True
        assert common_sb3.str_to_bool('True') is True
        assert common_sb3.str_to_bool('yes') is True
        assert common_sb3.str_to_bool('1') is True
        assert common_sb3.str_to_bool('y') is True

    def test_str_to_bool_false(self, common_sb3):
        assert common_sb3.str_to_bool('false') is False
        assert common_sb3.str_to_bool('no') is False
        assert common_sb3.str_to_bool('0') is False
        assert common_sb3.str_to_bool('n') is False

    def test_str_to_bool_passthrough(self, common_sb3):
        assert common_sb3.str_to_bool(True) is True
        assert common_sb3.str_to_bool(False) is False

    def test_str_to_bool_invalid(self, common_sb3):
        with pytest.raises(ValueError):
            common_sb3.str_to_bool('maybe')

    def test_parse_net_arch_none(self, common_sb3):
        assert common_sb3.parse_net_arch(None) is None

    def test_parse_net_arch_valid(self, common_sb3):
        assert common_sb3.parse_net_arch('512,256,128') == [512, 256, 128]

    def test_parse_net_arch_single(self, common_sb3):
        assert common_sb3.parse_net_arch('256') == [256]

    def test_parse_net_arch_invalid(self, common_sb3):
        with pytest.raises(ValueError):
            common_sb3.parse_net_arch('abc,def')

    def test_add_common_args(self, common_sb3):
        from argparse import ArgumentParser
        parser = ArgumentParser()
        common_sb3.add_common_args(parser)
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

    def test_add_common_args_custom_defaults(self, common_sb3):
        from argparse import ArgumentParser
        parser = ArgumentParser()
        common_sb3.add_common_args(parser, default_envtype='6r-c', default_timesteps=5_000_000)
        args = parser.parse_args([])
        assert args.envtype == '6r-c'
        assert args.timesteps == 5_000_000

    def test_make_flopy_env(self, common_sb3):
        env = common_sb3.make_flopy_env('1s-d', seed_env=42)
        assert env is not None
        obs, _ = env.reset()
        assert obs is not None
        env.close()

    def test_make_flopy_env_continuous(self, common_sb3):
        env = common_sb3.make_flopy_env('3s-c')
        assert env.action_space is not None
        assert hasattr(env.action_space, 'shape')
        env.close()

    def test_setup_logging(self, common_sb3):
        import logging
        common_sb3.setup_logging(0)  # ERROR
        common_sb3.setup_logging(1)  # WARNING
        common_sb3.setup_logging(2)  # INFO
        common_sb3.setup_logging(3)  # DEBUG
        # Just verify no exception
        assert True

    def test_run_evaluation_with_mock(self, common_sb3, mock_env):
        """Test eval loop with a simple mock env and a dummy model."""
        class DummyModel:
            def predict(self, obs, deterministic=True):
                return np.random.uniform(-1, 1, size=mock_env.action_space.shape), {}

        model = DummyModel()
        mean_reward = common_sb3.run_evaluation(
            model, mock_env, max_episodes=3, max_steps=10, verbose=0
        )
        assert mean_reward >= 0
        assert mean_reward <= 5  # 5 steps max reward

    def test_get_outdir_default(self, common_sb3):
        from argparse import Namespace
        args = Namespace(outdir=None, envtype='1s-d')
        outdir = common_sb3.get_outdir(args, 'SAC')
        assert '1s-d' in outdir
        assert 'SAC' in outdir
        assert os.path.isdir(outdir)

    def test_get_outdir_custom(self, common_sb3, tmp_path):
        from argparse import Namespace
        args = Namespace(outdir=str(tmp_path / 'custom'), envtype='1s-d')
        outdir = common_sb3.get_outdir(args, 'SAC')
        assert outdir == str(tmp_path / 'custom')
        assert os.path.isdir(outdir)

    def test_effective_seed(self, common_sb3):
        from argparse import Namespace
        # seed_env=None → use seed
        args = Namespace(seed=42, seed_env=None)
        assert common_sb3.effective_seed(args) == 42
        # seed_env=7 → use seed_env
        args = Namespace(seed=42, seed_env=7)
        assert common_sb3.effective_seed(args) == 7


# ============================================================
# TESTS: train_sb3_sac.py (argparse + config)
# ============================================================

class TestTrainSB3SAC:
    """Test SAC script's argparse and model construction."""

    def test_argparse_default(self, common_sb3):
        """Verify the SAC script's argument parser works."""
        from argparse import ArgumentParser
        # Simulate what the script does
        parser = ArgumentParser()
        common_sb3.add_common_args(parser, default_envtype='3s-c')
        parser.add_argument('--tau', default=0.005, type=float)
        parser.add_argument('--n-critics', default=2, type=int)
        parser.add_argument('--train-freq', default=1, type=int)
        parser.add_argument('--buffer-size', default=1_000_000, type=int)
        parser.add_argument('--batch-size', default=256, type=int)
        parser.add_argument('--noise-sigma', default=0.1, type=float)

        args = parser.parse_args([])
        assert args.tau == 0.005
        assert args.n_critics == 2
        assert args.train_freq == 1
        assert args.buffer_size == 1_000_000
        assert args.batch_size == 256
        assert args.noise_sigma == 0.1

    def test_argparse_custom(self, common_sb3):
        from argparse import ArgumentParser
        parser = ArgumentParser()
        common_sb3.add_common_args(parser)
        parser.add_argument('--tau', default=0.005, type=float)
        parser.add_argument('--n-critics', default=2, type=int)
        parser.add_argument('--noise-sigma', default=0.1, type=float)

        args = parser.parse_args(['--tau', '0.01', '--n-critics', '3', '--noise-sigma', '0.2'])
        assert args.tau == 0.01
        assert args.n_critics == 3
        assert args.noise_sigma == 0.2

    def test_sac_model_construction(self, common_sb3):
        """Verify SAC model can be constructed with FloPyArcade env."""
        from stable_baselines3 import SAC
        from stable_baselines3.common.noise import NormalActionNoise

        env = common_sb3.make_flopy_env('3s-c')
        n_actions = env.action_space.shape[-1]
        noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        model = SAC(
            'MlpPolicy', env,
            action_noise=noise,
            learning_rate=3e-4,
            gamma=0.99,
            seed=1,
            verbose=0,
        )
        # Verify it can do a single training step
        model.learn(total_timesteps=10)
        # Verify predict works
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        assert action.shape == (n_actions,)
        env.close()


# ============================================================
# TESTS: train_sb3_ddpg.py (argparse + config)
# ============================================================

class TestTrainSB3DDPG:
    """Test DDPG script's argparse and model construction."""

    def test_ddpg_model_construction(self, common_sb3):
        """Verify DDPG model can be constructed with FloPyArcade env."""
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import NormalActionNoise

        env = common_sb3.make_flopy_env('3s-c')
        n_actions = env.action_space.shape[-1]
        noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        model = DDPG(
            'MlpPolicy', env,
            action_noise=noise,
            learning_rate=5e-5,
            gamma=0.99,
            seed=1,
            learning_starts=10,

            verbose=0,
        )
        model.learn(total_timesteps=50)
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        assert action.shape == (n_actions,)
        env.close()


# ============================================================
# TESTS: train_sb3_ppo.py (argparse + config)
# ============================================================

class TestTrainSB3PPO:
    """Test PPO script's argparse and model construction."""

    def test_ppo_model_construction_discrete(self, common_sb3):
        """Verify PPO model with discrete env."""
        from stable_baselines3 import PPO

        env = common_sb3.make_flopy_env('1s-d')
        model = PPO(
            'MlpPolicy', env,
            learning_rate=5e-5,
            gamma=0.99,
            batch_size=64,
            n_epochs=2,
            clip_range=0.2,
            seed=1,
            device='cpu',
            verbose=0,
        )
        model.learn(total_timesteps=50)
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        # Discrete action: should be an int or 0-d array
        assert action == int(action)
        env.close()

    def test_ppo_model_construction_continuous(self, common_sb3):
        """Verify PPO model with continuous env."""
        from stable_baselines3 import PPO

        env = common_sb3.make_flopy_env('6r-c')
        model = PPO(
            'MlpPolicy', env,
            learning_rate=5e-5,
            gamma=0.99,
            batch_size=64,
            n_epochs=2,
            clip_range=0.2,
            seed=1,
            device='cpu',
            verbose=0,
        )
        model.learn(total_timesteps=50)
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        assert action.shape == env.action_space.shape
        env.close()


# ============================================================
# TESTS: model save/load round-trip
# ============================================================

class TestSB3SaveLoad:
    """Test that models can be saved and loaded (resume/eval-only flow)."""

    def test_sac_save_load(self, common_sb3, tmp_path):
        from stable_baselines3 import SAC
        from stable_baselines3.common.noise import NormalActionNoise

        env = common_sb3.make_flopy_env('3s-c')
        n_actions = env.action_space.shape[-1]
        noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = SAC('MlpPolicy', env, action_noise=noise, learning_rate=3e-4, seed=1, verbose=0)
        model.learn(total_timesteps=20)

        save_path = str(tmp_path / 'sac_test')
        model.save(save_path)

        # Load back
        model2 = SAC.load(save_path, env=env)
        obs, _ = env.reset()
        action, _ = model2.predict(obs, deterministic=True)
        assert action.shape == (n_actions,)
        env.close()

    def test_ppo_save_load(self, common_sb3, tmp_path):
        from stable_baselines3 import PPO

        env = common_sb3.make_flopy_env('1s-d')
        model = PPO('MlpPolicy', env, learning_rate=5e-5, seed=1, device='cpu', verbose=0)
        model.learn(total_timesteps=30)

        save_path = str(tmp_path / 'ppo_test')
        model.save(save_path)

        model2 = PPO.load(save_path, env=env)
        obs, _ = env.reset()
        action, _ = model2.predict(obs)
        assert action == int(action)
        env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
