"""Final coverage push: easy lines in well-covered files."""
import sys
import types
import numpy as np
from unittest.mock import MagicMock, patch
import pytest


def _add_ipykernel():
    """Add fake ipykernel to sys.modules so `if 'ipykernel' in modules` is True."""
    if 'ipykernel' not in sys.modules:
        fake = types.ModuleType('ipykernel')
        sys.modules['ipykernel'] = fake
    if 'IPython' not in sys.modules:
        fake_ipy = types.ModuleType('IPython')
        fake_ipy.display = MagicMock()
        sys.modules['IPython'] = fake_ipy
    if 'IPython.display' not in sys.modules:
        sys.modules['IPython.display'] = MagicMock()


class TestFloPyEnvPickle:
    """Cover FloPyEnv.__getstate__ and __setstate__ (flopyarcade.py:181-187)."""

    def test_getstate(self):
        from flopyarcade.flopyarcade import FloPyEnv
        # Create a minimal object that has the attributes
        env = FloPyEnv.__new__(FloPyEnv)
        env._obs_hasher = 'some_hash'
        env.x = 1
        state = FloPyEnv.__getstate__(env)
        assert '_obs_hasher' not in state
        assert state['x'] == 1

    def test_setstate(self):
        from flopyarcade.flopyarcade import FloPyEnv
        env = FloPyEnv.__new__(FloPyEnv)
        FloPyEnv.__setstate__(env, {'x': 2, 'y': 3})
        assert env.x == 2
        assert env.y == 3
        assert env._obs_hasher is None


class TestObservationsUnnormalize:
    """Cover ObservationsMixin.unnormalize (observations.py:792-803)."""

    def test_unnormalize(self):
        from flopyarcade.observations import ObservationsMixin
        obs = MagicMock()
        obs.minX = -100.0
        obs.extentX = 500.0
        obs.maxH = 80.0
        obs.minQ = -0.01
        obs.rewardMax = 200.0

        data = {
            'particleCoords': [10.0, 20.0],
            'heads': 50.0,
            'wellQ': -0.005,
            'wellCoords': [100.0, 200.0],
            'rewards': 150.0,
        }

        result = ObservationsMixin.unnormalize(obs, data)
        assert result is data
        assert obs._scale_data.call_count == 5


class TestIpykernelLines:
    """Cover line ~83 (ipykernel check) in multiple modules via reload."""

    def test_reload_flopyarcade(self):
        _add_ipykernel()
        import importlib
        import flopyarcade.flopyarcade as mod
        importlib.reload(mod)

    def test_reload_observations(self):
        _add_ipykernel()
        import importlib
        import flopyarcade.observations as mod
        importlib.reload(mod)

    def test_reload_modflow_runner(self):
        _add_ipykernel()
        import importlib
        import flopyarcade.modflow_runner as mod
        importlib.reload(mod)

    def test_reload_modpath_runner(self):
        _add_ipykernel()
        import importlib
        import flopyarcade.modpath_runner as mod
        importlib.reload(mod)

    def test_reload_actions(self):
        _add_ipykernel()
        import importlib
        import flopyarcade.actions as mod
        importlib.reload(mod)

    def test_reload_reward(self):
        _add_ipykernel()
        import importlib
        import flopyarcade.reward as mod
        importlib.reload(mod)


class TestMF6MP7RunModpath7:
    """Cover mf6_mp7_runner.run_modpath7 (lines ~219-227)."""

    def test_run_modpath7(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        runner = MagicMock(spec=MF6MP7Runner)
        runner._s = MagicMock()
        runner._s.timeStep = 2  # skip _init_mp7
        runner._s._transform_particle_coords = MagicMock()
        runner._write_mp7_particle = MagicMock()
        runner._s.mp7 = MagicMock()
        runner._s.mp7.run_model = MagicMock()
        MF6MP7Runner.run_modpath7(runner)
        runner._s.mp7.run_model.assert_called_once_with(silent=True)

    def test_run_modpath7_init(self):
        """Cover _init_mp7 path (timeStep==1)."""
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        runner = MagicMock(spec=MF6MP7Runner)
        runner._s = MagicMock()
        runner._s.timeStep = 1
        runner._s._transform_particle_coords = MagicMock()
        runner._init_mp7 = MagicMock()
        runner._write_mp7_particle = MagicMock()
        runner._s.mp7 = MagicMock()
        runner._s.mp7.run_model = MagicMock()
        MF6MP7Runner.run_modpath7(runner)
        runner._init_mp7.assert_called_once()
