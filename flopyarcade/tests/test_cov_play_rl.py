#!/usr/bin/env python3
"""Coverage tests for play3d, play_againstai, train_rllib_*, and arcade modules."""

import sys
import os
import io
import types
import importlib
import importlib.util
from unittest.mock import MagicMock, patch, PropertyMock, call
import pytest


# ============================================================
# HELPER: load a module from file path with mocked deps
# ============================================================

def _load_module_from_path(module_name, file_path):
    """Load a Python module from a file path. Caller must have set sys.modules mocks first."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _save_and_set_modules(mocks):
    """Save current sys.modules entries and replace with mocks. Returns saved dict."""
    saved = {}
    for name, mock in mocks.items():
        saved[name] = sys.modules.get(name, '__deleted__')
        sys.modules[name] = mock
    return saved


def _restore_modules(saved):
    """Restore sys.modules from saved dict."""
    for name, old in saved.items():
        if old == '__deleted__':
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


# ============================================================
# TESTS: play3d.py
# ============================================================

class TestPlay3D:
    """Tests for flopyarcade/play3d.py"""

    @pytest.fixture(autouse=True)
    def setup_play3d(self):
        """Mock heavy deps before importing play3d."""
        mpl = MagicMock()
        mpl.use = MagicMock()
        mpl.pyplot = MagicMock()
        mpl.cm = MagicMock()
        mpl.colors = MagicMock()

        render3d = MagicMock()
        render3d.Render3D = MagicMock()
        render3d.render_environment_3d = MagicMock(return_value='/tmp/test.png')

        mocks = {
            'matplotlib': mpl,
            'matplotlib.pyplot': mpl.pyplot,
            'matplotlib.cm': mpl.cm,
            'matplotlib.colors': mpl.colors,
            'PIL': MagicMock(),
            'PIL.Image': MagicMock(),
            'imageio': MagicMock(),
            'render3d': render3d,
            'flopyarcade': MagicMock(FloPyEnv=MagicMock()),
        }

        self._saved = _save_and_set_modules(mocks)
        self._mocks = mocks
        self._render3d = render3d
        self._mpl = mpl

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        play3d_path = os.path.join(project_dir, 'play3d.py')
        self.play3d = _load_module_from_path('play3d_test_mod', play3d_path)
        yield

        sys.modules.pop('play3d_test_mod', None)
        _restore_modules(self._saved)

    def test_mockenv_init(self):
        env = self.play3d.MockEnv(nrow=10, ncol=10)
        assert env.nrow == 10
        assert env.ncol == 10
        assert env.nlay == 1
        assert env.extentX == 10 * 100.0
        assert env.extentY == 10 * 100.0
        assert env.done is False
        assert env.rewardCurrent == 142
        assert len(env.heads.shape) == 2

    def test_mockenv_reset(self):
        env = self.play3d.MockEnv(nrow=5, ncol=5)
        env.particleX = 999.0
        result = env.reset()
        assert env.particleX == env.extentX * 0.05
        assert env.done is False
        assert 'heads' in result

    def test_mockenv_step(self):
        env = self.play3d.MockEnv(nrow=5, ncol=5)
        px_before = env.particleX
        coords, reward, done, truncated, info = env.step()
        assert coords[0] > px_before
        assert isinstance(reward, float)
        assert done is False
        assert isinstance(info, dict)

    def test_mockenv_step_done(self):
        env = self.play3d.MockEnv(nrow=5, ncol=5)
        env.particleX = env.extentX * 0.96
        coords, reward, done, truncated, info = env.step()
        assert done is True

    def test_mockenv_distance_to_well(self):
        env = self.play3d.MockEnv()
        d = env.distance_to_well([0, 0], [3, 4])
        assert abs(d - 5.0) < 1e-9

    def test_mockenv_generate_heads(self):
        env = self.play3d.MockEnv(nrow=8, ncol=8)
        heads = env._generate_heads()
        assert heads.shape == (8, 8)
        assert heads.min() < heads.max()

    def test_mockenv_well_position(self):
        env = self.play3d.MockEnv(nrow=10, ncol=10)
        assert env.wellX == env.extentX * 0.7
        assert env.wellY == env.extentY * 0.45

    def test_mockenv_particle_start(self):
        env = self.play3d.MockEnv(nrow=10, ncol=10)
        assert env.particleX == env.extentX * 0.05
        assert env.particleY == env.extentY * 0.5

    def test_mockenv_step_multiple(self):
        env = self.play3d.MockEnv(nrow=10, ncol=10)
        x0 = env.particleX
        for _ in range(5):
            env.step()
        assert env.particleX > x0
        assert env.timeStep == 5

    def test_render_frame(self):
        env = self.play3d.MockEnv(nrow=5, ncol=5)
        renderer = MagicMock()
        import numpy as np
        X = np.zeros((5, 5))
        Y = np.zeros((5, 5))
        Z = np.ones((5, 5))
        head_nodes = np.ones((5, 5)) * 55
        renderer._build_grid.return_value = (X, Y, Z, head_nodes)
        renderer.trajectory = [[0, 0, 1], [1, 1, 2]]
        renderer._get_particle_position.return_value = (1.0, 1.0, 1.0)
        renderer._get_well_position.return_value = (3.0, 3.0, 1.0)
        env.wellRadius = 100.0
        env.rewardCurrent = 100

        self._mpl.pyplot.figure.return_value = MagicMock()
        fig = self.play3d._render_frame(env, renderer, 1, 10)
        assert fig is not None

    def test_run_live(self):
        out_path = '/tmp/test_live.png'
        self._mocks['flopyarcade'].FloPyEnv.return_value = MagicMock()
        self._render3d.render_environment_3d.return_value = out_path
        result = self.play3d.run_live(output_path=out_path, envtype='1s-d')
        assert result == out_path
        self._mocks['flopyarcade'].FloPyEnv.assert_called()

    def test_main_demo_mode(self):
        with patch.object(sys, 'argv', ['play3d.py', '--mode', 'demo', '--output', '/tmp/test.gif',
                                        '--nsteps', '1', '--nrow', '5', '--ncol', '5']):
            with patch.object(self.play3d, 'run_demo') as mock_demo:
                self.play3d.main()
                mock_demo.assert_called_once()

    def test_main_live_mode(self):
        with patch.object(sys, 'argv', ['play3d.py', '--mode', 'live', '--output', '/tmp/test.png',
                                        '--envtype', '1s-d']):
            with patch.object(self.play3d, 'run_live') as mock_live:
                self.play3d.main()
                mock_live.assert_called_once()


# ============================================================
# TESTS: play_againstai.py
# ============================================================

class TestPlayAgainstAI:
    """Tests for flopyarcade/play_againstai.py"""

    @pytest.fixture(autouse=True)
    def setup_pa(self):
        ray_mock = MagicMock()
        ray_mock.tune = MagicMock()
        ray_mock.tune.run = MagicMock(return_value=MagicMock())
        ray_mock.tune.registry = MagicMock()
        ray_mock.tune.registry.register_env = MagicMock()
        ray_mock.init = MagicMock()
        ray_mock.rllib = MagicMock()
        ray_mock.rllib.agents = MagicMock()
        ray_mock.rllib.agents.dqn = MagicMock()
        ray_mock.rllib.agents.dqn.ApexTrainer = MagicMock()

        flopyarcade_mock = MagicMock(FloPyEnv=MagicMock(), FloPyAgent=MagicMock())

        mocks = {
            'ray': ray_mock,
            'ray.tune': ray_mock.tune,
            'ray.tune.registry': ray_mock.tune.registry,
            'ray.rllib': ray_mock.rllib,
            'ray.rllib.agents': ray_mock.rllib.agents,
            'ray.rllib.agents.dqn': ray_mock.rllib.agents.dqn,
            'flopyarcade': flopyarcade_mock,
            'flopyarcade.flopyarcade': flopyarcade_mock,
            'flopyarcade.agent': MagicMock(FloPyAgent=MagicMock()),
            'gymnasium': MagicMock(),
            'gymnasium.spaces': MagicMock(),
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
            'tensorflow': MagicMock(),
            'tensorflow.keras': MagicMock(),
        }

        self._saved = _save_and_set_modules(mocks)
        self._ray = ray_mock
        self._flopyarcade_mock = flopyarcade_mock

        self._old_argv = sys.argv
        sys.argv = ['play_againstai.py']

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pa_path = os.path.join(project_dir, 'play_againstai.py')
        self.pa = _load_module_from_path('play_againstai_test_mod', pa_path)
        yield

        sys.argv = self._old_argv
        sys.modules.pop('play_againstai_test_mod', None)
        _restore_modules(self._saved)

    def test_str_to_bool_true(self):
        assert self.pa.str_to_bool('true') is True
        assert self.pa.str_to_bool('True') is True
        assert self.pa.str_to_bool('1') is True
        assert self.pa.str_to_bool('yes') is True
        assert self.pa.str_to_bool('t') is True
        assert self.pa.str_to_bool('y') is True

    def test_str_to_bool_false(self):
        assert self.pa.str_to_bool('false') is False
        assert self.pa.str_to_bool('False') is False
        assert self.pa.str_to_bool('0') is False
        assert self.pa.str_to_bool('no') is False
        assert self.pa.str_to_bool('f') is False
        assert self.pa.str_to_bool('n') is False

    def test_str_to_bool_bool_passthrough(self):
        assert self.pa.str_to_bool(True) is True
        assert self.pa.str_to_bool(False) is False

    def test_str_to_bool_invalid(self):
        with pytest.raises(ValueError):
            self.pa.str_to_bool('maybe')

    def test_env_creator(self):
        env = self.pa.env_creator({})
        self._flopyarcade_mock.FloPyEnv.assert_called()

    def test_config_structure(self):
        assert 'env' in self.pa.config
        assert 'model' in self.pa.config
        assert 'framework' in self.pa.config
        assert 'seed' in self.pa.config

    def test_config_stop_criteria(self):
        assert 'training_iteration' in self.pa.config_stopCriteria
        assert 'timesteps_total' in self.pa.config_stopCriteria
        assert 'episode_reward_mean' in self.pa.config_stopCriteria

    def test_envtype_global(self):
        assert self.pa.ENVTYPE == '1s-d'


# ============================================================
# TESTS: train_rllib_apexdqn.py
# ============================================================

class TestTrainRLlibApexDQN:
    @pytest.fixture(autouse=True)
    def setup_apexdqn(self):
        ray_mock = MagicMock()
        ray_mock.tune = MagicMock()
        ray_mock.tune.run = MagicMock(return_value=MagicMock())
        ray_mock.tune.registry = MagicMock()
        ray_mock.tune.registry.register_env = MagicMock()
        ray_mock.init = MagicMock()
        ray_mock.rllib = MagicMock()
        ray_mock.rllib.algorithms = MagicMock()
        ray_mock.rllib.algorithms.dqn = MagicMock()
        ray_mock.rllib.algorithms.dqn.ApexTrainer = MagicMock()

        flopyarcade_mock = MagicMock(FloPyEnv=MagicMock(), FloPyAgent=MagicMock())

        mocks = {
            'ray': ray_mock,
            'ray.tune': ray_mock.tune,
            'ray.tune.registry': ray_mock.tune.registry,
            'ray.rllib': ray_mock.rllib,
            'ray.rllib.algorithms': ray_mock.rllib.algorithms,
            'ray.rllib.algorithms.dqn': ray_mock.rllib.algorithms.dqn,
            'flopyarcade': flopyarcade_mock,
            'flopyarcade.flopyarcade': flopyarcade_mock,
            'gymnasium': MagicMock(),
            'gymnasium.spaces': MagicMock(Box=MagicMock(), Discrete=MagicMock()),
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
        }

        self._saved = _save_and_set_modules(mocks)
        self._flopyarcade = flopyarcade_mock
        self._old_argv = sys.argv
        sys.argv = ['train_rllib_apexdqn.py']

        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        apex_path = os.path.join(project_dir, 'flopyarcade', 'examples', 'train_rllib_apexdqn.py')
        self.apex = _load_module_from_path('train_apex_test_mod', apex_path)
        yield

        sys.argv = self._old_argv
        sys.modules.pop('train_apex_test_mod', None)
        _restore_modules(self._saved)

    def test_str_to_bool(self):
        assert self.apex.str_to_bool('true') is True
        assert self.apex.str_to_bool('false') is False
        assert self.apex.str_to_bool(True) is True
        with pytest.raises(ValueError):
            self.apex.str_to_bool('xyz')

    def test_env_creator(self):
        self._flopyarcade.FloPyEnv.return_value = MagicMock()
        env = self.apex.env_creator({})
        self._flopyarcade.FloPyEnv.assert_called()

    def test_config_exists(self):
        assert self.apex.config is not None
        assert 'env' in self.apex.config

    def test_config_stop_criteria(self):
        assert 'episode_reward_mean' in self.apex.config_stopCriteria

    def test_envtype(self):
        assert self.apex.ENVTYPE == '1s-d'


# ============================================================
# TESTS: train_rllib_appo.py
# ============================================================

class TestTrainRLlibAPPO:
    @pytest.fixture(autouse=True)
    def setup_appo(self):
        ray_mock = MagicMock()
        ray_mock.tune = MagicMock()
        ray_mock.tune.run = MagicMock(return_value=MagicMock())
        ray_mock.tune.registry = MagicMock()
        ray_mock.tune.registry.register_env = MagicMock()
        ray_mock.init = MagicMock()
        ray_mock.rllib = MagicMock()
        ray_mock.rllib.algorithms = MagicMock()
        ray_mock.rllib.algorithms.appo = MagicMock()
        ray_mock.rllib.algorithms.appo.APPOTrainer = MagicMock()

        flopyarcade_mock = MagicMock(FloPyEnv=MagicMock(), FloPyAgent=MagicMock())

        mocks = {
            'ray': ray_mock,
            'ray.tune': ray_mock.tune,
            'ray.tune.registry': ray_mock.tune.registry,
            'ray.rllib': ray_mock.rllib,
            'ray.rllib.algorithms': ray_mock.rllib.algorithms,
            'ray.rllib.algorithms.appo': ray_mock.rllib.algorithms.appo,
            'flopyarcade': flopyarcade_mock,
            'flopyarcade.flopyarcade': flopyarcade_mock,
            'gymnasium': MagicMock(),
            'gymnasium.spaces': MagicMock(),
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
        }

        self._saved = _save_and_set_modules(mocks)
        self._flopyarcade = flopyarcade_mock
        self._old_argv = sys.argv
        sys.argv = ['train_rllib_appo.py']

        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        appo_path = os.path.join(project_dir, 'flopyarcade', 'examples', 'train_rllib_appo.py')
        self.appo = _load_module_from_path('train_appo_test_mod', appo_path)
        yield

        sys.argv = self._old_argv
        sys.modules.pop('train_appo_test_mod', None)
        _restore_modules(self._saved)

    def test_str_to_bool(self):
        assert self.appo.str_to_bool('true') is True
        assert self.appo.str_to_bool('false') is False
        assert self.appo.str_to_bool(True) is True
        with pytest.raises(ValueError):
            self.appo.str_to_bool('xyz')

    def test_env_creator(self):
        self._flopyarcade.FloPyEnv.return_value = MagicMock()
        env = self.appo.env_creator({})
        self._flopyarcade.FloPyEnv.assert_called()

    def test_config_exists(self):
        assert self.appo.config is not None

    def test_envtype(self):
        assert self.appo.ENVTYPE == '1s-d'


# ============================================================
# TESTS: train_rllib_impala.py
# ============================================================

class TestTrainRLlibIMPALA:
    @pytest.fixture(autouse=True)
    def setup_impala(self):
        ray_mock = MagicMock()
        ray_mock.tune = MagicMock()
        ray_mock.tune.run = MagicMock(return_value=MagicMock())
        ray_mock.tune.registry = MagicMock()
        ray_mock.tune.registry.register_env = MagicMock()
        ray_mock.init = MagicMock()
        ray_mock.rllib = MagicMock()
        ray_mock.rllib.algorithms = MagicMock()
        ray_mock.rllib.algorithms.impala = MagicMock()
        ray_mock.rllib.algorithms.impala.IMPALATrainer = MagicMock()

        flopyarcade_mock = MagicMock(FloPyEnv=MagicMock(), FloPyAgent=MagicMock())

        mocks = {
            'ray': ray_mock,
            'ray.tune': ray_mock.tune,
            'ray.tune.registry': ray_mock.tune.registry,
            'ray.rllib': ray_mock.rllib,
            'ray.rllib.algorithms': ray_mock.rllib.algorithms,
            'ray.rllib.algorithms.impala': ray_mock.rllib.algorithms.impala,
            'flopyarcade': flopyarcade_mock,
            'flopyarcade.flopyarcade': flopyarcade_mock,
            'gymnasium': MagicMock(),
            'gymnasium.spaces': MagicMock(),
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
        }

        self._saved = _save_and_set_modules(mocks)
        self._flopyarcade = flopyarcade_mock
        self._old_argv = sys.argv
        sys.argv = ['train_rllib_impala.py']

        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        impala_path = os.path.join(project_dir, 'flopyarcade', 'examples', 'train_rllib_impala.py')
        self.impala = _load_module_from_path('train_impala_test_mod', impala_path)
        yield

        sys.argv = self._old_argv
        sys.modules.pop('train_impala_test_mod', None)
        _restore_modules(self._saved)

    def test_str_to_bool(self):
        assert self.impala.str_to_bool('true') is True
        assert self.impala.str_to_bool('false') is False
        assert self.impala.str_to_bool(True) is True
        with pytest.raises(ValueError):
            self.impala.str_to_bool('xyz')

    def test_env_creator(self):
        self._flopyarcade.FloPyEnv.return_value = MagicMock()
        env = self.impala.env_creator({})
        self._flopyarcade.FloPyEnv.assert_called()

    def test_config_exists(self):
        assert self.impala.config is not None

    def test_envtype(self):
        assert self.impala.ENVTYPE == '1s-d'


# ============================================================
# TESTS: train_rllib_ppo.py
# ============================================================

class TestTrainRLlibPPO:
    @pytest.fixture(autouse=True)
    def setup_ppo(self):
        ray_mock = MagicMock()
        ray_mock.tune = MagicMock()
        ray_mock.tune.run = MagicMock(return_value=MagicMock())
        ray_mock.tune.registry = MagicMock()
        ray_mock.tune.registry.register_env = MagicMock()
        ray_mock.init = MagicMock()
        ray_mock.rllib = MagicMock()
        ray_mock.rllib.algorithms = MagicMock()
        ray_mock.rllib.algorithms.ppo = MagicMock()
        ray_mock.rllib.algorithms.ppo.PPOTrainer = MagicMock()

        flopyarcade_mock = MagicMock(FloPyEnv=MagicMock(), FloPyAgent=MagicMock())

        mocks = {
            'ray': ray_mock,
            'ray.tune': ray_mock.tune,
            'ray.tune.registry': ray_mock.tune.registry,
            'ray.rllib': ray_mock.rllib,
            'ray.rllib.algorithms': ray_mock.rllib.algorithms,
            'ray.rllib.algorithms.ppo': ray_mock.rllib.algorithms.ppo,
            'flopyarcade': flopyarcade_mock,
            'flopyarcade.flopyarcade': flopyarcade_mock,
            'gymnasium': MagicMock(),
            'gymnasium.spaces': MagicMock(),
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
        }

        self._saved = _save_and_set_modules(mocks)
        self._flopyarcade = flopyarcade_mock
        self._old_argv = sys.argv
        sys.argv = ['train_rllib_ppo.py']

        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ppo_path = os.path.join(project_dir, 'flopyarcade', 'examples', 'train_rllib_ppo.py')
        self.ppo = _load_module_from_path('train_ppo_test_mod', ppo_path)
        yield

        sys.argv = self._old_argv
        sys.modules.pop('train_ppo_test_mod', None)
        _restore_modules(self._saved)

    def test_str_to_bool(self):
        assert self.ppo.str_to_bool('true') is True
        assert self.ppo.str_to_bool('false') is False
        assert self.ppo.str_to_bool(True) is True
        with pytest.raises(ValueError):
            self.ppo.str_to_bool('xyz')

    def test_env_creator(self):
        self._flopyarcade.FloPyEnv.return_value = MagicMock()
        env = self.ppo.env_creator({})
        self._flopyarcade.FloPyEnv.assert_called()

    def test_config_exists(self):
        assert self.ppo.config is not None

    def test_envtype(self):
        assert self.ppo.ENVTYPE == '1s-d'


# ============================================================
# TESTS: arcade.py (FloPyArcade class)
# ============================================================

class TestArcade:
    """Tests for flopyarcade/arcade.py - FloPyArcade class."""

    @pytest.fixture(autouse=True)
    def setup_arcade(self):
        """Mock all heavy deps for arcade.py using exec-based loading."""
        flopy_mock = MagicMock()
        gym_mock = MagicMock()
        tf_mock = MagicMock()
        mpl_mock = MagicMock()
        pathos_mock = MagicMock()
        pathos_mock.helpers = MagicMock()
        pathos_mock.helpers.mp = MagicMock()
        pathos_mock.helpers.mp.context = MagicMock()
        pathos_mock.helpers.mp.context._force_start_method = MagicMock()
        pathos_mock.multiprocessing = MagicMock()
        pathos_mock.multiprocessing.cpu_count = MagicMock(return_value=4)
        pathos_mock.multiprocessing.Pool = MagicMock()
        pathos_mock.pools = MagicMock()
        pathos_mock.pools._ThreadPool = MagicMock()

        flopyarcade_pkg_mock = MagicMock()
        flopyarcade_pkg_mock.FloPyEnv = MagicMock()
        flopyarcade_pkg_mock.FloPyEnvSurrogate = MagicMock()

        agent_mock = MagicMock()
        agent_mock.FloPyAgent = MagicMock()
        agent_mock.FloPyAgent.return_value = MagicMock(wrkspc='/tmp/wrkspc')

        mocks = {
            'flopy': flopy_mock,
            'flopy.modflow': flopy_mock.modflow,
            'flopy.mf6': flopy_mock.mf6,
            'flopy.mf6.modflow': flopy_mock.mf6.modflow,
            'flopy.mf6.modflow.mfgwfwel': flopy_mock.mf6.modflow.mfgwfwel,
            'flopy.mf6.utils': flopy_mock.mf6.utils,
            'flopy.mf6.utils.binarygrid_util': flopy_mock.mf6.utils.binarygrid_util,
            'flopy.modpath': flopy_mock.modpath,
            'flopy.plot': flopy_mock.plot,
            'flopy.utils': flopy_mock.utils,
            'flopy.utils.binaryfile': flopy_mock.utils.binaryfile,
            'gymnasium': gym_mock,
            'gymnasium.spaces': gym_mock.spaces,
            'gymnasium.wrappers': gym_mock.wrappers,
            'tensorflow': tf_mock,
            'tensorflow.keras': tf_mock.keras,
            'tensorflow.keras.initializers': tf_mock.keras.initializers,
            'tensorflow.keras.layers': tf_mock.keras.layers,
            'tensorflow.keras.models': tf_mock.keras.models,
            'tensorflow.keras.optimizers': tf_mock.keras.optimizers,
            'tensorflow.keras.random': tf_mock.keras.random,
            'tensorflow.keras.backend': tf_mock.keras.backend,
            'tensorflow.compat': tf_mock.compat,
            'tensorflow.compat.v1': tf_mock.compat.v1,
            'matplotlib': mpl_mock,
            'matplotlib.pyplot': mpl_mock.pyplot,
            'matplotlib.ticker': mpl_mock.ticker,
            'matplotlib.colors': mpl_mock.colors,
            'matplotlib.transforms': mpl_mock.transforms,
            'matplotlib.text': mpl_mock.text,
            'pathos': pathos_mock,
            'pathos.helpers': pathos_mock.helpers,
            'pathos.multiprocessing': pathos_mock.multiprocessing,
            'pathos.pools': pathos_mock.pools,
            'xmipy': MagicMock(XmiWrapper=MagicMock()),
            'tqdm': MagicMock(tqdm=MagicMock()),
            'imageio': MagicMock(get_writer=MagicMock(), imread=MagicMock()),
            'joblib': MagicMock(dump=MagicMock(), load=MagicMock()),
            'flopyarcade': flopyarcade_pkg_mock,
            'flopyarcade.flopyarcade': flopyarcade_pkg_mock,
            'flopyarcade.agent': agent_mock,
            'IPython': MagicMock(),
            'IPython.display': MagicMock(),
        }

        self._saved = _save_and_set_modules(mocks)
        self._flopyarcade_pkg = flopyarcade_pkg_mock
        self._agent_mock = agent_mock

        # Load arcade.py by reading source and exec in controlled namespace
        # This avoids the relative import issue with sys.modules
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        arcade_path = os.path.join(project_dir, 'arcade.py')

        with open(arcade_path, 'r') as f:
            source = f.read()

        # Create module namespace with pre-injected relative imports
        mod = types.ModuleType('flopyarcade.arcade_test')
        mod.__package__ = 'flopyarcade'
        mod.__name__ = 'flopyarcade.arcade_test'
        mod.__file__ = arcade_path
        sys.modules['flopyarcade.arcade_test'] = mod

        # Pre-inject the relative imports so they don't trigger real imports
        mod.FloPyEnv = flopyarcade_pkg_mock.FloPyEnv
        mod.FloPyEnvSurrogate = flopyarcade_pkg_mock.FloPyEnvSurrogate

        exec(compile(source, arcade_path, 'exec'), mod.__dict__)
        self.arcade_mod = mod
        yield

        sys.modules.pop('flopyarcade.arcade_test', None)
        _restore_modules(self._saved)

    def test_init_config(self):
        arcade = self.arcade_mod.FloPyArcade(
            agent=None,
            modelName='TestModel',
            NAGENTSTEPS=100,
            ENVTYPE='1s-d',
            nLay=1,
            nRow=50,
            nCol=50,
            flagRender=False,
            flagBlit=True
        )
        assert arcade.ENVTYPE == '1s-d'
        assert arcade.MODELNAME == 'TestModel'
        assert arcade.NAGENTSTEPS == 100
        assert arcade.nLay == 1
        assert arcade.nRow == 50
        assert arcade.nCol == 50
        assert arcade.RENDER is False
        assert arcade.BLIT is True
        assert arcade.done is False
        assert '1s-d' in arcade.ENVTYPES

    def test_setup_env_with_env(self):
        mock_env = MagicMock()
        mock_env.ENVTYPE = '1s-d'
        mock_env.observationsVectorNormalized = [1, 2, 3]
        mock_env.done = False
        mock_env.actionSpace = ['a', 'b', 'c']

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )
        arcade._setup_env(mock_env, '1s-d', None)
        assert arcade.env is mock_env

    def test_setup_env_creates_flopyenv(self):
        mock_env = MagicMock()

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )

        with patch.object(self.arcade_mod, 'FloPyEnv', return_value=mock_env) as mock_flopyenv:
            arcade._setup_env(None, '1s-d', 42)
            mock_flopyenv.assert_called()

    def test_setup_env_surrogate(self):
        mock_env = MagicMock()

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True,
            surrogateSimulator='my_surrogate'
        )

        with patch.object(self.arcade_mod, 'FloPyEnvSurrogate', return_value=mock_env) as mock_surrogate:
            arcade._setup_env(None, '1s-d', None)
            mock_surrogate.assert_called()

    def test_setup_game(self):
        mock_env = MagicMock()
        mock_env.observationsVectorNormalized = [1, 2, 3]
        mock_env.done = False
        mock_env.actionSpace = ['a', 'b']
        mock_env.heads = [[1.0]]
        mock_env.wellCoords = [5, 5]
        mock_env.success = True
        mock_env.trajectories = []

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True,
            keepTimeSeries=True
        )
        arcade.env = mock_env
        arcade._setup_game()

        assert arcade.observations == [1, 2, 3]
        assert arcade.done is False
        assert arcade.actionSpace == ['a', 'b']
        assert arcade.rewardTotal == 0.
        assert 'statesNormalized' in arcade.timeSeries

    def test_get_action_predefined(self):
        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True,
            actions=['a', 'b', 'c']
        )
        arcade.timeSteps = 1
        action = arcade._get_action()
        assert action == 'b'

    def test_get_action_manual(self):
        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True,
            flagManualControl=True
        )
        arcade.timeSteps = 0
        arcade.env = MagicMock(keyPressed='left', actionType='string')
        arcade.agentObj = MagicMock()
        arcade.agentObj.getAction.return_value = 'left'

        action = arcade._get_action()
        assert action == 'left'
        arcade.agentObj.getAction.assert_called()

    def test_get_action_random(self):
        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )
        arcade.timeSteps = 0
        arcade.env = MagicMock(actionType='string')
        arcade.agentObj = MagicMock()
        arcade.agentObj.getAction.return_value = 'random_action'

        action = arcade._get_action()
        assert action == 'random_action'
        arcade.agentObj.getAction.assert_called_with('random', actionType='string')

    def test_get_action_modelname_load(self):
        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True,
            modelNameLoad='/path/to/model'
        )
        arcade.timeSteps = 0
        arcade.env = MagicMock(observationsVectorNormalized=[1,2,3], actionType='string')
        arcade.agentObj = MagicMock()
        arcade.agentObj.getAction.return_value = 'loaded_action'

        action = arcade._get_action()
        assert action == 'loaded_action'
        arcade.agentObj.getAction.assert_called_with(
            'modelNameLoad',
            modelNameLoad='/path/to/model',
            state=[1,2,3],
            actionType='string'
        )

    def test_get_action_with_agent(self):
        mock_agent = MagicMock()
        arcade = self.arcade_mod.FloPyArcade(
            agent=mock_agent, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )
        arcade.timeSteps = 0
        arcade.env = MagicMock(observationsVectorNormalized=[1,2,3], actionType='string')
        arcade.agentObj = MagicMock()
        arcade.agentObj.getAction.return_value = 'agent_action'

        action = arcade._get_action()
        assert action == 'agent_action'

    def test_run_game_loop(self):
        mock_env = MagicMock()
        mock_env.observationsVectorNormalized = [1, 2, 3]
        mock_env.done = False
        mock_env.actionSpace = ['a', 'b']
        mock_env.step = MagicMock(side_effect=[
            ([1,2,3], 10.0, False, {}, {}),
            ([1,2,3], 20.0, True, {}, {}),
        ])

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )
        arcade.env = mock_env
        arcade.done = False
        arcade.timeSteps = 0
        arcade.agentObj = MagicMock()
        arcade.agentObj.getAction.return_value = 'a'
        arcade.rewardTotal = 0.

        arcade._run_game_loop()

        assert mock_env.step.call_count == 2
        assert arcade.rewardTotal == 30.0
        assert arcade.done is True

    def test_run_game_loop_immediate_done(self):
        mock_env = MagicMock()
        mock_env.step = MagicMock(return_value=([1,2,3], 10.0, True, {}, {}))

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )
        arcade.env = mock_env
        arcade.done = True
        arcade.timeSteps = 0
        arcade.rewardTotal = 0.

        arcade._run_game_loop()

        assert mock_env.step.call_count == 0

    def test_init_time_series(self):
        mock_env = MagicMock()
        mock_env.heads = [[1.0, 2.0]]
        mock_env.wellCoords = [5, 5]

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )
        arcade.env = mock_env
        arcade.done = False

        arcade._init_time_series([1, 2, 3])

        assert arcade.timeSeries['statesNormalized'] == [[1, 2, 3]]
        assert arcade.timeSeries['rewards'] == [0.]
        assert arcade.timeSeries['doneFlags'] == [False]
        assert arcade.timeSeries['heads'] == [[[1.0, 2.0]]]
        assert arcade.timeSeries['wellCoords'] == [[5, 5]]

    def test_update_time_series(self):
        mock_env = MagicMock()
        mock_env.observationsVectorNormalized = [4, 5, 6]
        mock_env.stressesVectorNormalized = [0.5, 0.3]
        mock_env.heads = [[3.0, 4.0]]
        mock_env.wellCoords = [10, 10]
        mock_env.success = True

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )
        arcade.env = mock_env
        arcade.done = True
        arcade._init_time_series([1, 2, 3])

        arcade._update_time_series(42.0, 'action_x')

        assert arcade.timeSeries['rewards'] == [0., 42.0]
        assert arcade.timeSeries['actions'] == ['action_x']
        assert arcade.timeSeries['statesNormalized'][-1] == [4, 5, 6]
        assert arcade.timeSeries['successFlags'][-1] == 1

    def test_update_time_series_not_done(self):
        mock_env = MagicMock()
        mock_env.observationsVectorNormalized = [1, 2]
        mock_env.stressesVectorNormalized = [0.5]
        mock_env.heads = [[1.0]]
        mock_env.wellCoords = [5, 5]

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )
        arcade.env = mock_env
        arcade.done = False
        arcade._init_time_series([0, 0])

        arcade._update_time_series(10.0, 'act')

        assert arcade.timeSeries['successFlags'][-1] == -1

    def test_finalize_game(self):
        mock_env = MagicMock()
        mock_env.success = True
        mock_env.ENVTYPE = '1s-d'
        mock_env.trajectories = [[1,2,3]]

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True,
            keepTimeSeries=True
        )
        arcade.env = mock_env
        arcade.done = False
        arcade.rewardTotal = 100.
        arcade.timeSteps = 5
        arcade.timeSeries = {}

        arcade._finalize_game(verbose=False, t0=0.0)

        assert arcade.success is True
        assert arcade.runtime is not None
        assert arcade.timeSeries['trajectories'] == [[1,2,3]]

    def test_finalize_game_verbose(self, capsys):
        mock_env = MagicMock()
        mock_env.success = True
        mock_env.ENVTYPE = '1s-d'
        mock_env.trajectories = []

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True,
            keepTimeSeries=True
        )
        arcade.env = mock_env
        arcade.rewardTotal = 150.
        arcade.timeSteps = 3
        arcade.timeSeries = {}

        arcade._finalize_game(verbose=True, t0=0.0)

        captured = capsys.readouterr()
        assert 'won' in captured.out
        assert '150' in captured.out

    def test_finalize_game_0s_c_teardown(self):
        mock_env = MagicMock()
        mock_env.success = False
        mock_env.ENVTYPE = '0s-c'
        mock_env.trajectories = []
        mock_env.teardown = MagicMock()

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )
        arcade.env = mock_env
        arcade.timeSteps = 1

        arcade._finalize_game(verbose=False, t0=0.0)

        mock_env.teardown.assert_called_once()

    def test_surrogate_finalize(self, capsys):
        mock_env = MagicMock()
        mock_env.success = False
        mock_env.ENVTYPE = '1s-d'
        mock_env.trajectories = []

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True,
            keepTimeSeries=True,
            surrogateSimulator='surrogate_model'
        )
        arcade.env = mock_env
        arcade.rewardTotal = 50.
        arcade.timeSteps = 5
        arcade.timeSeries = {}

        arcade._finalize_game(verbose=True, t0=0.0)

        captured = capsys.readouterr()
        assert 'surrogate' in captured.out
        assert 'lost' in captured.out

    def test_play_returns_reward(self):
        mock_env = MagicMock()
        mock_env.observationsVectorNormalized = [1, 2, 3]
        mock_env.done = False
        mock_env.actionSpace = ['a', 'b']
        mock_env.heads = [[1.0]]
        mock_env.wellCoords = [5, 5]
        mock_env.success = True
        mock_env.ENVTYPE = '1s-d'
        mock_env.trajectories = []
        mock_env.step = MagicMock(return_value=([1,2,3], 50.0, True, {}, {}))

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True
        )

        with patch.object(arcade, '_setup_env'):
            arcade.env = mock_env
            reward = arcade.play(env=mock_env, ENVTYPE='1s-d', returnReward=True)

        assert reward == 50.0

    def test_play_with_manual_control(self):
        mock_env = MagicMock()
        mock_env.observationsVectorNormalized = [1, 2, 3]
        mock_env.done = False
        mock_env.actionSpace = ['a', 'b']
        mock_env.heads = [[1.0]]
        mock_env.wellCoords = [5, 5]
        mock_env.success = True
        mock_env.ENVTYPE = '1s-d'
        mock_env.trajectories = []
        mock_env.keyPressed = 'left'
        mock_env.actionType = 'string'
        mock_env.step = MagicMock(return_value=([1,2,3], 10.0, True, {}, {}))

        arcade = self.arcade_mod.FloPyArcade(
            agent=None, ENVTYPE='1s-d', NAGENTSTEPS=10,
            nRow=10, nCol=10, flagBlit=True,
            flagManualControl=True
        )

        with patch.object(arcade, '_setup_env'):
            with patch.dict(self.arcade_mod.__dict__, {'sleep': MagicMock()}):
                arcade.env = mock_env
                arcade.play(env=mock_env, ENVTYPE='1s-d')

    def test_modelname_fallback(self):
        a1 = self.arcade_mod.FloPyArcade(modelName='Explicit', nRow=10, nCol=10, flagBlit=True)
        assert a1.MODELNAME == 'Explicit'

        # modelName defaults to 'FloPyArcade', so it's always truthy
        a2 = self.arcade_mod.FloPyArcade(nRow=10, nCol=10, flagBlit=True)
        assert a2.MODELNAME == 'FloPyArcade'

        # Passing None explicitly triggers modelNameLoad fallback
        a3 = self.arcade_mod.FloPyArcade(modelName=None, modelNameLoad='Loaded', nRow=10, nCol=10, flagBlit=True)
        assert a3.MODELNAME == 'Loaded'

        # Both None triggers 'untitled'
        a4 = self.arcade_mod.FloPyArcade(modelName=None, modelNameLoad=None, nRow=10, nCol=10, flagBlit=True)
        assert a4.MODELNAME == 'untitled'

    def test_animationfolder_fallback(self):
        a1 = self.arcade_mod.FloPyArcade(animationFolder='/anim', nRow=10, nCol=10, flagBlit=True)
        assert a1.ANIMATIONFOLDER == '/anim'

        a2 = self.arcade_mod.FloPyArcade(modelName='MyModel', nRow=10, nCol=10, flagBlit=True)
        assert a2.ANIMATIONFOLDER == 'MyModel'

    def test_envtypes_list(self):
        arcade = self.arcade_mod.FloPyArcade(nRow=10, nCol=10, flagBlit=True)
        expected = ['0s-c', '1s-d', '1s-c', '1r-d', '1r-c', '2s-d', '2s-c',
                    '2r-d', '2r-c', '3s-c', '6r-c']
        for et in expected:
            assert et in arcade.ENVTYPES
