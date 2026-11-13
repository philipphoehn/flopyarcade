#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Coverage tests for play.py, play_againstai.py, archive_hdf5.py,
benchmark.py, mf6_prt_runner.py, __init__.py, env_config.py.
"""

import csv
import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# env_config.py
# ---------------------------------------------------------------------------


class TestEnvConfig:
    """Cover env_config.py (3 stmts, 3 missed)."""

    def test_import_and_keys(self):
        from flopyarcade.env_config import ENV_CONFIG, REWARD_DISPATCH
        assert '1s-d' in ENV_CONFIG
        assert ENV_CONFIG['1s-d']['action_discrete'] == ['down', 'up', 'keep']
        assert ENV_CONFIG['1s-d']['reward'] == 'trajectory'

    def test_reward_dispatch(self):
        from flopyarcade.env_config import REWARD_DISPATCH
        assert REWARD_DISPATCH['trajectory'] == 'calculateGameRewardTrajectory'
        assert REWARD_DISPATCH['operation_cost'] == 'calculateGameRewardOperationCost'

    def test_all_envtypes_present(self):
        from flopyarcade.env_config import ENV_CONFIG
        expected = ['0s-c', '1s-d', '1s-c', '2s-d', '3s-d', '4s-d', '5s-d', '6s-d']
        for e in expected:
            assert e in ENV_CONFIG, f'{e} missing'
            assert ENV_CONFIG[e]['n'] is not None


# ---------------------------------------------------------------------------
# __init__.py
# ---------------------------------------------------------------------------


class TestInit:
    """Cover __init__.py (37 stmts, 19 missed)."""

    def test_version_and_author(self):
        import flopyarcade
        assert flopyarcade.__version__ == '0.7.0'
        assert flopyarcade.__author__ == 'Philipp Hoehn'

    def test_all_exports(self):
        import flopyarcade
        for name in flopyarcade.__all__:
            assert hasattr(flopyarcade, name), f'{name} not found'

    def test_flopyenv_importable(self):
        from flopyarcade import FloPyEnv
        assert FloPyEnv is not None

    def test_flopyarcade_importable(self):
        from flopyarcade import FloPyArcade
        assert FloPyArcade is not None


# ---------------------------------------------------------------------------
# play.py
# ---------------------------------------------------------------------------


class TestPlay:
    """Cover play.py (63 stmts, 63 missed)."""

    def test_str_to_bool_valid(self):
        from flopyarcade.play import str_to_bool
        assert str_to_bool('true') is True
        assert str_to_bool('false') is False
        assert str_to_bool('1') is True
        assert str_to_bool('0') is False
        assert str_to_bool('yes') is True
        assert str_to_bool('no') is False
        assert str_to_bool('t') is True
        assert str_to_bool('f') is False
        assert str_to_bool(True) is True
        assert str_to_bool(False) is False

    def test_str_to_bool_invalid(self):
        from flopyarcade.play import str_to_bool
        with pytest.raises(ValueError):
            str_to_bool('maybe')

    def test_has_arguments_no_custom(self):
        from flopyarcade.play import has_arguments
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--foo', default='bar')
        ns = parser.parse_args([])
        assert has_arguments(parser, ns) is False

    def test_has_arguments_custom(self):
        from flopyarcade.play import has_arguments
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--foo', default='bar')
        ns = parser.parse_args(['--foo', 'baz'])
        assert has_arguments(parser, ns) is True

    def test_main_default_args(self):
        """Test main() with default (no) CLI args - FloPyArcade is mocked."""
        from flopyarcade import play

        mock_game = MagicMock()
        mock_game.env.envhash = 'test_hash_123'
        mock_game.play.return_value = None

        with patch.object(play, 'FloPyArcade', return_value=mock_game) as MockArcade:
            with patch('sys.argv', ['play.py']):
                result = play.main()

        assert result == 'test_hash_123'
        MockArcade.assert_called_once()
        call_kwargs = MockArcade.call_args[1]
        assert call_kwargs['ENVTYPE'] == '3s-d'
        assert call_kwargs['NAGENTSTEPS'] == 200

    def test_main_with_custom_args(self):
        """Test main() with custom CLI args."""
        from flopyarcade import play

        mock_game = MagicMock()
        mock_game.env.envhash = 'custom_hash'
        mock_game.play.return_value = None

        with patch.object(play, 'FloPyArcade', return_value=mock_game) as MockArcade:
            with patch('sys.argv', ['play.py', '--envtype', '1s-d', '--ngames', '5', '--nagentsteps', '10']):
                result = play.main()

        assert result == 'custom_hash'
        call_kwargs = MockArcade.call_args[1]
        assert call_kwargs['ENVTYPE'] == '1s-d'
        assert call_kwargs['NAGENTSTEPS'] == 10

    def test_main_saveplot_true(self):
        """Test main() with --saveplot true."""
        from flopyarcade import play

        mock_game = MagicMock()
        mock_game.env.envhash = 'hash_sp'

        with patch.object(play, 'FloPyArcade', return_value=mock_game) as MockArcade:
            with patch('sys.argv', ['play.py', '--saveplot', 'true', '--manualcontrol', 'true']):
                result = play.main()

        call_kwargs = MockArcade.call_args[1]
        assert call_kwargs['flagSavePlot'] is True
        assert call_kwargs['flagManualControl'] is True


# ---------------------------------------------------------------------------
# play_againstai.py
# ---------------------------------------------------------------------------

# We need to mock ray.rllib.agents which doesn't exist in this environment.
# Using patch.dict as a patch context manager (standard unittest.mock pattern).


class TestPlayAgainstAI:
    """Cover play_againstai.py (179 stmts, 125 missed)."""

    @pytest.fixture(autouse=True)
    def mock_ray_rllib(self):
        """Mock the missing ray.rllib.agents module and clean argv before import."""
        import sys as _sys
        mock_agents = MagicMock()
        mock_agents.dqn.ApexTrainer = MagicMock()

        # Remove cached module so it re-imports with our mocks
        saved_mod = _sys.modules.pop('flopyarcade.play_againstai', None)
        saved_argv = _sys.argv

        with patch.dict('sys.modules', {
            'ray.rllib.agents': mock_agents,
            'ray.rllib.agents.dqn': mock_agents.dqn,
        }):
            with patch.object(_sys, 'argv', ['play_againstai.py']):
                yield

        _sys.argv = saved_argv
        # Restore cached module if it existed
        if saved_mod is not None:
            _sys.modules['flopyarcade.play_againstai'] = saved_mod

    def test_str_to_bool(self):
        from flopyarcade import play_againstai
        assert play_againstai.str_to_bool('true') is True
        assert play_againstai.str_to_bool('false') is False
        with pytest.raises(ValueError):
            play_againstai.str_to_bool('xyz')

    def test_env_creator(self):
        from flopyarcade import play_againstai

        mock_env = MagicMock()
        with patch.object(play_againstai, 'FloPyEnv', return_value=mock_env) as MockEnv:
            env = play_againstai.env_creator({})

        assert env is mock_env
        MockEnv.assert_called_once()
        call_kwargs = MockEnv.call_args[1]
        assert 'ENVTYPE' in call_kwargs['env_config']

    def test_config_globals(self):
        """Verify module-level config dicts exist and are well-formed."""
        from flopyarcade import play_againstai
        assert 'env' in play_againstai.config
        assert 'model' in play_againstai.config
        assert 'seed' in play_againstai.config
        assert 'training_iteration' in play_againstai.config_stopCriteria

    def test_config_model(self):
        from flopyarcade import play_againstai
        assert play_againstai.config_model['fcnet_hiddens'] == [512, 1024, 512]
        assert play_againstai.config_model['fcnet_activation'] == 'relu'

    def test_test_function_immediate_done(self):
        """Test the test() function where env is immediately done."""
        from flopyarcade import play_againstai

        mock_env = MagicMock()
        mock_env.done = True
        mock_env.observationsVectorNormalized = np.zeros(84)
        mock_env.MANUALCONTROLTIME = 0.01
        mock_env.RENDER = False

        mock_agent = MagicMock()

        with patch.object(play_againstai, 'FloPyAgent') as MockAgent:
            MockAgent.return_value = MagicMock()
            reward = play_againstai.test(mock_agent, mock_env)

        assert reward == 0.0


# ---------------------------------------------------------------------------
# archive_hdf5.py
# ---------------------------------------------------------------------------


class TestArchiveHDF5:
    """Cover archive_hdf5.py (58 stmts, 49 missed)."""

    def test_load_nonexistent(self):
        from flopyarcade.archive_hdf5 import NoveltyArchiveHDF5
        arc = NoveltyArchiveHDF5('/tmp/nonexistent_hdf5_test_xyz.h5')
        result = arc.load()
        assert result == {}

    def test_len_nonexistent(self):
        from flopyarcade.archive_hdf5 import NoveltyArchiveHDF5
        arc = NoveltyArchiveHDF5('/tmp/nonexistent_hdf5_test_xyz.h5')
        assert len(arc) == 0

    def test_keys_nonexistent(self):
        from flopyarcade.archive_hdf5 import NoveltyArchiveHDF5
        arc = NoveltyArchiveHDF5('/tmp/nonexistent_hdf5_test_xyz.h5')
        assert arc.keys() == []

    def test_save_and_load_roundtrip(self, tmp_path):
        from flopyarcade.archive_hdf5 import NoveltyArchiveHDF5

        path = str(tmp_path / 'test_archive.h5')
        arc = NoveltyArchiveHDF5(path)

        archive = {
            'agent1': {'novelty': 0.5, 'actionsUniqueID': 42, 'modelFile': 'models/a1.keras'},
            'agent2': {'novelty': 0.9, 'actionsUniqueID': 99, 'modelFile': 'models/a2.keras'},
        }
        arc.save(archive)

        loaded = arc.load()
        assert 'agent1' in loaded
        assert loaded['agent1']['novelty'] == pytest.approx(0.5)
        assert loaded['agent1']['actionsUniqueID'] == 42
        assert loaded['agent1']['modelFile'] == 'models/a1.keras'
        assert 'agent2' in loaded
        assert loaded['agent2']['novelty'] == pytest.approx(0.9)

    def test_len_and_keys_after_save(self, tmp_path):
        from flopyarcade.archive_hdf5 import NoveltyArchiveHDF5

        path = str(tmp_path / 'test_archive2.h5')
        arc = NoveltyArchiveHDF5(path)

        archive = {'a1': {'novelty': 1.0, 'actionsUniqueID': 1, 'modelFile': 'm1'}}
        arc.save(archive)

        assert len(arc) == 1
        assert arc.keys() == ['a1']

    def test_append_new_key(self, tmp_path):
        from flopyarcade.archive_hdf5 import NoveltyArchiveHDF5

        path = str(tmp_path / 'test_archive3.h5')
        arc = NoveltyArchiveHDF5(path)

        arc.save({'a1': {'novelty': 0.1, 'actionsUniqueID': 1, 'modelFile': 'm1'}})
        arc.save({'a2': {'novelty': 0.2, 'actionsUniqueID': 2, 'modelFile': 'm2'}})

        loaded = arc.load()
        assert len(loaded) == 2
        assert 'a1' in loaded and 'a2' in loaded


# ---------------------------------------------------------------------------
# benchmark.py
# ---------------------------------------------------------------------------


class _FakePolicy:
    """A picklable object with predict_on_batch to test _load_rllib_pickle."""
    def predict_on_batch(self, obs):
        return np.array([1.0, 2.0])


class TestBenchmark:
    """Cover benchmark.py remaining 43 lines."""

    def test_try_layer_get_direct(self):
        from flopyarcade.benchmark import _try_layer_get
        policy = {
            'fc_1/kernel': np.ones((4, 3)),
            'fc_1/bias': np.zeros(3),
        }
        k, b = _try_layer_get(policy, '', 'fc_1')
        assert k is not None
        assert b is not None
        assert k.shape == (4, 3)

    def test_try_layer_get_with_prefix(self):
        from flopyarcade.benchmark import _try_layer_get
        policy = {
            'default_policy/fc_1/kernel': np.ones((4, 3)),
            'default_policy/fc_1/bias': np.zeros(3),
        }
        k, b = _try_layer_get(policy, 'default_policy/', 'fc_1')
        assert k is not None
        assert b is not None

    def test_try_layer_get_fallback_search(self):
        from flopyarcade.benchmark import _try_layer_get
        policy = {
            'some_random_fc_1_hidden_kernel': np.ones((4, 3)),
            'some_random_fc_1_hidden_bias': np.zeros(3),
        }
        k, b = _try_layer_get(policy, '', 'fc_1')
        assert k is not None
        assert b is not None

    def test_try_layer_get_not_found(self):
        from flopyarcade.benchmark import _try_layer_get
        policy = {'unrelated_key': np.ones(3)}
        k, b = _try_layer_get(policy, '', 'fc_1')
        assert k is None
        assert b is None

    def test_load_rllib_pickle_policy_object(self):
        """Test _load_rllib_pickle with a picklable object that has predict_on_batch."""
        from flopyarcade.benchmark import _load_rllib_pickle

        fake_policy = _FakePolicy()

        with tempfile.NamedTemporaryFile(suffix='.p', delete=False) as f:
            pickle.dump(fake_policy, f)
            fpath = f.name

        try:
            result = _load_rllib_pickle(fpath)
            assert hasattr(result, 'predict_on_batch')
        finally:
            os.unlink(fpath)

    def test_load_rllib_pickle_dict_error(self):
        """Test _load_rllib_pickle with a dict that has no policy keys."""
        from flopyarcade.benchmark import _load_rllib_pickle

        with tempfile.NamedTemporaryFile(suffix='.p', delete=False) as f:
            pickle.dump({'unrelated': 1}, f)
            fpath = f.name

        try:
            with pytest.raises((ValueError, TypeError)):
                _load_rllib_pickle(fpath)
        finally:
            os.unlink(fpath)

    def test_load_npz_policy(self):
        """Test _load_npz_policy with a valid npz file."""
        from flopyarcade.benchmark import _load_npz_policy

        w1 = np.random.randn(84, 64)
        b1 = np.random.randn(64)
        w3 = np.random.randn(64, 3)
        b3 = np.random.randn(3)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fpath = f.name

        np.savez(fpath, fc_1_kernel=w1, fc_1_bias=b1, fc_out_kernel=w3, fc_out_bias=b3)

        try:
            # Mock the keras imports inside the function
            mock_model = MagicMock()
            with patch.dict('sys.modules', {
                'tensorflow': MagicMock(),
                'tensorflow.keras': MagicMock(),
                'tensorflow.keras.models': MagicMock(),
                'tensorflow.keras.layers': MagicMock(),
            }):
                # The function imports from tensorflow.keras.models and layers
                # We need to patch them where they're looked up
                import tensorflow.keras.models as tkm
                import tensorflow.keras.layers as tkl
                with patch.object(tkm, 'Sequential', return_value=mock_model):
                    with patch.object(tkl, 'Dense', return_value=MagicMock()):
                        with patch.object(tkl, 'Activation', return_value=MagicMock()):
                            result = _load_npz_policy(fpath)
        finally:
            os.unlink(fpath)

    def test_run_benchmark_mocked(self):
        """Test run_benchmark with fully mocked arcade."""
        from flopyarcade.benchmark import run_benchmark

        mock_arcade = MagicMock()
        mock_arcade.play.return_value = 42.0

        with patch('flopyarcade.benchmark.load_keras_agent', return_value=MagicMock()) as MockLoad:
            with patch('flopyarcade.FloPyArcade', return_value=mock_arcade) as MockArcade:
                with patch('flopyarcade.FloPyAgent') as MockAgent:
                    results = run_benchmark(
                        model_path='/fake/model.p',
                        envtype='1s-d',
                        n_games=3,
                        seed=1,
                        n_steps=10,
                        verbose=False,
                    )

        assert results['n_games'] == 3
        assert results['mean_reward'] == 42.0
        assert results['envtype'] == '1s-d'

    def test_run_benchmark_with_output(self, tmp_path):
        """Test run_benchmark saving output JSON."""
        import json
        from flopyarcade.benchmark import run_benchmark

        mock_arcade = MagicMock()
        mock_arcade.play.return_value = 10.0

        out_path = str(tmp_path / 'results.json')

        with patch('flopyarcade.benchmark.load_keras_agent', return_value=MagicMock()):
            with patch('flopyarcade.FloPyArcade', return_value=mock_arcade):
                with patch('flopyarcade.FloPyAgent') as MockAgent:
                    results = run_benchmark(
                        model_path='/fake/model.p',
                        envtype='1s-d',
                        n_games=2,
                        seed=1,
                        n_steps=5,
                        verbose=False,
                        output_path=out_path,
                    )

        assert os.path.exists(out_path)
        with open(out_path) as f:
            saved = json.load(f)
        assert 'per_game_rewards' in saved
        assert len(saved['per_game_rewards']) == 2


# ---------------------------------------------------------------------------
# mf6_prt_runner.py
# ---------------------------------------------------------------------------


class TestMF6PRTMixin:
    """Cover mf6_prt_runner.py (161 stmts, 99 missed)."""

    def _make_env(self, **overrides):
        """Create a minimal mock environment with MF6PRTMixin attributes."""
        from flopyarcade.mf6_prt_runner import MF6PRTMixin

        class FakeEnv(MF6PRTMixin):
            pass

        env = FakeEnv()
        env.MODELNAME = 'TestModel'
        env.nLay = 1
        env.nRow = 10
        env.nCol = 10
        env.dRow = 1.0
        env.dCol = 1.0
        env.zTop = 1.0
        env.strt = 1.0
        env.periodSteps = 10
        env.particleCoords = [5.0, 5.0, 0.5]
        env.modelpth = '/tmp'
        env.trajectories = {'x': [], 'y': [], 'z': []}
        env.wellCellLayer = 1
        env.wellCellRow = 5
        env.wellCellColumn = 5
        env.wellQ = -100.0
        env.nHelperWells = 0
        env.chd_west = 5.0
        env.chd_east = 0.0
        env.rch_rate = None
        env.headSpecWest = None
        env.headSpecEast = None
        env.hk = None
        # _filterByTime is expected to exist from the parent env class
        env._filterByTime = lambda df: np.ones(len(df), dtype=bool)

        for k, v in overrides.items():
            setattr(env, k, v)
        return env

    def test_cell_from_coords(self):
        env = self._make_env()
        row, col = env._mf6prt_cell_from_coords(5.0, 5.0)
        assert 0 <= row < 10
        assert 0 <= col < 10

    def test_cell_from_coords_clamped(self):
        env = self._make_env()
        row, col = env._mf6prt_cell_from_coords(100.0, 100.0)
        assert row == 0
        assert col == 9

    def test_get_wells_single(self):
        env = self._make_env()
        wells = env._mf6prt_get_wells()
        assert len(wells) == 1
        # Structure is [(layer, row, col), wellQ]
        assert wells[0][0] == (0, 4, 4)
        assert wells[0][1] == -100.0

    def test_get_wells_with_helpers(self):
        env = self._make_env(nHelperWells=2)
        env.helperWells = {
            'l1': 1, 'r1': 3, 'c1': 3, 'wellQ1': -50.0,
            'l2': 1, 'r2': 6, 'c2': 6, 'wellQ2': 30.0,
        }
        wells = env._mf6prt_get_wells()
        assert len(wells) == 3

    def test_get_hydraulics_defaults(self):
        env = self._make_env(hk=None)
        hk, vka, ss, sy, chd_w, chd_e, rch = env._mf6prt_get_hydraulics()
        assert hk == 10.0
        assert vka == 10.0
        assert ss == 1e-5
        assert sy == 0.15

    def test_get_hydraulics_custom(self):
        env = self._make_env(hk=5.0, chd_west=8.0, chd_east=2.0, rch_rate=0.001)
        hk, vka, ss, sy, chd_w, chd_e, rch = env._mf6prt_get_hydraulics()
        assert hk == 5.0
        assert chd_w == 8.0
        assert chd_e == 2.0
        assert rch == 0.001

    def test_parse_track_csv(self, tmp_path):
        env = self._make_env()
        csv_path = str(tmp_path / 'test_track.csv')
        with open(csv_path, 'w') as f:
            f.write('x,y,z,t\n')
            f.write('1.0,2.0,0.5,0.0\n')
            f.write('1.5,2.5,0.6,1.0\n')
            f.write('2.0,3.0,0.7,2.0\n')

        df = env._mf6prt_parse_track_csv(csv_path)
        assert len(df) == 3
        assert list(df.columns) == ['x', 'y', 'z', 'time']
        assert df['x'].iloc[0] == 1.0
        assert df['time'].iloc[2] == 2.0

    def test_parse_track_csv_empty(self, tmp_path):
        env = self._make_env()
        csv_path = str(tmp_path / 'empty_track.csv')
        with open(csv_path, 'w') as f:
            f.write('x,y,z,t\n')

        with pytest.raises(RuntimeError, match='empty'):
            env._mf6prt_parse_track_csv(csv_path)

    def test_parse_track_csv_bad_rows(self, tmp_path):
        env = self._make_env()
        csv_path = str(tmp_path / 'bad_track.csv')
        with open(csv_path, 'w') as f:
            f.write('x,y,z,t\n')
            f.write('abc,def,ghi,jkl\n')
            f.write('1.0,2.0,0.5,0.0\n')

        df = env._mf6prt_parse_track_csv(csv_path)
        assert len(df) == 1  # only the valid row

    def test_step_run_simulation_mf6prt(self):
        """Test _step_run_simulation_mf6prt with mocked build_and_run."""
        env = self._make_env()
        env.modelpth = tempfile.mkdtemp()

        mock_df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0],
            'z': [0.1, 0.2, 0.3],
            'time': [0.0, 1.0, 2.0],
        })

        with patch.object(env, '_mf6prt_build_and_run', return_value=mock_df) as MockBuild:
            env._step_run_simulation_mf6prt()

        MockBuild.assert_called_once()
        assert len(env.trajectories['x']) == 1
        assert len(env.trajectories['y']) == 1
        assert len(env.trajectories['z']) == 1
        assert env.particleCoords == [3.0, 6.0, 0.3]

    def test_step_run_simulation_with_filter(self):
        """Test with _filterByTime returning a boolean mask."""
        env = self._make_env()
        env.modelpth = tempfile.mkdtemp()

        mock_df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0],
            'z': [0.1, 0.2, 0.3],
            'time': [0.0, 1.0, 2.0],
        })

        mask = np.array([True, True, False])
        env._filterByTime = lambda df: mask

        with patch.object(env, '_mf6prt_build_and_run', return_value=mock_df):
            env._step_run_simulation_mf6prt()

        # Last valid position should be index 1
        assert env.particleCoords == [2.0, 5.0, 0.2]

