"""Coverage tests: benchmark, actions, observations, env_base, render, arcade."""
import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call
from collections import defaultdict
import pickle


# ===================== BENCHMARK =====================

class TestTryLayerGet:
    def test_direct_match(self):
        from flopyarcade.benchmark import _try_layer_get
        policy = {'fc_1/kernel': np.ones((4,8)), 'fc_1/bias': np.zeros(8)}
        k, b = _try_layer_get(policy, '', 'fc_1')
        assert k is not None and b is not None

    def test_with_prefix(self):
        from flopyarcade.benchmark import _try_layer_get
        policy = {'policy/fc_out/kernel': np.ones((3,5)), 'policy/fc_out/bias': np.zeros(5)}
        k, b = _try_layer_get(policy, 'policy/', 'fc_out')
        assert k is not None and b is not None

    def test_no_match(self):
        from flopyarcade.benchmark import _try_layer_get
        policy = {'other/kernel': np.ones((2,2)), 'other/bias': np.zeros(2)}
        k, b = _try_layer_get(policy, '', 'fc_1')
        assert k is None and b is None

    def test_fallback_search(self):
        from flopyarcade.benchmark import _try_layer_get
        policy = {'weird_fc_1_kernel': np.ones((4,3)), 'weird_fc_1_bias': np.zeros(3)}
        k, b = _try_layer_get(policy, '', 'fc_1')
        assert k is not None


class TestLoadKerasAgent:
    def test_pickle_dir(self, tmp_path):
        from flopyarcade.benchmark import load_keras_agent
        d = tmp_path / 'chkpt'
        d.mkdir()
        (d / '.tune_metadata').write_bytes(b'x')
        with patch('flopyarcade.benchmark._load_rllib_checkpoint', return_value='mock') as m:
            r = load_keras_agent(str(d))
        m.assert_called_once()

    def test_keras_in_dir(self, tmp_path):
        from flopyarcade.benchmark import load_keras_agent
        d = tmp_path / 'mymodel'
        d.mkdir()
        (d / 'model.keras').write_bytes(b'x')
        with patch('tensorflow.keras.models.load_model', return_value='m') as lm:
            r = load_keras_agent(str(d))
        lm.assert_called()

    def test_dir_no_model(self, tmp_path):
        from flopyarcade.benchmark import load_keras_agent
        d = tmp_path / 'empty'
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            load_keras_agent(str(d))

    def test_pickle_file(self):
        from flopyarcade.benchmark import load_keras_agent
        with patch('flopyarcade.benchmark._load_rllib_pickle', return_value='obj') as m:
            r = load_keras_agent('x.p')
        m.assert_called_once_with('x.p')

    def test_keras_file(self):
        from flopyarcade.benchmark import load_keras_agent
        with patch('tensorflow.keras.models.load_model', return_value='m') as lm:
            r = load_keras_agent('model.keras')
        lm.assert_called_with('model.keras', compile=False)


class TestLoadRllibPickle:


    def test_invalid_type(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_pickle
        import pickle
        p = str(tmp_path / 'x.p')
        with open(p, 'wb') as f:
            pickle.dump(42, f)
        with pytest.raises(TypeError):
            _load_rllib_pickle(p)


class TestLoadNpzPolicy:
    def test_valid_npz(self, tmp_path):
        from flopyarcade.benchmark import _load_npz_policy
        p = str(tmp_path / 'w.npz')
        np.savez(p, layer0_kernel=np.ones((4,8)), layer0_bias=np.zeros(8),
                  layer1_kernel=np.ones((8,3)), layer1_bias=np.zeros(3))
        r = _load_npz_policy(p)
        assert r is not None

    def test_invalid_npz(self, tmp_path):
        from flopyarcade.benchmark import _load_npz_policy
        p = str(tmp_path / 'bad.npz')
        np.savez(p, foo=np.ones(3))
        with pytest.raises(ValueError):
            _load_npz_policy(p)


class TestRunBenchmark:
    def test_basic(self, tmp_path):
        from flopyarcade.benchmark import run_benchmark
        p = str(tmp_path / 'm.keras')
        out = str(tmp_path / 'out.json')
        with patch('flopyarcade.benchmark.load_keras_agent', return_value='mockmodel'):
            with patch('flopyarcade.FloPyArcade') as FA:
                FA.return_value.play.return_value = 100.0
                with patch('flopyarcade.FloPyAgent.suppressTensorFlowWarnings'):
                    r = run_benchmark(p, envtype='1s-d', n_games=3, verbose=True, output_path=out)
        assert r['n_games'] == 3
        assert os.path.exists(out)

    def test_quiet(self, tmp_path):
        from flopyarcade.benchmark import run_benchmark
        p = str(tmp_path / 'm.keras')
        with patch('flopyarcade.benchmark.load_keras_agent', return_value='m'):
            with patch('flopyarcade.FloPyArcade') as FA:
                FA.return_value.play.return_value = 50.0
                r = run_benchmark(p, envtype='1s-d', n_games=2, verbose=False)
        assert r['mean_reward'] == 50.0


class TestBenchmarkMain:
    def test_main(self, tmp_path, capsys):
        from flopyarcade.benchmark import main
        p = str(tmp_path / 'm.keras')
        args = ['bench', '--model', p, '--envtype', '1s-d', '--games', '2', '--quiet']
        with patch('sys.argv', args):
            with patch('flopyarcade.benchmark.load_keras_agent', return_value='m'):
                with patch('flopyarcade.FloPyArcade') as FA:
                    FA.return_value.play.return_value = 10.0
                    r = main()
        assert r['n_games'] == 2

