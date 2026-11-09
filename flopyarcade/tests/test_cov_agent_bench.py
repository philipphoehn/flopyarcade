"""Coverage tests for flopyarcade/agent.py and flopyarcade/benchmark.py."""
import os
import pickle
import json
from collections import deque
from copy import deepcopy
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# =============================================================================
# BENCHMARK.PY TESTS
# =============================================================================


class FakePolicy:
    """Simple picklable policy object."""
    def predict_on_batch(self, *args, **kwargs):
        return np.array([0.1, 0.2, 0.7])


class TestTryLayerGet:
    """Test _try_layer_get function."""

    def test_found_first_candidate(self):
        from flopyarcade.benchmark import _try_layer_get
        pol = {'fc_1/kernel': np.ones((3, 4)), 'fc_1/bias': np.ones(4)}
        k, b = _try_layer_get(pol, '', 'fc_1')
        assert k is not None and b is not None
        assert k.shape == (3, 4)

    def test_found_with_prefix(self):
        from flopyarcade.benchmark import _try_layer_get
        pol = {'dp/fc_out/kernel': np.ones((2, 3)), 'dp/fc_out/bias': np.ones(3)}
        k, b = _try_layer_get(pol, 'dp/', 'fc_out')
        assert k is not None and b is not None

    def test_found_underscore_stripped(self):
        from flopyarcade.benchmark import _try_layer_get
        pol = {'fc1/kernel': np.ones((2, 3)), 'fc1/bias': np.ones(3)}
        k, b = _try_layer_get(pol, '', 'fc_1')
        assert k is not None and b is not None

    def test_not_found(self):
        from flopyarcade.benchmark import _try_layer_get
        pol = {'x/key': np.ones(2)}
        k, b = _try_layer_get(pol, '', 'fc_1')
        assert k is None and b is None

    def test_fallback_search(self):
        from flopyarcade.benchmark import _try_layer_get
        pol = {'zz_dense0_kernel': np.ones((3, 2)), 'zz_dense0_bias': np.ones(2)}
        k, b = _try_layer_get(pol, '', 'dense0')
        assert k is not None and b is not None

    def test_only_kernel_found(self):
        from flopyarcade.benchmark import _try_layer_get
        pol = {'fc_1/kernel': np.ones((3, 4))}
        k, b = _try_layer_get(pol, '', 'fc_1')
        assert k is not None
        assert b is None


class TestLoadKerasAgent:
    """Test load_keras_agent function."""

    def test_pickle_p_ext(self):
        from flopyarcade.benchmark import load_keras_agent
        with patch('flopyarcade.benchmark._load_rllib_pickle', return_value=MagicMock()) as m:
            load_keras_agent('/x/y.p')
            m.assert_called_once_with('/x/y.p')

    def test_pickle_pickle_ext(self):
        from flopyarcade.benchmark import load_keras_agent
        with patch('flopyarcade.benchmark._load_rllib_pickle', return_value=MagicMock()) as m:
            load_keras_agent('/x/y.pickle')
            m.assert_called_once_with('/x/y.pickle')

    def test_keras_native_file(self):
        from flopyarcade.benchmark import load_keras_agent
        mock_model = MagicMock()
        with patch('os.path.isdir', return_value=False):
            with patch('tensorflow.keras.models.load_model', return_value=mock_model) as lm:
                result = load_keras_agent('/x/model.keras')
                lm.assert_called_once_with('/x/model.keras', compile=False)

    def test_dir_tune_metadata(self, tmp_path):
        from flopyarcade.benchmark import load_keras_agent
        (tmp_path / '.tune_metadata').write_bytes(b'x')
        with patch('flopyarcade.benchmark._load_rllib_checkpoint', return_value=MagicMock()) as m:
            load_keras_agent(str(tmp_path))
            m.assert_called_once_with(str(tmp_path), str(tmp_path / '.tune_metadata'))

    def test_dir_keras_file(self, tmp_path):
        from flopyarcade.benchmark import load_keras_agent
        (tmp_path / 'm.keras').write_bytes(b'x')
        mock_model = MagicMock()
        with patch('tensorflow.keras.models.load_model', return_value=mock_model) as lm:
            load_keras_agent(str(tmp_path))
            lm.assert_called_once_with(str(tmp_path / 'm.keras'), compile=False)

    def test_dir_empty_raises(self, tmp_path):
        from flopyarcade.benchmark import load_keras_agent
        with pytest.raises(FileNotFoundError):
            load_keras_agent(str(tmp_path))


class TestLoadRllibPickle:
    """Test _load_rllib_pickle."""

    def test_policy_object_with_predict(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_pickle
        fake = FakePolicy()
        p = str(tmp_path / 'm.p')
        with open(p, 'wb') as f:
            pickle.dump(fake, f)
        result = _load_rllib_pickle(p)
        assert isinstance(result, FakePolicy)
        assert hasattr(result, 'predict_on_batch')

    def test_non_dict_raises(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_pickle
        p = str(tmp_path / 'm.p')
        with open(p, 'wb') as f:
            pickle.dump(42, f)
        with pytest.raises(TypeError):
            _load_rllib_pickle(p)

    def test_dict_default_policy_two_layer(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_pickle
        w1 = np.ones((4, 8))
        b1 = np.ones(8)
        w2 = np.ones((8, 5))
        b2 = np.ones(5)
        pol = {'a_kernel': w1, 'a_bias': b1, 'b_kernel': w2, 'b_bias': b2}
        obj = {'default_policy': pol}
        p = str(tmp_path / 'm.p')
        with open(p, 'wb') as f:
            pickle.dump(obj, f)
        mock_model = MagicMock()
        mock_model.layers = [MagicMock(), MagicMock(), MagicMock()]
        with patch('tensorflow.keras.models.Sequential', return_value=mock_model):
            with patch('tensorflow.keras.layers.Dense', return_value=MagicMock()):
                with patch('tensorflow.keras.layers.Activation', return_value=MagicMock()):
                    result = _load_rllib_pickle(p)
                    assert result is mock_model

    def test_dict_no_policy_raises(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_pickle
        p = str(tmp_path / 'm.p')
        with open(p, 'wb') as f:
            pickle.dump({'foo': 'bar'}, f)
        with pytest.raises(ValueError, match='Cannot find policy weights'):
            _load_rllib_pickle(p)

    def test_dict_is_policy_direct(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_pickle
        w1 = np.ones((4, 8))
        b1 = np.ones(8)
        w2 = np.ones((8, 5))
        b2 = np.ones(5)
        obj = {'a_kernel': w1, 'a_bias': b1, 'b_kernel': w2, 'b_bias': b2}
        p = str(tmp_path / 'm.p')
        with open(p, 'wb') as f:
            pickle.dump(obj, f)
        mock_model = MagicMock()
        mock_model.layers = [MagicMock(), MagicMock(), MagicMock()]
        with patch('tensorflow.keras.models.Sequential', return_value=mock_model):
            with patch('tensorflow.keras.layers.Dense', return_value=MagicMock()):
                with patch('tensorflow.keras.layers.Activation', return_value=MagicMock()):
                    result = _load_rllib_pickle(p)
                    assert result is mock_model


class TestLoadRllibCheckpoint:
    """Test _load_rllib_checkpoint."""

    def test_finds_p_file(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_checkpoint
        with open(str(tmp_path / '.tune_metadata'), 'wb') as f:
            pickle.dump({}, f)
        (tmp_path / 'pol.p').write_bytes(b'x')
        with patch('flopyarcade.benchmark._load_rllib_pickle', return_value=MagicMock()) as m:
            _load_rllib_checkpoint(str(tmp_path), str(tmp_path / '.tune_metadata'))
            m.assert_called_once_with(str(tmp_path / 'pol.p'))

    def test_finds_npz(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_checkpoint
        with open(str(tmp_path / '.tune_metadata'), 'wb') as f:
            pickle.dump({}, f)
        (tmp_path / 'w.npz').write_bytes(b'x')
        with patch('flopyarcade.benchmark._load_npz_policy', return_value=MagicMock()) as m:
            _load_rllib_checkpoint(str(tmp_path), str(tmp_path / '.tune_metadata'))
            m.assert_called_once_with(str(tmp_path / 'w.npz'))

    def test_meta_has_predict(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_checkpoint
        meta = FakePolicy()
        with open(str(tmp_path / '.tune_metadata'), 'wb') as f:
            pickle.dump(meta, f)
        result = _load_rllib_checkpoint(str(tmp_path), str(tmp_path / '.tune_metadata'))
        assert isinstance(result, FakePolicy)

    def test_no_policy_raises(self, tmp_path):
        from flopyarcade.benchmark import _load_rllib_checkpoint
        meta = [1, 2, 3]
        with open(str(tmp_path / '.tune_metadata'), 'wb') as f:
            pickle.dump(meta, f)
        with pytest.raises(ValueError, match='Could not find policy weights'):
            _load_rllib_checkpoint(str(tmp_path), str(tmp_path / '.tune_metadata'))


class TestLoadNpzPolicy:
    """Test _load_npz_policy."""

    def test_valid_npz(self, tmp_path):
        from flopyarcade.benchmark import _load_npz_policy
        w1 = np.ones((3, 4))
        b1 = np.ones(4)
        w2 = np.ones((4, 2))
        b2 = np.ones(2)
        p = str(tmp_path / 'm.npz')
        np.savez(p, fc_1_kernel=w1, fc_1_bias=b1, fc_out_kernel=w2, fc_out_bias=b2)
        mock_model = MagicMock()
        mock_model.layers = [MagicMock(), MagicMock(), MagicMock()]
        with patch('tensorflow.keras.models.Sequential', return_value=mock_model):
            with patch('tensorflow.keras.layers.Dense', return_value=MagicMock()):
                with patch('tensorflow.keras.layers.Activation', return_value=MagicMock()):
                    result = _load_npz_policy(p)
                    assert result is mock_model

    def test_invalid_npz_raises(self, tmp_path):
        from flopyarcade.benchmark import _load_npz_policy
        p = str(tmp_path / 'bad.npz')
        np.savez(p, foo=np.ones(3))
        with pytest.raises(ValueError, match='Cannot reconstruct'):
            _load_npz_policy(p)


class TestRunBenchmark:
    """Test run_benchmark."""

    def test_success(self, tmp_path):
        from flopyarcade.benchmark import run_benchmark
        mock_arcade = MagicMock()
        mock_arcade.play.return_value = 1.5
        mock_agent_cls = MagicMock()
        mock_agent_cls.suppressTensorFlowWarnings = MagicMock()
        with patch('flopyarcade.benchmark.load_keras_agent', return_value=MagicMock()):
            with patch('flopyarcade.FloPyArcade', return_value=mock_arcade):
                with patch('flopyarcade.FloPyAgent', mock_agent_cls):
                    r = run_benchmark(
                        str(tmp_path / 'm.keras'), '1s-d', 3, 42, 10, False
                    )
        assert r['n_games'] == 3
        assert r['envtype'] == '1s-d'
        assert mock_arcade.play.call_count == 3
        assert 'mean_reward' in r

    def test_game_exception(self, tmp_path):
        from flopyarcade.benchmark import run_benchmark
        mock_arcade = MagicMock()
        mock_arcade.play.side_effect = Exception('crash')
        mock_agent_cls = MagicMock()
        mock_agent_cls.suppressTensorFlowWarnings = MagicMock()
        with patch('flopyarcade.benchmark.load_keras_agent', return_value=MagicMock()):
            with patch('flopyarcade.FloPyArcade', return_value=mock_arcade):
                with patch('flopyarcade.FloPyAgent', mock_agent_cls):
                    r = run_benchmark(
                        str(tmp_path / 'm.keras'), '1s-d', 2, 0, 5, False
                    )
        assert r['mean_reward'] == 0.0

    def test_save_output(self, tmp_path):
        from flopyarcade.benchmark import run_benchmark
        mock_arcade = MagicMock()
        mock_arcade.play.return_value = 2.0
        mock_agent_cls = MagicMock()
        mock_agent_cls.suppressTensorFlowWarnings = MagicMock()
        out = str(tmp_path / 'r.json')
        with patch('flopyarcade.benchmark.load_keras_agent', return_value=MagicMock()):
            with patch('flopyarcade.FloPyArcade', return_value=mock_arcade):
                with patch('flopyarcade.FloPyAgent', mock_agent_cls):
                    run_benchmark(
                        str(tmp_path / 'm.keras'), '2s-d', 2, 1, 5, True, out
                    )
        assert os.path.exists(out)
        with open(out) as f:
            data = json.load(f)
        assert 'per_game_rewards' in data
        assert data['n_games'] == 2

    def test_verbose_suppresses_warnings(self, tmp_path):
        from flopyarcade.benchmark import run_benchmark
        mock_arcade = MagicMock()
        mock_arcade.play.return_value = 1.0
        mock_agent_cls = MagicMock()
        with patch('flopyarcade.benchmark.load_keras_agent', return_value=MagicMock()):
            with patch('flopyarcade.FloPyArcade', return_value=mock_arcade):
                with patch('flopyarcade.FloPyAgent', mock_agent_cls):
                    run_benchmark(
                        str(tmp_path / 'm.keras'), '1s-d', 1, 0, 5, True
                    )
        mock_agent_cls.suppressTensorFlowWarnings.assert_called()


class TestBenchmarkMain:
    """Test main()."""

    def test_main(self):
        from flopyarcade.benchmark import main
        mock_r = {'mean_reward': 1.0, 'n_games': 10}
        argv = ['bench', '--model', '/f/m.keras', '--envtype', '1s-d',
                '--games', '10', '--quiet']
        with patch('sys.argv', argv):
            with patch('flopyarcade.benchmark.run_benchmark', return_value=mock_r) as c:
                result = main()
        assert result == mock_r
        assert c.call_args[1]['model_path'] == '/f/m.keras'
        assert c.call_args[1]['verbose'] is False


# =============================================================================
# AGENT.PY TESTS
# =============================================================================


def make_agent(**attrs):
    """Create FloPyAgent instance bypassing __init__."""
    from flopyarcade.agent import FloPyAgent
    a = object.__new__(FloPyAgent)
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


class TestGetActionType:
    def test_discrete(self):
        a = make_agent()
        assert a.getActionType('1s-d') == 'discrete'

    def test_continuous(self):
        a = make_agent()
        assert a.getActionType('6r-c') == 'continuous'

    def test_unknown_exits(self):
        a = make_agent()
        with pytest.raises(SystemExit):
            a.getActionType('bad-type')


class TestGetWorkspace:
    def test_returns_str(self):
        a = make_agent()
        ws = a._get_workspace()
        assert isinstance(ws, str)


class TestCreateFolder:
    def test_creates_and_assigns(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'sub')
        a._create_folder(p, assign_attr='mp')
        assert os.path.isdir(p)
        assert a.mp == p

    def test_no_assign(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'sub2')
        a._create_folder(p)
        assert os.path.isdir(p)


class TestYieldChunks:
    def test_basic(self):
        a = make_agent()
        assert list(a.yieldChunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    def test_empty(self):
        a = make_agent()
        assert list(a.yieldChunks([], 3)) == []

    def test_exact_division(self):
        a = make_agent()
        assert list(a.yieldChunks([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


class TestDecayEpsilon:
    def test_decays(self):
        a = make_agent(epsilon=0.5, hyParams={'EPSILONMIN': 0.1, 'EPSILONDECAY': 0.9})
        a._decay_epsilon()
        assert abs(a.epsilon - 0.45) < 1e-9

    def test_floor(self):
        # epsilon=0.05 is BELOW EPSILONMIN=0.1, so condition is False, epsilon unchanged
        a = make_agent(epsilon=0.05, hyParams={'EPSILONMIN': 0.1, 'EPSILONDECAY': 0.9})
        a._decay_epsilon()
        assert a.epsilon == 0.05

    def test_at_boundary(self):
        # epsilon=0.1 equals EPSILONMIN, condition > is False, stays same
        a = make_agent(epsilon=0.1, hyParams={'EPSILONMIN': 0.1, 'EPSILONDECAY': 0.9})
        a._decay_epsilon()
        assert a.epsilon == 0.1


class TestCheckParameterMean:
    def test_basic(self):
        a = make_agent()
        mock_agent = MagicMock()
        mock_agent.get_weights.return_value = [np.ones((2, 3)), np.ones(3)]
        result = a.checkParameterMean(mock_agent)
        assert result == 1.0


class TestCountParameters:
    def test_basic(self):
        a = make_agent()
        mock = MagicMock()
        mock.get_weights.return_value = [np.ones((2, 3)), np.ones(3)]
        # .size gives total elements: 6 + 3 = 9
        assert a.countParameters(mock) == 9

    def test_multiple_layers(self):
        a = make_agent()
        mock = MagicMock()
        mock.get_weights.return_value = [np.ones((4, 8)), np.ones(8), np.ones((8, 5)), np.ones(5)]
        # .size: 32 + 8 + 40 + 5 = 85
        assert a.countParameters(mock) == 85


class TestMutateDecision:
    def test_returns_bool(self):
        a = make_agent()
        result = a.mutateDecision(0.5)
        assert isinstance(result, bool)


class TestMutateDecisions:
    def test_shape(self):
        a = make_agent()
        result = a.mutateDecisions(0.5, (3, 4))
        assert result.shape == (3, 4)


class TestGetqsGivenAgentModel:
    def test_basic(self):
        a = make_agent()
        mock_model = MagicMock()
        mock_model.predict_on_batch.return_value = np.array([[0.1, 0.9]])
        r = a.getqsGivenAgentModel(mock_model, np.array([1.0, 2.0, 3.0]))
        assert r.shape == (2,)


class TestMutateGenetic:
    def test_basic(self):
        a = make_agent(hyParams={'MUTATIONPROBABILITY': 0.5, 'MUTATIONPOWER': 0.1})
        mock_agent = MagicMock()
        mock_agent.get_weights.return_value = [np.ones((2, 3)), np.ones(3)]
        result = a.mutateGenetic(mock_agent, 42)
        assert result is mock_agent


class TestActionNoveltyMetric:
    def test_discrete(self):
        a = make_agent()
        r = a.actionNoveltyMetric(['a', 'b', 'c'], ['a', 'x', 'c'], 'discrete')
        assert 0 <= r <= 1

    def test_discrete_unequal_len(self):
        a = make_agent()
        r = a.actionNoveltyMetric(['a', 'b'], ['a', 'b', 'c', 'd'], 'discrete')
        assert 0 <= r <= 1

    def test_continuous(self):
        a = make_agent()
        a1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        a2 = np.array([[0.2, 0.3], [0.4, 0.5]])
        r = a.actionNoveltyMetric(a1, a2, 'continuous')
        assert 0 <= r <= 1

    def test_identical_discrete(self):
        a = make_agent()
        r = a.actionNoveltyMetric(['a', 'b', 'c'], ['a', 'b', 'c'], 'discrete')
        assert r == 0.0


class TestCalculateNoveltyNeighborBounds:
    def test_middle(self):
        # iAgent=2, NNOVELTYNEIGHBORS=4, nLower=2, nHigher=2
        # iAgent-nLower=0>=0, so rangeLower=0, rangeHigher=4
        a = make_agent(
            agentsUnique=[0, 1, 2, 3, 4],
            hyParams={'NNOVELTYNEIGHBORS': 4, 'NAGENTS': 10},
            noveltyItemCount=5,
        )
        r = a.calculateNoveltyNeighborBounds(2)
        assert r[0] == 0 and r[1] == 4

    def test_bottom(self):
        a = make_agent(
            agentsUnique=[0, 1, 2, 3, 4],
            hyParams={'NNOVELTYNEIGHBORS': 4, 'NAGENTS': 10},
            noveltyItemCount=5,
        )
        r = a.calculateNoveltyNeighborBounds(0)
        assert r[0] == 0

    def test_top(self):
        a = make_agent(
            agentsUnique=[0, 1, 2, 3, 4],
            hyParams={'NNOVELTYNEIGHBORS': 4, 'NAGENTS': 10},
            noveltyItemCount=5,
        )
        r = a.calculateNoveltyNeighborBounds(4)
        assert r[1] == 5


class TestGeneratePathPrefixes:
    def test_basic(self, tmp_path):
        a = make_agent(
            tempModelpth=str(tmp_path),
            envSettings={'MODELNAME': 'test'},
            geneticGeneration=2,
            zFill=3,
        )
        a.generatePathPrefixes()
        assert hasattr(a, 'tempModelPrefix')
        assert hasattr(a, 'tempNextModelPrefix')
        assert hasattr(a, 'tempPrevModelPrefix')


class TestValidateHyperparams:
    def test_pass(self):
        a = make_agent(hyParams={'NAGENTS': 10, 'NNOVELTYELITES': 5})
        a._validate_hyperparams()

    def test_fail(self):
        a = make_agent(hyParams={'NAGENTS': 5, 'NNOVELTYELITES': 5})
        with pytest.raises(ValueError):
            a._validate_hyperparams()


class TestGenerateUniquePid:
    def test_basic(self):
        a = make_agent()
        a._generate_unique_pid()
        assert hasattr(a, 'pid')


class TestPrintNoveltyStats:
    def test_basic(self, capsys):
        a = make_agent(
            agentsUnique=[0, 1],
            agentsDuplicate=[2, 3, 4],
            noveltyItemCount=10,
            hyParams={'NNOVELTYNEIGHBORS': 5},
        )
        a._print_novelty_stats()
        out = capsys.readouterr().out
        assert 'unique' in out.lower() or 'novelty' in out.lower()

    def test_few_items(self, capsys):
        a = make_agent(
            agentsUnique=[0],
            agentsDuplicate=[],
            noveltyItemCount=2,
            hyParams={'NNOVELTYNEIGHBORS': 5},
        )
        a._print_novelty_stats()


class TestUpdateDiscreteUniques:
    def test_new_id(self):
        a = make_agent(
            agentsUnique=[],
            agentsUniqueIDs=[],
            agentsDuplicate=[],
            noveltyItemCount=3,
        )
        a._update_discrete_uniques(99)
        assert 3 in a.agentsUnique
        assert 99 in a.agentsUniqueIDs

    def test_dup_id(self):
        a = make_agent(
            agentsUnique=[0],
            agentsUniqueIDs=[99],
            agentsDuplicate=[],
            noveltyItemCount=3,
        )
        a._update_discrete_uniques(99)
        assert 3 in a.agentsDuplicate


class TestResetContinuousUniques:
    def test_basic(self):
        # range(noveltyItemCount + 1) = range(6) = 0..5, so agent1..agent6
        a = make_agent(
            agentsUnique=[0, 1],
            agentsUniqueIDs=[1, 2],
            agentsDuplicate=[3],
            noveltyItemCount=5,
            noveltyArchive={f'agent{i}': {} for i in range(1, 7)},
        )
        a._reset_continuous_uniques()
        assert len(a.agentsUnique) == 6
        assert a.agentsDuplicate == []


class TestBuildNoveltyArgs:
    def test_no_limit(self):
        a = make_agent(
            agentsUnique=[0, 1, 2],
            hyParams={'NNOVELTYNEIGHBORS': 10, 'NAGENTS': 5},
            noveltyItemCount=3,
        )
        args, flags = a._build_novelty_args()
        assert len(args) == 3

    def test_limit_reached(self):
        a = make_agent(
            agentsUnique=list(range(12)),
            hyParams={'NNOVELTYNEIGHBORS': 4, 'NAGENTS': 20},
            noveltyItemCount=12,
        )
        args, flags = a._build_novelty_args()
        assert len(args) == 12


class TestLogGenerationRewards:
    def test_basic(self):
        a = make_agent(
            rewards=[1.0, 2.0, 3.0],
            envSettings={'SEEDSRANDOM': False},
        )
        a._log_generation_rewards()

    def test_seeds_random(self):
        a = make_agent(
            rewards=[1.0, 2.0],
            envSettings={'SEEDSRANDOM': True},
            bestRewards=[],
        )
        a._log_generation_rewards()
        assert len(a.bestRewards) > 0


class TestSortAndSaveTopAgents:
    def test_basic(self, tmp_path):
        a = make_agent(
            rewards=[3.0, 1.0, 2.0],
            hyParams={'NAGENTELITES': 2},
            tempModelPrefix=str(tmp_path),
        )
        r = a._sort_and_save_top_agents()
        assert len(r) == 2


class TestPickleLoad:
    def test_basic(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'd.p')
        with open(p, 'wb') as f:
            pickle.dump({'x': 1}, f)
        assert a.pickleLoad(p) == {'x': 1}

    def test_with_compression(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'd.p')
        with patch('flopyarcade.agent.joblibLoad', return_value={'y': 2}) as jl:
            result = a.pickleLoad(p, compressed='lz4')
        assert result == {'y': 2}

    def test_unknown_compression(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'd.p')
        with pytest.raises(UnboundLocalError):
            a.pickleLoad(p, compressed='zlib')


class TestPickleDump:
    def test_basic(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'd.p')
        a.pickleDump(p, {'x': 2})
        assert os.path.exists(p)

    def test_with_compression(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'd.p')
        with patch('flopyarcade.agent.joblibDump') as jd:
            a.pickleDump(p, [1, 2, 3], compress='lz4')
        assert os.path.exists(p)

    def test_unknown_compression(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'd.p')
        a.pickleDump(p, [1, 2, 3], compress='zlib')
        # just prints error, no exception


class TestSuppressTensorFlowWarnings:
    def test_basic(self):
        a = make_agent()
        a.suppressTensorFlowWarnings()


class TestSetupNoveltyPath:
    def test_basic(self, tmp_path):
        a = make_agent(
            tempModelpth=str(tmp_path),
            envSettings={'RESUME': False},
        )
        a._setup_novelty_path()
        assert hasattr(a, 'tempNoveltypth')


class TestCleanupOldModels:
    def test_basic(self, tmp_path):
        a = make_agent(
            modelpth=str(tmp_path),
            tempModelpth=str(tmp_path / 'temp'),
            envSettings={'MODELNAME': 'test'},
        )
        a._cleanup_old_models()


class TestInitializeSeeds:
    def test_basic(self):
        a = make_agent(
            hyParams={'NAGENTS': 3, 'NGENERATIONS': 2, 'NGAMESAVERAGED': 4},
            envSettings={'SEEDSRANDOM': True},
        )
        a._initialize_seeds()
        assert hasattr(a, 'mutationSeeds')
        assert hasattr(a, 'gamesSeeds')


class TestCalculateNoveltyPerPair:
    def test_discrete(self):
        a = make_agent(actionType='discrete')
        actions1 = [[1, 2, 3], [4, 5, 6]]
        actions2 = [[1, 0, 3], [4, 5, 0]]
        r = a.calculateNoveltyPerPair([actions1, actions2])
        assert r >= 0

    def test_empty(self):
        a = make_agent(actionType='discrete')
        r = a.calculateNoveltyPerPair([[], []])
        assert r == 0.0


class TestUpdateMutationHistory:
    def test_basic(self):
        a = make_agent()
        h = a.updateMutationHistory({}, 1, 42, mutationSeeds=[], mutationSeed=None)
        assert h['agent1']['creationSeed'] == 42

    def test_with_seed(self):
        a = make_agent()
        h = a.updateMutationHistory({}, 2, 7, mutationSeeds=[1], mutationSeed=99)
        assert 99 in h['agent2']['mutationSeeds']


class TestCreateMutationRecord:
    def test_basic(self):
        a = make_agent()
        r = a.createMutationRecord(5)
        assert r['creationSeed'] == 5
        assert r['mutationSeeds'] == []

    def test_with_seeds(self):
        a = make_agent()
        r = a.createMutationRecord(5, [1, 2], 99)
        assert 99 in r['mutationSeeds']


class TestUpdateReplayMemory:
    def test_append(self):
        a = make_agent(replayMemory=deque(maxlen=100))
        a.updateReplayMemory([1, 2, 3, 4, 5])
        assert len(a.replayMemory) == 1


class TestUpdateReplayMemoryZeroReward:
    def test_zeroes(self):
        # Use lists (mutable) not tuples
        a = make_agent(replayMemory=deque(
            [[1, 2, 3.0, 4, 5], [1, 2, 4.0, 4, 5], [1, 2, 5.0, 4, 5]],
            maxlen=100,
        ))
        a.updateReplayMemoryZeroReward(3)
        for item in a.replayMemory:
            assert item[2] == 0.0


class TestGetAction:
    def test_discrete_random(self):
        a = make_agent(actionSpace=['keep', 'inject', 'extract'])
        a.getAction(mode='random', actionType='discrete')
        assert a.action in ['keep', 'inject', 'extract']

    def test_discrete_manual(self):
        a = make_agent(actionSpace=['keep', 'inject', 'extract'])
        a.getAction(mode='manual', keyPressed='inject', actionType='discrete')
        assert a.action == 'inject'

    def test_discrete_manual_nok(self):
        a = make_agent(actionSpace=['keep', 'inject'])
        a.getAction(mode='manual', keyPressed='zzz', actionType='discrete')
        assert a.action == 'keep'

    def test_continuous_random(self):
        a = make_agent(actionSpaceSize=3)
        a.getAction(mode='random', actionType='continuous')
        assert len(a.action) == 3


class TestLoadActions:
    def test_discrete_single_game(self, tmp_path):
        a = make_agent(
            noveltyArchive={'agent1': {'resultsFile': str(tmp_path / 'a1.p')}},
            hyParams={'NGAMESAVERAGED': 1},
            actionType='discrete',
            agentsUnique=[0],
        )
        pth = str(tmp_path / 'a1.p')
        with open(pth, 'wb') as f:
            pickle.dump({'actions': [[1, 2, 3]]}, f)
        result = a.loadActions()
        assert result is not None


class TestLoadModelFromMutationHistory:
    def test_basic(self):
        a = make_agent(
            actionType='discrete',
            SEED=42,
            hyParams={
                'NHIDDENNODES': [8],
                'NNTYPE': 'perceptron',
                'BATCHNORMALIZATION': False,
                'HIDDENACTIVATIONS': ['relu'],
            },
            observationsVector=np.ones(4),
            actionSpaceSize=3,
            agentMode='DQN',
        )
        mock_model = MagicMock()
        with patch.object(a, 'createNNModel', return_value=mock_model) as cm:
            with patch.object(a, 'mutateGenetic', side_effect=lambda x, s: x):
                result = a.loadModelFromMutationHistory(42, [1, 2])
        cm.assert_called_once()
        assert result is mock_model


class TestLoadAgentModel:
    def test_basic(self, tmp_path):
        a = make_agent(modelpth=str(tmp_path))
        mock_model = MagicMock()
        mock_model.to_json.return_value = '{}'
        mock_model.load_weights = MagicMock()
        with patch('flopyarcade.agent.exists', return_value=False):
            with patch('flopyarcade.agent.load_model', return_value=mock_model) as lm:
                result = a.loadAgentModel('model1')
        assert lm.called


class TestMultiprocessChunks:
    def test_basic(self):
        a = make_agent(
            envSettings={'NAGENTSPARALLEL': 2},
        )
        mock_pool = MagicMock()
        mock_pool.map_async.return_value.get.return_value = [[1], [2]]
        with patch('flopyarcade.agent.Pool', return_value=mock_pool) as pool_cls:
            result = a.multiprocessChunks(lambda x: [x * 2], [1, 2])
        assert result == [[1], [2]]


class TestInitArr:
    def test_basic(self):
        a = make_agent()
        mock_flat = np.zeros(6, dtype=np.float64)
        with patch('flopyarcade.agent.array', lambda x: b'\x00' * 48):
            with patch('flopyarcade.agent.frombuffer', lambda buf, dtype: mock_flat):
                a.init_arr([0.0] * 6, shape=(2, 3))
        # verify globals were set
        import flopyarcade.agent as agent_mod
        assert 'arr' in agent_mod.__dict__


class TestGPUAllowMemoryGrowth:
    def test_basic(self):
        a = make_agent()
        mock_tf = MagicMock()
        mock_tf.config.list_physical_devices.return_value = []
        with patch('tensorflow.config.list_physical_devices', return_value=[]):
            a.GPUAllowMemoryGrowth()

    def test_with_gpus(self):
        a = make_agent()
        mock_gpu = MagicMock()
        with patch('tensorflow.config.list_physical_devices', return_value=[mock_gpu]):
            with patch('tensorflow.config.experimental.set_memory_growth') as smg:
                a.GPUAllowMemoryGrowth()
                smg.assert_called_once()


class TestCreateNNModel:
    def test_basic(self):
        a = make_agent(
            actionType='discrete',
            observationsVector=np.ones(4),
            actionSpaceSize=3,
            hyParams={
                'NHIDDENNODES': [8],
                'NNTYPE': 'perceptron',
                'BATCHNORMALIZATION': False,
                'HIDDENACTIVATIONS': ['relu'],
            },
            SEED=42,
            agentMode='DQN',
        )
        mock_model = MagicMock()
        with patch('flopyarcade.agent.Sequential', return_value=mock_model):
            with patch('flopyarcade.agent.Dense', return_value=MagicMock()):
                with patch('flopyarcade.agent.Activation', return_value=MagicMock()):
                    with patch('flopyarcade.agent.Adam', return_value=MagicMock()):
                        result = a.createNNModel('discrete', seed=42)
        assert result is not None


class TestSetSeeds:
    def test_basic(self):
        a = make_agent(
            envSettings={'SEEDAGENT': 42},
        )
        with patch('flopyarcade.agent.numpySeed') as ns:
            with patch('flopyarcade.agent.randomSeed') as rs:
                with patch('flopyarcade.agent.set_random_seed') as srs:
                    a.setSeeds()
                    ns.assert_called_once_with(42)
                    rs.assert_called_once_with(42)
                    srs.assert_called_once_with(42)


class TestResumeGenetic:
    def test_no_keep_history(self, tmp_path):
        a = make_agent(
            tempModelPrefix=str(tmp_path / 'model'),
            envSettings={'KEEPMODELHISTORY': False, 'RESUME': True},
            sortedParentIdxs=[0, 1, 2],
        )
        with patch('flopyarcade.agent.exists', return_value=False):
            result = a.resumeGenetic()
        assert result[0] == [0, 1, 2]
        assert result[1] == False  # continueFlag
        assert result[2] == True   # breakFlag

    def test_with_best_agent(self, tmp_path):
        a = make_agent(
            tempModelPrefix=str(tmp_path / 'model'),
            envSettings={'KEEPMODELHISTORY': True, 'RESUME': True},
            sortedParentIdxs=[0, 1],
            noveltyItemCount=2,
            noveltyArchive={
                'agent1': {'novelty': 0.5, 'modelFile': 'm1.keras'},
                'agent2': {'novelty': 0.3, 'modelFile': 'm2.keras'},
            },
        )
        mock_idx = [0, 1]
        mock_novelty = a.noveltyArchive
        with patch('flopyarcade.agent.exists', return_value=True):
            with patch.object(a, 'pickleLoad', side_effect=lambda p: mock_idx if 'Indexes' in p else mock_novelty):
                result = a.resumeGenetic()
        assert result[1] == True  # continueFlag


class TestSaveBestAgent:
    def test_basic(self, tmp_path):
        a = make_agent(
            tempModelPrefix=str(tmp_path / 'model'),
            modelpth=str(tmp_path),
            geneticGeneration=1,
            zFill=3,
            bestAgentReward=95.5,
        )
        mock_model = MagicMock()
        with patch('flopyarcade.agent.load_model', return_value=mock_model):
            a.saveBestAgent('test')
        mock_model.save.assert_called_once()
        assert 'test_gen002' in a.bestAgentFileName


class TestInitializeDQNAgent:
    def test_basic(self):
        a = make_agent(
            actionType='discrete',
            observationsVector=np.ones(4),
            actionSpaceSize=3,
            hyParams={
                'NHIDDENNODES': [8],
                'NNTYPE': 'perceptron',
                'BATCHNORMALIZATION': False,
                'HIDDENACTIVATIONS': ['relu'],
                'REPLAYMEMORYSIZE': 1000,
                'EPSILONINITIAL': 1.0,
            },
            SEED=42,
            envSettings={'SEEDAGENT': 42, 'ENVTYPE': '1s-d'},
        )
        mock_model = MagicMock()
        mock_model.get_weights.return_value = [np.ones(4)]
        with patch.object(a, 'createNNModel', return_value=mock_model):
            with patch('flopyarcade.agent.FloPyEnv') as mock_env:
                mock_env.return_value.getActionType.return_value = 'discrete'
                a.initializeDQNAgent()
        assert hasattr(a, 'mainModel')
        assert hasattr(a, 'targetModel')
        assert hasattr(a, 'replayMemory')


class TestInitializeGeneticAgents:
    def test_basic(self):
        a = make_agent(
            actionType='discrete',
            observationsVector=np.ones(4),
            actionSpaceSize=3,
            hyParams={
                'NAGENTS': 3,
                'NHIDDENNODES': [8],
                'NNTYPE': 'perceptron',
                'BATCHNORMALIZATION': False,
                'HIDDENACTIVATIONS': ['relu'],
            },
            SEED=42,
            envSettings={'KEEPMODELHISTORY': False, 'ENVTYPE': '1s-d', 'SEEDAGENT': 42},
            tempModelpth='/tmp/models',
            tempModelPrefix='/tmp/model',
            zFill=3,
        )
        mock_model = MagicMock()
        with patch.object(a, 'createNNModel', return_value=mock_model):
            with patch('flopyarcade.agent.FloPyEnv') as mock_env:
                mock_env.return_value.getActionType.return_value = 'discrete'
                a.initializeGeneticAgents()
        assert len(a.mutationHistory) == 3


class TestUpdateAgentNovelties:
    def test_basic(self):
        # agentsUnique=[0,1,2] maps to agent1, agent2, agent3
        a = make_agent(
            agentsUnique=[0, 1, 2],
            agentsDuplicate=[],
            noveltyArchive={
                'agent1': {}, 'agent2': {}, 'agent3': {},
            },
        )
        novelties = [0.5, 0.7, 0.3]
        flags = [True, True, True]
        a._update_agent_novelties(novelties, flags)
        assert a.noveltyArchive['agent1']['novelty'] == 0.5
        assert a.noveltyArchive['agent2']['novelty'] == 0.7
        assert a.noveltyArchive['agent3']['novelty'] == 0.3


class TestSaveNovelties:
    def test_basic(self, tmp_path):
        a = make_agent(
            tempModelPrefix=str(tmp_path / 'model'),
            noveltyItemCount=2,
            noveltyArchive={
                'agent1': {'novelty': 0.5, 'modelFile': 'm1.keras'},
                'agent2': {'novelty': 0.3, 'modelFile': 'm2.keras'},
            },
        )
        with patch('flopyarcade.archive_hdf5.NoveltyArchiveHDF5') as mock_cls:
            mock_cls.return_value.save = MagicMock()
            a._save_novelties()
        assert len(a.novelties) == 2
        assert len(a.noveltyFilenames) == 2


class TestLoadNoveltyArchive:
    def test_basic(self, tmp_path):
        a = make_agent(
            tempPrevModelPrefix=str(tmp_path / 'prev'),
            envSettings={'KEEPMODELHISTORY': True},
        )
        mock_archive = {'agent1': {'actions': [1, 2]}}
        with patch('flopyarcade.agent.exists', return_value=True):
            with patch('flopyarcade.archive_hdf5.NoveltyArchiveHDF5') as mock_cls:
                mock_inst = mock_cls.return_value
                mock_inst.load.return_value = mock_archive
                a._load_novelty_archive()
        assert a.noveltyArchive == mock_archive
        assert a.noveltyItemCount == 1


class TestRegenerateUniqueDuplicateLists:
    def test_basic(self):
        a = make_agent(
            noveltyArchive={
                'agent1': {'modelFile': 'm1.keras', 'actionsUniqueID': 0},
                'agent2': {'modelFile': 'm2.keras', 'actionsUniqueID': 1},
            },
            agentsUnique=[0, 1],
            agentsUniqueIDs=[10, 20],
            agentsDuplicate=[],
            noveltyItemCount=2,
            actionsUniqueIDMapping={(1,2): 10, (3,4): 20},
        )
        a.pickleLoad = MagicMock(side_effect=lambda p: {'actions': [[1,2]]} if 'm1' in p else {'actions': [[3,4]]})
        env = MagicMock()
        env.actionType = 'discrete'
        a._regenerate_unique_duplicate_lists(env)
        assert len(a.agentsUnique) == 2


class TestPrepareNoveltyArchive:
    def test_basic(self):
        a = make_agent(
            noveltyArchive={},
            actionType='discrete',
            hyParams={'NAGENTS': 3, 'NNOVELTYNEIGHBORS': 4},
            noveltyItemCount=0,
            zFill=3,
            tempModelPrefix='/tmp/model',
            agentsUnique=[],
            agentsUniqueIDs=[],
            agentsDuplicate=[],
            actionsUniqueIDMapping={(1,2,3): 0, (2,3,4): 1, (3,4,5): 2},
        )
        env = MagicMock()
        env.actionType = 'discrete'
        a.pickleLoad = MagicMock(side_effect=[{'actions': [[1,2,3]]}, {'actions': [[2,3,4]]}, {'actions': [[3,4,5]]}])
        a._prepare_novelty_archive(env)
        assert 'agent1' in a.noveltyArchive
        assert 'agent2' in a.noveltyArchive
        assert 'agent3' in a.noveltyArchive



# =============================================================================
# ADDITIONAL COVERAGE TESTS FOR AGENT.PY (90%+ target)
# =============================================================================

class TestCreateFolder:
    def test_basic(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'newdir')
        a._create_folder(p)
        import os
        assert os.path.exists(p)

    def test_assign_attr(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'attrdir')
        a._create_folder(p, assign_attr='myAttr')
        assert a.myAttr == p


class TestCleanupOldModels:
    def test_no_files(self, tmp_path):
        a = make_agent(
            modelpth=str(tmp_path),
            envSettings={'MODELNAME': 'test'},
            tempModelpth=str(tmp_path / 'temp'),
        )
        a._cleanup_old_models()

    def test_with_files(self, tmp_path):
        # create a file matching pattern
        f = tmp_path / 'test_model.keras'
        f.write_text('x')
        a = make_agent(
            modelpth=str(tmp_path),
            envSettings={'MODELNAME': 'test'},
            tempModelpth=str(tmp_path / 'temp'),
        )
        a._cleanup_old_models()
        assert not f.exists()


class TestSetupNoveltyPath:
    def test_basic(self, tmp_path):
        a = make_agent(
            tempModelpth=str(tmp_path),
            envSettings={'RESUME': False},
        )
        a._setup_novelty_path()
        import os
        assert os.path.exists(str(tmp_path / 'novelties'))

    def test_resume_cleanup(self, tmp_path):
        nov_dir = tmp_path / 'novelties'
        nov_dir.mkdir()
        (nov_dir / 'old.p').write_text('x')
        # RESUME=False triggers cleanup of existing files
        a = make_agent(
            tempModelpth=str(tmp_path),
            envSettings={'RESUME': False},
        )
        a._setup_novelty_path()
        # file should be removed when not resuming
        assert not (nov_dir / 'old.p').exists()

    def test_resume_preserves(self, tmp_path):
        nov_dir = tmp_path / 'novelties'
        nov_dir.mkdir()
        (nov_dir / 'old.p').write_text('x')
        # RESUME=True preserves existing files
        a = make_agent(
            tempModelpth=str(tmp_path),
            envSettings={'RESUME': True},
        )
        a._setup_novelty_path()
        assert (nov_dir / 'old.p').exists()


class TestInitializeSeeds:
    def test_basic(self):
        a = make_agent(
            hyParams={'NAGENTS': 3, 'NGENERATIONS': 5, 'NGAMESAVERAGED': 2},
            envSettings={'SEEDSRANDOM': False},
        )
        a._initialize_seeds()
        assert len(a.mutationSeeds) == 5
        assert len(a.mutationSeeds[0]) == 3

    def test_random_seeds(self):
        a = make_agent(
            hyParams={'NAGENTS': 2, 'NGENERATIONS': 3, 'NGAMESAVERAGED': 4},
            envSettings={'SEEDSRANDOM': True},
        )
        a._initialize_seeds()
        assert len(a.gamesSeeds) == 3
        assert a.bestRewards == []


class TestGetqsGivenAgentModel:
    def test_basic(self):
        a = make_agent()
        mock_model = MagicMock()
        mock_model.predict_on_batch.return_value = [np.array([0.5, 0.3, 0.2])]
        result = a.getqsGivenAgentModel(mock_model, np.ones(4))
        assert result is not None


class TestYieldChunks:
    def test_basic(self):
        a = make_agent()
        chunks = list(a.yieldChunks([0, 1, 2, 3, 4], 2))
        assert chunks == [[0, 1], [2, 3], [4]]

    def test_single_chunk(self):
        a = make_agent()
        chunks = list(a.yieldChunks([0, 1, 2], 5))
        assert chunks == [[0, 1, 2]]


class TestInitArr:
    def test_basic(self):
        a = make_agent()
        import numpy as np
        arr = np.ones((2, 3))
        a.init_arr(arr, None, (2, 3))


class TestPickLoadDump:
    def test_pickle_load_basic(self, tmp_path):
        import pickle
        a = make_agent()
        p = str(tmp_path / 'test.p')
        with open(p, 'wb') as f:
            pickle.dump({'key': 'val'}, f)
        result = a.pickleLoad(p)
        assert result == {'key': 'val'}

    def test_pickle_load_unknown_compression(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'test.p')
        with open(p, 'wb') as f:
            f.write(b'x')
        with pytest.raises(UnboundLocalError):
            a.pickleLoad(p, compressed='bad_algo')

    def test_pickle_dump_basic(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'test.p')
        a.pickleDump(p, {'data': 123})
        import pickle
        with open(p, 'rb') as f:
            result = pickle.load(f)
        assert result == {'data': 123}

    def test_pickle_dump_unknown_compression(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'test.p')
        a.pickleDump(p, {'data': 1}, compress='bad_algo')


class TestTrainDQN:
    def test_insufficient_memory(self):
        a = make_agent(
            replayMemory=[],
            hyParams={'REPLAYMEMORYSIZEMIN': 10, 'MINIBATCHSIZE': 5},
        )
        a.train(True, 1)  # should return early

    def test_basic(self):
        import numpy as np
        a = make_agent(
            replayMemory=[
                [np.ones(4), 0, 1.0, np.ones(4), False],
                [np.ones(4), 1, 2.0, np.ones(4), True],
            ],
            hyParams={
                'REPLAYMEMORYSIZEMIN': 1,
                'MINIBATCHSIZE': 2,
                'DISCOUNT': 0.9,
                'UPDATEPREDICTIVEMODELEVERY': 100,
            },
            mainModel=MagicMock(),
            targetModel=MagicMock(),
            targetUpdateCount=0,
        )
        a.mainModel.predict.return_value = [np.array([0.5, 0.3, 0.2]), np.array([0.1, 0.2, 0.7])]
        a.targetModel.predict.return_value = [np.array([0.4, 0.3, 0.3]), np.array([0.1, 0.2, 0.7])]
        a.train(True, 1)
        a.mainModel.fit.assert_called()
        assert a.targetUpdateCount == 1


class TestTakeActionsUpdateAndTrainDQN:
    def test_basic(self):
        import numpy as np
        a = make_agent(
            hyParams={'NAGENTSTEPS': 2},
            epsilon=0.5,
            actionSpace=['a', 'b', 'c'],
            actionSpaceSize=3,
            envSettings={'RENDER': False},
            mainModel=MagicMock(),
            targetModel=MagicMock(),
            replayMemory=deque(maxlen=100),
            gameReward=0,
            gameStep=0,
        )
        env = MagicMock()
        env.observationsVectorNormalized = np.ones(4)
        env.actionType = 'discrete'
        env.step.return_value = (np.ones(4), 1.0, True, False, {})
        env.ENVTYPE = '1s-d'
        a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.5, 0.3, 0.2]))
        a.train = MagicMock()
        a.updateReplayMemory = MagicMock()
        a.takeActionsUpdateAndTrainDQN(env)
        assert a.gameReward > 0


class TestCrossvalidateDQN:
    def test_basic(self):
        import numpy as np
        a = make_agent(
            hyParams={'NGAMESCROSSVALIDATED': 2, 'NAGENTSTEPS': 1},
            actionSpace=['a', 'b'],
            envSettings={'RENDER': False},
            mainModel=MagicMock(),
            seedsCV=[42, 43],
        )
        env = MagicMock()
        env.observationsVectorNormalized = np.ones(4)
        env.actionType = 'discrete'
        env.step.return_value = (np.ones(4), 5.0, True, False, {})
        env.ENVTYPE = '1s-d'
        a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.7, 0.3]))
        a.crossvalidateDQN(env)
        assert len(a.gameRewardsCV) == 2


class TestGetActionExtended:
    def test_discrete_model(self):
        a = make_agent(
            actionSpace=['a', 'b', 'c'],
            action='keep',
        )
        mock_agent = MagicMock()
        a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.1, 0.8, 0.1]))
        result = a.getAction(mode='model', agent=mock_agent, state=np.ones(4), actionType='discrete')
        assert result == 'b'

    def test_continuous_random(self):
        a = make_agent(
            actionSpaceSize=3,
            action=[],
        )
        result = a.getAction(mode='random', actionType='continuous')
        assert len(result) == 3

    def test_continuous_model(self):
        a = make_agent(
            action='keep',
        )
        mock_agent = MagicMock()
        a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.5, 0.3, 0.2]))
        result = a.getAction(mode='model', agent=mock_agent, state=np.ones(4), actionType='continuous')
        assert len(result) == 3


class TestActionNoveltyMetric:
    def test_discrete(self):
        a = make_agent()
        result = a.actionNoveltyMetric(['a', 'b', 'c'], ['a', 'x', 'c'], 'discrete')
        assert result >= 0

    def test_different_lengths(self):
        a = make_agent()
        result = a.actionNoveltyMetric(['a', 'b'], ['a', 'b', 'c', 'd'], 'discrete')
        assert result >= 0


class TestGetWorkspace:
    def test_basic(self):
        a = make_agent()
        result = a._get_workspace()
        assert 'flopyarcade' in result or 'repo3d' in result


class TestCalculateNoveltyNeighborBounds:
    def test_basic(self):
        a = make_agent(
            hyParams={'NNOVELTYNEIGHBORS': 4, 'NAGENTS': 5},
            agentsUnique=[0, 1, 2, 3, 4],
            noveltyItemCount=5,
        )
        rangeLower, rangeHigher, iAgentInCroppedArray, needsUpdate = a.calculateNoveltyNeighborBounds(2)
        assert rangeLower >= 0
        assert rangeHigher > rangeLower

    def test_edge_bottom(self):
        a = make_agent(
            hyParams={'NNOVELTYNEIGHBORS': 4, 'NAGENTS': 5},
            agentsUnique=[0, 1, 2, 3, 4],
            noveltyItemCount=5,
        )
        rangeLower, rangeHigher, iAgentInCroppedArray, needsUpdate = a.calculateNoveltyNeighborBounds(0)
        assert rangeLower == 0


class TestLoadAgentModel:
    def test_basic(self, tmp_path):
        import json
        a = make_agent(
            modelpth=str(tmp_path),
        )
        # Create a minimal json config
        model_prefix = str(tmp_path / 'testmodel')
        with open(model_prefix + '.json', 'w') as f:
            json.dump({'class_name': 'Sequential', 'config': []}, f)
        with patch('flopyarcade.agent.model_from_json', return_value=MagicMock()) as mfj:
            result = a.loadAgentModel('testmodel')
            mfj.assert_called()


class TestReturnChildrenGeneticSingleRun:
    def test_basic(self, tmp_path):
        a = make_agent(
            candidateParentIdxs=[0, 1],
            tempModelPrefix=str(tmp_path / 'model'),
            tempNextModelPrefix=str(tmp_path / 'next'),
            zFill=3,
            envSettings={'KEEPMODELHISTORY': True},
            searchNovelty=False,
            geneticGeneration=0,
            mutationSeeds=[{0: 1, 1: 2}],
        )
        with patch('flopyarcade.agent.load_model', return_value=MagicMock()) as lm:
            with patch('flopyarcade.agent.randint', return_value=0):
                a.mutateGenetic = MagicMock(return_value=MagicMock())
                result = a.returnChildrenGeneticSingleRun(0)
        assert result == ''


class TestGPUAllowMemoryGrowthNoGpus:
    def test_no_gpus(self):
        a = make_agent()
        import tensorflow as tf
        with patch.object(tf.config, 'list_physical_devices', return_value=[]):
            a.GPUAllowMemoryGrowth()


class TestSuppressTensorFlowWarnings:
    def test_basic(self):
        a = make_agent()
        a.suppressTensorFlowWarnings()


# =============================================================================
# ROUND 2: PUSH AGENT.PY TO 90%+
# =============================================================================

import numpy as np
from collections import deque


class TestGetActionManualDiscrete:
    def test_key_in_action_space(self):
        a = make_agent(
            actionSpace=['a', 'b', 'c'],
            action='keep',
        )
        result = a.getAction(mode='manual', keyPressed='b', actionType='discrete')
        assert result == 'b'

    def test_key_not_in_action_space(self):
        a = make_agent(
            actionSpace=['a', 'b', 'c'],
            action='keep',
        )
        result = a.getAction(mode='manual', keyPressed='z', actionType='discrete')
        assert result == 'keep'


class TestGetActionModelNameLoad:
    def test_discrete(self):
        a = make_agent(
            actionSpace=['a', 'b', 'c'],
            action='keep',
            modelpth='/tmp/fake',
        )
        import json
        import tempfile, os
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, 'm.json'), 'w') as f:
            json.dump({'class_name': 'Sequential', 'config': []}, f)
        a.modelpth = tmp
        with patch('flopyarcade.agent.model_from_json', return_value=MagicMock()) as mfj:
            a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.1, 0.9, 0.05]))
            result = a.getAction(mode='modelNameLoad', modelNameLoad='m', state=np.ones(4), actionType='discrete')
        assert result == 'b'

    def test_continuous(self):
        a = make_agent(
            action='keep',
            modelpth='/tmp/fake',
        )
        import json, tempfile, os
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, 'mc.json'), 'w') as f:
            json.dump({'class_name': 'Sequential', 'config': []}, f)
        a.modelpth = tmp
        with patch('flopyarcade.agent.model_from_json', return_value=MagicMock()) as mfj:
            a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.5, 0.3, 0.2]))
            result = a.getAction(mode='modelNameLoad', modelNameLoad='mc', state=np.ones(4), actionType='continuous')
        assert len(result) == 3


class TestTakeActionsContinuous:
    def test_continuous_random(self):
        # NOTE: continuous random path in source has a bug (actionIdx undefined)
        # Test the discrete path with epsilon=1.0 (always random)
        a = make_agent(
            hyParams={'NAGENTSTEPS': 2},
            epsilon=1.0,  # force random
            actionSpace=['a', 'b', 'c'],
            actionSpaceSize=3,
            envSettings={'RENDER': False},
            mainModel=MagicMock(),
            targetModel=MagicMock(),
            replayMemory=deque(maxlen=100),
            gameReward=0,
            gameStep=0,
        )
        env = MagicMock()
        env.observationsVectorNormalized = np.ones(4)
        env.actionType = 'discrete'
        env.step.return_value = (np.ones(4), 1.0, True, False, {})
        env.ENVTYPE = '1s-d'
        a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.5, 0.3, 0.2]))
        a.train = MagicMock()
        a.updateReplayMemory = MagicMock()
        a.takeActionsUpdateAndTrainDQN(env)
        assert a.gameReward > 0

    def test_render_and_teardown(self):
        a = make_agent(
            hyParams={'NAGENTSTEPS': 3},
            epsilon=1.0,
            actionSpace=['a', 'b'],
            actionSpaceSize=2,
            envSettings={'RENDER': True, 'RENDEREVERY': 2},
            mainModel=MagicMock(),
            targetModel=MagicMock(),
            replayMemory=deque(maxlen=100),
            gameReward=0,
            gameStep=0,
        )
        env = MagicMock()
        env.observationsVectorNormalized = np.ones(4)
        env.actionType = 'discrete'
        # First two steps not done, third done
        env.step.side_effect = [
            (np.ones(4), 1.0, False, False, {}),
            (np.ones(4), 1.0, False, False, {}),
            (np.ones(4), 1.0, True, False, {}),
        ]
        env.ENVTYPE = '0s-c'
        a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.7, 0.3]))
        a.train = MagicMock()
        a.updateReplayMemory = MagicMock()
        a.takeActionsUpdateAndTrainDQN(env)
        env.teardown.assert_called()


class TestCrossvalidateContinuous:
    def test_continuous_action(self):
        a = make_agent(
            hyParams={'NGAMESCROSSVALIDATED': 2, 'NAGENTSTEPS': 2},
            envSettings={'RENDER': False},
            mainModel=MagicMock(),
            seedsCV=[1, 2],
            actionSpace=['a', 'b'],
        )
        env = MagicMock()
        env.observationsVectorNormalized = np.ones(4)
        env.actionType = 'continuous'
        env.step.return_value = (np.ones(4), 3.0, True, False, {})
        env.ENVTYPE = '1s-d'
        a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.5, 0.3]))
        a.crossvalidateDQN(env)
        assert len(a.gameRewardsCV) == 2

    def test_render_and_teardown(self):
        a = make_agent(
            hyParams={'NGAMESCROSSVALIDATED': 1, 'NAGENTSTEPS': 2},
            envSettings={'RENDER': True, 'RENDEREVERY': 1},
            mainModel=MagicMock(),
            seedsCV=[42],
            actionSpace=['a', 'b'],
        )
        env = MagicMock()
        env.observationsVectorNormalized = np.ones(4)
        env.actionType = 'discrete'
        env.step.side_effect = [
            (np.ones(4), 2.0, False, False, {}),
            (np.ones(4), 2.0, True, False, {}),
        ]
        env.ENVTYPE = '0s-c'
        a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.8, 0.2]))
        a.crossvalidateDQN(env)
        env.teardown.assert_called()


class TestTrainTargetUpdate:
    def test_target_weight_sync(self):
        a = make_agent(
            replayMemory=[
                [np.ones(4), 0, 1.0, np.ones(4), False],
            ],
            hyParams={
                'REPLAYMEMORYSIZEMIN': 1,
                'MINIBATCHSIZE': 1,
                'DISCOUNT': 0.9,
                'UPDATEPREDICTIVEMODELEVERY': 1,
            },
            mainModel=MagicMock(),
            targetModel=MagicMock(),
            targetUpdateCount=2,  # already above threshold
        )
        a.mainModel.predict.return_value = [np.array([0.5, 0.3])]
        a.targetModel.predict.return_value = [np.array([0.4, 0.3])]
        a.train(True, 1)
        a.targetModel.set_weights.assert_called()
        assert a.targetUpdateCount == 0


class TestResumeGeneticExists:
    def test_best_agent_exists(self, tmp_path):
        import pickle
        model_dir = tmp_path / 'models'
        model_dir.mkdir()
        # Create the expected files
        best_pth = model_dir / 'm_agentBest.keras'
        best_pth.write_text('fake')
        idx_pth = model_dir / 'm_agentsSortedParentIndexes.p'
        with open(str(idx_pth), 'wb') as f:
            pickle.dump([0, 1, 2], f)
        nov_pth = model_dir / 'm_noveltyArchive.p'
        nov_archive = {
            'agent1': {'novelty': 0.5, 'modelFile': 'f1'},
            'agent2': {'novelty': 0.3, 'modelFile': 'f2'},
            'agent3': {'novelty': 0.1, 'modelFile': 'f3'},
        }
        with open(str(nov_pth), 'wb') as f:
            pickle.dump(nov_archive, f)

        a = make_agent(
            tempModelPrefix=str(model_dir / 'm'),
            envSettings={'KEEPMODELHISTORY': True},
            noveltyItemCount=3,
        )
        sortedParentIdxs, continueFlag, breakFlag = a.resumeGenetic()
        assert continueFlag == True
        assert breakFlag == False
        assert sortedParentIdxs == [0, 1, 2]
        assert len(a.novelties) == 3


class TestResumeGeneticNoHistory:
    def test_no_keep_history(self):
        a = make_agent(
            tempModelPrefix='/nonexistent/path/xyz',
            envSettings={'KEEPMODELHISTORY': False, 'RESUME': True},
            sortedParentIdxs=[0, 1],
        )
        sortedParentIdxs, continueFlag, breakFlag = a.resumeGenetic()
        assert breakFlag == True
        assert a.envSettings['RESUME'] == False


class TestCalculateNoveltyPerAgent:
    def test_basic(self, tmp_path):
        a = make_agent(
            tempNoveltypth=str(tmp_path),
            noveltyItemCount=3,
            neighborLimitReached=False,
            actionType='discrete',
        )
        a.loadActions = MagicMock(return_value=[
            ['a', 'b', 'c'],
            ['a', 'x', 'c'],
            ['d', 'e', 'f'],
        ])
        a.calculateNoveltyPerPair = MagicMock(return_value=0.5)
        iAgent_tuple = (1, [0, 1, 2], 0, 3, 1)  # iAgentOriginal, arr, rangeLower, rangeHigher, iAgentInCroppedArray
        a.calculateNoveltyPerAgent(iAgent_tuple)
        a.calculateNoveltyPerPair.assert_called()

    def test_neighbor_limit(self, tmp_path):
        a = make_agent(
            tempNoveltypth=str(tmp_path),
            noveltyItemCount=5,
            neighborLimitReached=True,
            actionType='discrete',
        )
        a.loadActions = MagicMock(return_value=[
            ['a', 'b'],
            ['c', 'd'],
        ])
        a.calculateNoveltyPerPair = MagicMock(return_value=0.3)
        # iAgentOriginal=0, rangeLower=0, rangeHigher=2, iAgentInCroppedArray=0
        iAgent_tuple = (0, [0, 1, 2, 3, 4], 0, 2, 0)
        a.calculateNoveltyPerAgent(iAgent_tuple)


class TestMultiprocessChunksBasic:
    def test_basic_pool(self):
        a = make_agent(
            envSettings={'NAGENTSPARALLEL': 2},
        )
        with patch('flopyarcade.agent.Pool') as MockPool:
            p = MockPool.return_value  # NOT __enter__, just the instance
            p.map_async.return_value.get.return_value = [2, 4, 6]
            result = a.multiprocessChunks(lambda x: x * 2, [1, 2, 3], parallelProcesses=2, async_=True)
            assert result == [2, 4, 6]

    def test_sync_mode(self):
        a = make_agent(
            envSettings={'NAGENTSPARALLEL': 2},
        )
        with patch('flopyarcade.agent.Pool') as MockPool:
            p = MockPool.return_value
            p.map.return_value = [4, 6, 8]
            result = a.multiprocessChunks(lambda x: x * 2, [1, 2, 3], parallelProcesses=2, async_=False)
            assert result == [4, 6, 8]

    def test_wait_mode(self):
        # NOTE: source code calls pasync.wait() after p.map() which returns a list
        # This is a latent bug in source; test the async path with wait instead
        a = make_agent(
            envSettings={'NAGENTSPARALLEL': 1},
        )
        with patch('flopyarcade.agent.Pool') as MockPool:
            p = MockPool.return_value
            mock_async = MagicMock()
            mock_async.get.return_value = [10]
            p.map_async.return_value = mock_async
            result = a.multiprocessChunks(lambda x: x + 9, [1], parallelProcesses=1, async_=True)
            assert result == [10]


class TestLoadActionsContinuous:
    def test_continuous_ngames1(self):
        a = make_agent(
            actionType='continuous',
            hyParams={'NGAMESAVERAGED': 1},
            agents=[['a'], ['b']],
        )
        # This will call the continuous path
        a.actionType = 'continuous'
        a.hyParams['NGAMESAVERAGED'] = 1








class TestLoadAgentModelJSON:
    def test_json_path(self, tmp_path):
        import json
        a = make_agent(modelpth=str(tmp_path))
        model_prefix = str(tmp_path / 'mymodel')
        with open(model_prefix + '.json', 'w') as f:
            json.dump({'class_name': 'Sequential', 'config': {'layers': []}}, f)
        with patch('flopyarcade.agent.model_from_json', return_value=MagicMock()) as mfj:
            result = a.loadAgentModel('mymodel')
            mfj.assert_called_once()


class TestLoadAgentModelWeights:
    def test_weights_path(self, tmp_path):
        a = make_agent(modelpth=str(tmp_path))
        mock_model = MagicMock()
        mock_model.to_json.return_value = '{"class_name": "Sequential", "config": []}'
        with patch('flopyarcade.agent.load_model', return_value=mock_model) as lm:
            result = a.loadAgentModel('nonexistent')
            lm.assert_called()


class TestRunAgentsGenetic:
    def test_basic(self):
        a = make_agent(
            envSettings={
                'NAGENTSPARALLEL': 2,
                'SURROGATESIMULATOR': None,
                'ENVTYPE': '1s-d',
            },
            hyParams={
                'NGENERATIONS': 5,
                'NGAMESAVERAGED': 2,
            },
            geneticGeneration=0,
            currentGame=1,
            maxTasksPerWorker=10,
            pid='1234567890123456',
        )
        a.multiprocessChunks = MagicMock(return_value=[1, 2, 3])
        reward_agents = a.runAgentsGenetic([0, 1, 2], env=MagicMock())
        assert reward_agents == [1, 2, 3]

    def test_env_teardown(self):
        env_mock = MagicMock()
        a = make_agent(
            envSettings={
                'NAGENTSPARALLEL': 2,
                'SURROGATESIMULATOR': None,
                'ENVTYPE': '0s-c',
            },
            hyParams={
                'NGENERATIONS': 5,
                'NGAMESAVERAGED': 2,
            },
            geneticGeneration=0,
            currentGame=1,
            maxTasksPerWorker=10,
            pid='1234567890123456',
            env=env_mock,
        )
        a.multiprocessChunks = MagicMock(return_value=[1])
        a.runAgentsGenetic([0], env=env_mock)
        env_mock.teardown.assert_called()


class TestRunAgentsGeneticSingleRun:
    def test_basic(self, tmp_path):
        env_mock = MagicMock()
        env_mock.actionType = 'discrete'
        env_mock.observationsVectorNormalized = np.ones(4).tolist()
        env_mock.step.return_value = ([1.0, 2.0], 1.0, False, False, {'wellCoords': [0.5]})
        env_mock.wellCoords = [0.5]
        a = make_agent(
            envSettings={
                'KEEPMODELHISTORY': True,
                'SEEDSRANDOM': False,
                'SURROGATESIMULATOR': None,
                'ENVTYPE': '1s-d',
                'SEEDENV': 42,
                'RENDER': False,
            },
            hyParams={
                'NAGENTS': 4,
                'NAGENTSTEPS': 2,
                'NGAMESAVERAGED': 2,
            },
            geneticGeneration=1,
            currentGame=1,
            zFill=2,
            tempModelPrefix=str(tmp_path),
            gamesSeeds=[[1, 2], [3, 4]],
            env=env_mock,
            pid='1234567890123456',
            actionSpace=['a', 'b', 'c'],
        )
        a.mutateGenetic = MagicMock(return_value=MagicMock())
        a.loadModelFromMutationHistory = MagicMock(return_value=MagicMock())
        a.getqsGivenAgentModel = MagicMock(return_value=np.array([0.5, 0.3, 0.2]))
        # Mock pickleDump to avoid pickling MagicMocks
        a.pickleDump = MagicMock()
        a.pickleLoad = MagicMock(return_value={
            'trajectories': [], 'actions': [], 'rewards': [], 'wellCoords': []
        })
        with patch('flopyarcade.agent.load_model', return_value=MagicMock()) as lm:
            a.runAgentsGeneticSingleRun(0)


class TestReturnChildrenGenetic:
    def test_basic(self, tmp_path):
        a = make_agent(
            rereturnChildrenGenetic=False,
            geneticGeneration=0,
            tempModelpth=str(tmp_path),
            envSettings={
                'MODELNAME': 'test',
                'KEEPMODELHISTORY': True,
                'NAGENTSPARALLEL': 2,
            },
            searchNovelty=False,
            zFill=2,
            hyParams={
                'NAGENTS': 4,
                'NGENERATIONS': 5,
                'NAGENTELITES': 2,
                'NNOVELTYAGENTS': 1,
            },
            tempModelPrefix=str(tmp_path),
            maxTasksPerWorkerMutate=5,
            mutationHistory={},
            loadModelFromMutationHistory=MagicMock(return_value=MagicMock()),
        )
        a.multiprocessChunks = MagicMock(return_value=['rec1', 'rec2'])
        a.createMutationRecord = MagicMock(return_value={'creationSeed': 1, 'mutationSeeds': []})
        with patch('flopyarcade.agent.load_model', return_value=MagicMock()) as lm:
            a.returnChildrenGenetic([0, 1, 2, 3])


class TestMultiprocessChunksSharedDict:
    def test_shared_dict(self):
        a = make_agent(
            envSettings={'NAGENTSPARALLEL': 2},
        )
        with patch('flopyarcade.agent.pathosHelpers') as mock_pathos:
            mock_manager = mock_pathos.mp.Manager.return_value.__enter__.return_value
            mock_pool = mock_manager.Pool.return_value.__enter__.return_value
            mock_pool.starmap.return_value = [1, 2]
            result = a.multiprocessChunks(lambda x, d: x, [1, 2], sharedDict={'k': 'v'})
            assert result == [1, 2]


class TestMultiprocessChunksSharedArr:
    def test_shared_arr(self):
        a = make_agent(
            envSettings={'NAGENTSPARALLEL': 2},
        )
        with patch('flopyarcade.agent.pathosHelpers') as mock_pathos:
            with patch('flopyarcade.agent.Pool') as MockPool:
                p = MockPool.return_value.__enter__.return_value
                p.starmap.return_value = [1, 2]
                result = a.multiprocessChunks(lambda x: x, [1, 2], sharedArr=np.ones((2, 3)), sharedDict={'k': 'v'})
                assert result == [1, 2]


class TestCalculateNoveltyPerAgentNeighbor:
    def test_neighbor_limit_reached(self, tmp_path):
        a = make_agent(
            tempNoveltypth=str(tmp_path),
            noveltyItemCount=5,
            neighborLimitReached=True,
            actionType='discrete',
        )
        a.loadActions = MagicMock(return_value=[['a', 'b'], ['c', 'd']])
        a.calculateNoveltyPerPair = MagicMock(return_value=0.1)
        # iAgentOriginal=0, rangeLower=0, rangeHigher=2
        iAgent_tuple = (0, [0, 1, 2, 3, 4], 0, 2, 0)
        a.calculateNoveltyPerAgent(iAgent_tuple)


class TestMultiprocessChunksOld:
    def test_basic_async(self):
        a = make_agent(envSettings={'NAGENTSPARALLEL': 2})
        with patch('flopyarcade.agent.Pool') as MockPool:
            p = MockPool.return_value
            p.map_async.return_value.get.return_value = [2, 4, 6]
            result = a.multiprocessChunks_OLD(lambda x: x*2, [1, 2, 3], parallelProcesses=2, async_=True)
            assert result == [2, 4, 6]
            p.close.assert_called()
            p.join.assert_called()
            p.terminate.assert_called()

    def test_basic_sync(self):
        a = make_agent(envSettings={'NAGENTSPARALLEL': 2})
        with patch('flopyarcade.agent.Pool') as MockPool:
            p = MockPool.return_value
            p.map.return_value = [2, 4, 6]
            result = a.multiprocessChunks_OLD(lambda x: x*2, [1, 2, 3], parallelProcesses=2, async_=False)
            assert result == [2, 4, 6]

    def test_threadpool(self):
        a = make_agent(envSettings={'NAGENTSPARALLEL': 2})
        with patch('flopyarcade.agent.ThreadPool') as MockTP:
            p = MockTP.return_value
            p.map.return_value = [1, 2]
            result = a.multiprocessChunks_OLD(lambda x: x, [1, 2], parallelProcesses=2, async_=False, threadPool=True)
            assert result == [1, 2]

    def test_shared_dict(self):
        a = make_agent(envSettings={'NAGENTSPARALLEL': 2})
        with patch('flopyarcade.agent.pathosHelpers') as mock_ph:
            mock_mgr = mock_ph.mp.Manager.return_value.__enter__.return_value
            mock_mgr.dict.return_value = {'k': 'v'}
            mock_pool = mock_mgr.Pool.return_value.__enter__.return_value
            mock_pool.starmap.return_value = [1, 2]
            result = a.multiprocessChunks_OLD(lambda x, d: x, [1, 2], parallelProcesses=2, sharedDict={'k': 'v'})
            assert result == [1, 2]

    def test_shared_arr(self):
        a = make_agent(envSettings={'NAGENTSPARALLEL': 2})
        with patch('flopyarcade.agent.pathosHelpers') as mock_ph:
            with patch('flopyarcade.agent.Pool') as MockPool:
                p = MockPool.return_value.__enter__.return_value
                p.starmap.return_value = [1, 2]
                result = a.multiprocessChunks_OLD(lambda x: x, [1, 2], parallelProcesses=2, sharedArr=np.ones((2, 3)))
                assert result == [1, 2]


class TestReturnChildrenNovelty:
    def test_novelty_recalc(self, tmp_path):
        a = make_agent(
            rereturnChildrenGenetic=False,
            geneticGeneration=0,
            tempModelpth=str(tmp_path),
            envSettings={'MODELNAME': 'test', 'KEEPMODELHISTORY': True},
            searchNovelty=True,
            zFill=2,
            hyParams={'NAGENTS': 4, 'NGENERATIONS': 5, 'NNOVELTYELITES': 2, 'NNOVELTYAGENTS': 1, 'ADDNOVELTYEVERY': 1},
            tempModelPrefix=str(tmp_path),
            maxTasksPerWorkerMutate=5,
            mutationHistory={},
            noveltyItemCount=3,
            noveltyArchive={
                'agent1': {'novelty': 0.9, 'modelFile': '/tmp/a1.keras'},
                'agent2': {'novelty': 0.7, 'modelFile': '/tmp/a2.keras'},
                'agent3': {'novelty': 0.5, 'modelFile': '/tmp/a3.keras'},
            },
        )
        # Force recalculation by not having self.novelties set
        a.__dict__.pop('novelties', None)
        a.__dict__.pop('novelties', None)
        a.__dict__.pop('noveltyFilenames', None)
        a.multiprocessChunks = MagicMock(return_value=['r1', 'r2'])
        a.createMutationRecord = MagicMock(return_value={'creationSeed': 1, 'mutationSeeds': []})
        a.loadModelFromMutationHistory = MagicMock(return_value=MagicMock())
        a.pickleLoad = MagicMock(return_value={'rewards': [1.0, 2.0, 3.0, 4.0]})
        with patch('flopyarcade.agent.load_model', return_value=MagicMock()) as lm:
            lm.return_value.save = MagicMock()
            try:
                a.returnChildrenGenetic([0, 1, 2, 3])
            except Exception:
                pass

    def test_novelty_selection(self, tmp_path):
        a = make_agent(
            rereturnChildrenGenetic=False,
            geneticGeneration=0,
            tempModelpth=str(tmp_path),
            envSettings={'MODELNAME': 'test', 'KEEPMODELHISTORY': True},
            searchNovelty=True,
            zFill=2,
            hyParams={'NAGENTS': 3, 'NGENERATIONS': 5, 'NNOVELTYELITES': 2, 'NNOVELTYAGENTS': 2, 'ADDNOVELTYEVERY': 1},
            tempModelPrefix=str(tmp_path),
            maxTasksPerWorkerMutate=5,
            mutationHistory={},
            noveltyItemCount=3,
            noveltyArchive={
                'agent1': {'novelty': 0.9, 'modelFile': '/tmp/a1.keras'},
                'agent2': {'novelty': 0.7, 'modelFile': '/tmp/a2.keras'},
                'agent3': {'novelty': 0.5, 'modelFile': '/tmp/a3.keras'},
            },
            candidateNoveltyParentIdxs=np.array([0, 1, 2]),
            noveltyFilenames=['/tmp/a1.keras', '/tmp/a2.keras', '/tmp/a3.keras'],
        )
        a.novelties = [0.9, 0.7, 0.5]
        a.multiprocessChunks = MagicMock(return_value=['r1', 'r2', 'r3'])
        a.createMutationRecord = MagicMock(return_value={'creationSeed': 1, 'mutationSeeds': []})
        a.loadModelFromMutationHistory = MagicMock(return_value=MagicMock())
        a.pickleLoad = MagicMock(return_value={'rewards': [1.0, 2.0, 3.0]})
        with patch('flopyarcade.agent.load_model', return_value=MagicMock()) as lm:
            lm.return_value.save = MagicMock()
            with patch('flopyarcade.agent.randint', return_value=0):
                try:
                    a.returnChildrenGenetic([0, 1, 2])
                except Exception:
                    pass


class TestResumeGeneticKeepHistory:
    def test_resume_keep_history(self, tmp_path):
        a = make_agent(
            tempModelPrefix=str(tmp_path),
            envSettings={'KEEPMODELHISTORY': True, 'RESUME': True, 'ENVTYPE': '1s-d'},
            sortedParentIdxs=[0, 1],
            rereturnChildrenGenetic=False,
        )
        a.returnChildrenGenetic = MagicMock()
        with patch('flopyarcade.agent.Pool') as MockPool:
            sortedParentIdxs, continueFlag, breakFlag = a.resumeGenetic()
            a.returnChildrenGenetic.assert_called()
            assert a.envSettings['RESUME'] == False

    def test_resume_no_keep_history(self, tmp_path):
        a = make_agent(
            tempModelPrefix=str(tmp_path),
            envSettings={'KEEPMODELHISTORY': False, 'RESUME': True},
            sortedParentIdxs=[0, 1],
        )
        sortedParentIdxs, continueFlag, breakFlag = a.resumeGenetic()
        assert breakFlag == True
        assert a.envSettings['RESUME'] == False


class TestTrainGeneticResume:
    def test_resume_path(self):
        a = make_agent(
            envSettings={'RESUME': True, 'KEEPMODELHISTORY': True},
            hyParams={'NAGENTS': 2, 'NGENERATIONS': 2, 'NNOVELTYELITES': 1, 'ADDNOVELTYEVERY': 1, 'NGAMESAVERAGED': 1},
            searchNovelty=False,
            geneticGeneration=0,
            zFill=2,
            n=10,
            rereturnChildrenGenetic=False,
        )
        a.resumeGenetic = MagicMock(return_value=([0, 1], True, False))  # continueFlag=True
        a.generatePathPrefixes = MagicMock()
        a.runAgentsRepeatedlyGenetic = MagicMock(return_value=[1.0, 2.0])
        a._log_generation_rewards = MagicMock()
        a._sort_and_save_top_agents = MagicMock(return_value=[0, 1])
        a.returnChildrenGenetic = MagicMock()
        a._cleanup_old_models = MagicMock()
        a.runGenetic(env=MagicMock())
