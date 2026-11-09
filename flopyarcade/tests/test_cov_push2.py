
"""Coverage tests - push to 90%."""
import pytest
import numpy as np
import os
import sys
import tempfile
import pickle
import json
import shutil
from unittest.mock import MagicMock, patch, PropertyMock
from collections import defaultdict
from itertools import count


# === BENCHMARK.PY ===

class TestBenchMore:
    def test_load_keras_agent_dir_tune(self):
        from flopyarcade.benchmark import load_keras_agent
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, '.tune_metadata'), 'wb') as f:
                pickle.dump({}, f)
            with patch('flopyarcade.benchmark._load_rllib_checkpoint', return_value=MagicMock()) as m:
                load_keras_agent(d)
                m.assert_called_once()
        finally:
            shutil.rmtree(d)

    def test_load_keras_agent_keras_file(self):
        from flopyarcade.benchmark import load_keras_agent
        with patch('tensorflow.keras.models.load_model', return_value=MagicMock()) as m:
            load_keras_agent('/tmp/x.keras')
            m.assert_called_once_with('/tmp/x.keras', compile=False)

    def test_load_rllib_pickle_dict_no_policy(self):
        from flopyarcade.benchmark import _load_rllib_pickle
        tmpf = tempfile.NamedTemporaryFile(suffix='.p', delete=False)
        try:
            with open(tmpf.name, 'wb') as f:
                pickle.dump({'foo': 1}, f)
            with pytest.raises(ValueError):
                _load_rllib_pickle(tmpf.name)
        finally:
            os.unlink(tmpf.name)

    def test_load_rllib_pickle_invalid_type(self):
        from flopyarcade.benchmark import _load_rllib_pickle
        tmpf = tempfile.NamedTemporaryFile(suffix='.p', delete=False)
        try:
            with open(tmpf.name, 'wb') as f:
                pickle.dump(42, f)
            with pytest.raises(TypeError):
                _load_rllib_pickle(tmpf.name)
        finally:
            os.unlink(tmpf.name)

    def test_load_npz_policy_valid(self):
        from flopyarcade.benchmark import _load_npz_policy
        p = tempfile.mktemp(suffix='.npz')
        np.savez(p, kernel_1=np.random.rand(5, 10).astype(np.float32),
                 bias_1=np.random.rand(10).astype(np.float32),
                 kernel_2=np.random.rand(10, 3).astype(np.float32),
                 bias_2=np.random.rand(3).astype(np.float32))
        m = _load_npz_policy(p)
        assert m is not None
        os.unlink(p)

    def test_load_npz_policy_invalid(self):
        from flopyarcade.benchmark import _load_npz_policy
        p = tempfile.mktemp(suffix='.npz')
        np.savez(p, unrelated=np.array([1.0]))
        with pytest.raises(ValueError):
            _load_npz_policy(p)
        os.unlink(p)

    def test_run_benchmark_verbose(self):
        from flopyarcade.benchmark import run_benchmark
        with patch('flopyarcade.benchmark.load_keras_agent', return_value=MagicMock()):
            with patch('flopyarcade.FloPyArcade') as FA:
                with patch('flopyarcade.FloPyAgent.suppressTensorFlowWarnings'):
                    FA.return_value.play.return_value = 50.0
                    r = run_benchmark('/m.keras', envtype='1s-d', n_games=2, verbose=True, output_path=None)
                    assert r['n_games'] == 2

    def test_run_benchmark_game_fail(self):
        from flopyarcade.benchmark import run_benchmark
        with patch('flopyarcade.benchmark.load_keras_agent', return_value=MagicMock()):
            with patch('flopyarcade.FloPyArcade') as FA:
                with patch('flopyarcade.FloPyAgent.suppressTensorFlowWarnings'):
                    FA.return_value.play.side_effect = Exception('boom')
                    r = run_benchmark('/m.keras', envtype='1s-d', n_games=1, verbose=False)
                    assert r['mean_reward'] == 0.0

    def test_main_quiet(self):
        from flopyarcade.benchmark import main
        with patch('sys.argv', ['b', '--model', '/m.keras', '--games', '1', '--quiet']):
            with patch('flopyarcade.benchmark.run_benchmark', return_value={'mean_reward': 1.0}) as m:
                r = main()
                assert r['mean_reward'] == 1.0

    def test_load_rllib_checkpoint_pickle(self):
        from flopyarcade.benchmark import _load_rllib_checkpoint
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, '.tune_metadata'), 'wb') as f:
                pickle.dump({}, f)
            pf = os.path.join(d, 'pol.p')
            policy = {
                'fc_1/kernel': np.random.rand(5, 10).astype(np.float32),
                'fc_1/bias': np.random.rand(10).astype(np.float32),
                'dense/kernel': np.random.rand(10, 3).astype(np.float32),
                'dense/bias': np.random.rand(3).astype(np.float32)
            }
            with open(pf, 'wb') as f:
                pickle.dump(policy, f)
            r = _load_rllib_checkpoint(d, os.path.join(d, '.tune_metadata'))
            assert r is not None
        finally:
            shutil.rmtree(d)

    def test_load_rllib_checkpoint_not_found(self):
        from flopyarcade.benchmark import _load_rllib_checkpoint
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, '.tune_metadata'), 'wb') as f:
                pickle.dump({}, f)
            with pytest.raises(ValueError):
                _load_rllib_checkpoint(d, os.path.join(d, '.tune_metadata'))
        finally:
            shutil.rmtree(d)


# === AGENT.PY ===

def make_agent():
    from flopyarcade.agent import FloPyAgent
    a = object.__new__(FloPyAgent)
    a.hyParams = {
        'EPSILONMIN': 0.01, 'EPSILONDECAY': 0.95, 'NAGENTS': 4,
        'NNOVELTYELITES': 1, 'NNOVELTYNEIGHBORS': 2, 'NGAMESAVERAGED': 1,
        'NAGENTELITES': 2, 'KEEPMODELHISTORY': False,
        'NHIDDENNODES': [16], 'NNTYPE': 'perceptron',
        'HIDDENACTIVATIONS': ['relu'], 'BATCHNORMALIZATION': False,
        'DROPOUTS': [0.0], 'REPLAYMEMORYSIZEMIN': 10, 'MINIBATCHSIZE': 8,
        'DISCOUNT': 0.99, 'UPDATEPREDICTIVEMODELEVERY': 100,
        'NAGENTSTEPS': 5, 'MUTATIONPROBABILITY': 0.1, 'MUTATIONPOWER': 0.5,
        'NGAMESCROSSVALIDATED': 1, 'CROSSVALIDATEEVERY': 5,
        'NGENERATIONS': 2, 'NOVELTYSEARCH': False, 'ADDNOVELTYEVERY': 1,
        'NNOVELTYAGENTS': 1, 'ARCHITECTUREVARY': False,
        'NGAMES': 1, 'REWARDMINTOSAVE': 0.5, 'SEEDAGENT': 42, 'SEEDENV': 100
    }
    a.envSettings = {
        'SEEDSRANDOM': False, 'NAGENTSPARALLEL': 1, 'RESUME': False,
        'KEEPMODELHISTORY': False, 'ENVTYPE': '1s-d',
        'MODELNAME': 'test', 'SURROGATESIMULATOR': None,
        'RENDER': False, 'RENDEREVERY': 10,
        'NLAY': 1, 'NROW': 10, 'NCOL': 10,
        'PATHMF2005': '/tmp', 'PATHMP6': '/tmp',
        'SEEDAGENT': 42, 'SEEDENV': 100
    }
    a.epsilon = 0.5
    a.actionType = 'discrete'
    a.agentsUnique = [0, 1]
    a.agentsUniqueIDs = [10, 20]
    a.agentsDuplicate = [2]
    a.noveltyItemCount = 3
    a.noveltyArchive = {
        'agent1': {'actionsUniqueID': 10, 'modelFile': '/tmp/a1.keras', 'resultsFile': '/tmp/a1_results.p'},
        'agent2': {'actionsUniqueID': 20, 'modelFile': '/tmp/a2.keras', 'resultsFile': '/tmp/a2_results.p'},
        'agent3': {'actionsUniqueID': 10, 'modelFile': '/tmp/a3.keras', 'resultsFile': '/tmp/a3_results.p'}
    }
    a.zFill = 4
    a.rewards = [10, 20, 30, 40]
    a.tempModelPrefix = '/tmp/t'
    a.tempPrevModelPrefix = '/tmp/p'
    a.tempModelpth = '/tmp/temp'
    a.tempNextModelPrefix = '/tmp/tn'
    a.actionsUniqueIDMapping = defaultdict(count().__next__)
    a.searchNovelty = False
    a.cores = 1
    a.maxTasksPerWorkerNoveltySearch = 100
    a.maxTasksPerWorkerMutate = 100
    a.maxTasksPerWorker = 100
    a.mutationHistory = {}
    a.env = MagicMock()
    a.env.actionType = 'discrete'
    a.agentMode = 'genetic'
    a.rereturnChildrenGenetic = False
    a.n = 1
    a.tempNoveltypth = '/tmp/novelty'
    a.wrkspc = tempfile.mkdtemp()
    a.modelpth = a.wrkspc
    a.actionSpace = ['up', 'down', 'left', 'right', 'keep']
    a.actionSpaceSize = 5
    a.pid = 'a' * 36
    a.geneticGeneration = 0
    a.currentGame = 1
    a.bestAgentReward = 0.0
    a.neighborLimitReached = False
    a.novelties = []
    a.noveltyFilenames = []
    a.candidateNoveltyParentIdxs = []
    a.candidateParentIdxs = [0, 1]
    a.mutationSeeds = [[1, 2, 3, 4] for _ in range(2)]
    a.gamesSeeds = [[1, 2] for _ in range(2)]
    a.seedsCV = [42]
    a.gameRewardsCV = []
    a.average_rewardCV = 0.0
    a.min_rewardCV = 0.0
    a.max_rewardCV = 0.0
    a.flagSkipGeneration = False
    a.sortedParentIdxs = [0, 1]
    a.gameReward = 0
    a.gameStep = 1
    a.SEED = 42
    a.observationsVector = np.array([1.0, 2.0, 3.0])
    return a


class TestAgentMore:
    def test_decay_eps(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        FloPyAgent._decay_epsilon(a)
        assert a.epsilon < 0.5

    def test_decay_eps_below_min(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.epsilon = 0.005
        FloPyAgent._decay_epsilon(a)
        assert a.epsilon == 0.005

    def test_validate_hyp_ok(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        FloPyAgent._validate_hyperparams(a)

    def test_validate_hyp_err(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.hyParams['NAGENTS'] = 1
        a.hyParams['NNOVELTYELITES'] = 2
        with pytest.raises(ValueError):
            FloPyAgent._validate_hyperparams(a)

    def test_init_env(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        env = MagicMock()
        env.actionType = 'discrete'
        FloPyAgent._initialize_environment(a, env, True)
        assert a.env is env

    def test_init_parallel(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        FloPyAgent._initialize_parallel_settings(a)
        assert a.cores == 1

    def test_gen_pid(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        FloPyAgent._generate_unique_pid(a)
        assert len(a.pid) == 36

    def test_get_action_type_d(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        assert FloPyAgent.getActionType(a, '1s-d') == 'discrete'

    def test_get_action_type_c(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        assert FloPyAgent.getActionType(a, '1s-c') == 'continuous'

    def test_log_gen_rewards(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        with patch('flopyarcade.agent.logger'):
            FloPyAgent._log_generation_rewards(a)

    def test_sort_save_top(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        with patch.object(FloPyAgent, 'pickleDump'):
            r = FloPyAgent._sort_and_save_top_agents(a)
            assert len(r) == 2

    def test_update_discrete_uniques(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.agentsUnique = []
        a.agentsUniqueIDs = []
        a.agentsDuplicate = []
        a.noveltyItemCount = 0
        FloPyAgent._update_discrete_uniques(a, 99)
        assert 99 in a.agentsUniqueIDs

    def test_update_discrete_dup(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.agentsUnique = [0]
        a.agentsUniqueIDs = [99]
        a.agentsDuplicate = []
        a.noveltyItemCount = 1
        FloPyAgent._update_discrete_uniques(a, 99)
        assert len(a.agentsDuplicate) == 1

    def test_reset_cont_uniques(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.noveltyItemCount = 2
        FloPyAgent._reset_continuous_uniques(a)
        assert len(a.agentsUnique) == 3

    def test_print_novelty_stats(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        FloPyAgent._print_novelty_stats(a)

    def test_build_novelty_args(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        args, flags = FloPyAgent._build_novelty_args(a)
        assert len(args) == 2

    def test_novelty_search_skip(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.searchNovelty = False
        FloPyAgent.perform_novelty_search(a, MagicMock())

    def test_novelty_search_run(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.searchNovelty = True
        env = MagicMock()
        env.actionType = 'discrete'
        with patch.object(FloPyAgent, '_prepare_novelty_archive'):
            with patch.object(FloPyAgent, '_print_novelty_stats'):
                with patch.object(FloPyAgent, '_build_novelty_args', return_value=([], [True])):
                    with patch.object(FloPyAgent, '_run_novelty_search', return_value=[1.0]):
                        with patch.object(FloPyAgent, '_update_agent_novelties'):
                            with patch.object(FloPyAgent, '_save_novelties'):
                                FloPyAgent.perform_novelty_search(a, env)

    def test_update_agent_novelties(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.agentsUnique = [0, 1]
        a.agentsDuplicate = [2]
        a.noveltyArchive['agent1']['actionsUniqueID'] = 0
        a.noveltyArchive['agent2']['actionsUniqueID'] = 1
        a.noveltyArchive['agent3']['actionsUniqueID'] = 0
        FloPyAgent._update_agent_novelties(a, [1.0, 2.0], [True, True])

    def test_prepare_novelty_archive(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.noveltyItemCount = 0
        a.noveltyArchive = {}
        a.agentsUnique = []
        a.agentsUniqueIDs = []
        a.agentsDuplicate = []
        env = MagicMock()
        env.actionType = 'discrete'
        with patch.object(FloPyAgent, 'pickleLoad', return_value={'actions': [[1], [2]]}):
            with patch.object(FloPyAgent, '_update_discrete_uniques'):
                FloPyAgent._prepare_novelty_archive(a, env)
                assert a.noveltyItemCount == 4

    def test_save_novelties(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.noveltyArchive['agent1']['novelty'] = 0.5
        a.noveltyArchive['agent2']['novelty'] = 0.3
        a.noveltyArchive['agent3']['novelty'] = 0.7
        with patch('flopyarcade.archive_hdf5.NoveltyArchiveHDF5'):
            FloPyAgent._save_novelties(a)
        assert len(a.novelties) == 3

    def test_load_novelty_archive(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.tempPrevModelPrefix = '/tmp/nonexistent'
        with patch.object(FloPyAgent, 'pickleLoad', return_value={'agent1': {}}):
            with patch('flopyarcade.agent.exists', return_value=False):
                FloPyAgent._load_novelty_archive(a)

    def test_count_parameters(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        agent_model = MagicMock()
        agent_model.get_weights.return_value = [np.array([1, 2, 3]), np.array([4, 5])]
        n = FloPyAgent.countParameters(a, agent_model)
        assert n == 5

    def test_mutate_decision(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        r = FloPyAgent.mutateDecision(a, 1.0)
        assert r is True

    def test_mutate_decisions(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        r = FloPyAgent.mutateDecisions(a, 1.0, (3,))
        assert len(r) == 3

    def test_getqs_given_agent(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        model = MagicMock()
        model.predict_on_batch.return_value = np.array([[0.1, 0.5, 0.4]])
        r = FloPyAgent.getqsGivenAgentModel(a, model, np.array([1.0, 2.0]))
        assert r is not None

    def test_check_parameter_mean(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        agent = MagicMock()
        agent.get_weights.return_value = [np.array([1.0, 2.0, 3.0])]
        r = FloPyAgent.checkParameterMean(a, agent)
        assert r == 2.0

    def test_load_model_mutation(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.actionType = 'discrete'
        with patch.object(FloPyAgent, 'createNNModel', return_value=MagicMock()):
            with patch.object(FloPyAgent, 'mutateGenetic', side_effect=lambda agent, s: agent):
                r = FloPyAgent.loadModelFromMutationHistory(a, 1, [2, 3])
                assert r is not None

    def test_update_mutation_history(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        r = FloPyAgent.updateMutationHistory(a, {}, 1, 5, mutationSeeds=[], mutationSeed=None)
        assert 'agent1' in r

    def test_create_mutation_record(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        r = FloPyAgent.createMutationRecord(a, 5, mutationSeeds=[1, 2], mutationSeed=None)
        assert 'creationSeed' in r

    def test_update_replay_memory(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.replayMemory = []
        FloPyAgent.updateReplayMemory(a, (1, 2, 3, 4, 5))
        assert len(a.replayMemory) == 1

    def test_update_replay_memory_zero(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.replayMemory = [[0.1, 0, 0.5, 0.2, False], [0.3, 1, 0.7, 0.4, False]]
        FloPyAgent.updateReplayMemoryZeroReward(a, 2)
        assert a.replayMemory[0][2] == 0.0
        assert a.replayMemory[1][2] == 0.0

    def test_get_action_random(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        with patch('flopyarcade.agent.randint', return_value=0):
            r = FloPyAgent.getAction(a, mode='random')
        assert r == 'up'

    def test_get_action_model(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        agent = MagicMock()
        with patch.object(FloPyAgent, 'getqsGivenAgentModel', return_value=np.array([0.1, 0.5, 0.2, 0.1, 0.1])):
            with patch('flopyarcade.agent.argmax', return_value=1):
                r = FloPyAgent.getAction(a, mode='model', agent=agent, state=np.array([1.0]))
        assert r == 'down'

    def test_load_actions(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.agentsUnique = [0, 1]
        a.noveltyArchive['agent1']['resultsFile'] = '/tmp/a1_results.p'
        a.noveltyArchive['agent2']['resultsFile'] = '/tmp/a2_results.p'
        a.hyParams['NGAMESAVERAGED'] = 1
        with patch.object(FloPyAgent, 'pickleLoad', return_value={'actions': [['up', 'down']]}):
            r = FloPyAgent.loadActions(a, agents=[0, 1])
        assert r is not None

    def test_multiprocess_chunks(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        with patch('flopyarcade.agent.Pool') as PP:
            p = PP.return_value
            p.map_async.return_value.get.return_value = [2, 4, 6]
            r = FloPyAgent.multiprocessChunks(a, lambda x: x * 2, [1, 2, 3])
            assert r == [2, 4, 6]

    def test_yield_chunks(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        chunks = list(FloPyAgent.yieldChunks(a, [1, 2, 3, 4, 5], 2))
        assert sum(len(c) for c in chunks) == 5

    def test_pickle_load(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        tmpf = tempfile.NamedTemporaryFile(suffix='.p', delete=False)
        with open(tmpf.name, 'wb') as f:
            pickle.dump({'x': 1}, f)
        r = FloPyAgent.pickleLoad(a, tmpf.name)
        assert r == {'x': 1}
        os.unlink(tmpf.name)

    def test_pickle_dump(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        tmpf = tempfile.NamedTemporaryFile(suffix='.p', delete=False)
        tmpf.close()
        FloPyAgent.pickleDump(a, tmpf.name, {'y': 2})
        with open(tmpf.name, 'rb') as f:
            r = pickle.load(f)
        assert r == {'y': 2}
        os.unlink(tmpf.name)

    def test_init_arr(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        FloPyAgent.init_arr(a, [1.0, 2.0, 3.0], shape=(3,))

    def test_generate_path_prefixes(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.tempModelpth = '/tmp/model'
        a.zFill = 4
        a.geneticGeneration = 0
        FloPyAgent.generatePathPrefixes(a)
        assert 'test_gen0001' in a.tempModelPrefix
        assert 'test_gen0002' in a.tempNextModelPrefix

    def test_save_best_agent(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.wrkspc = tempfile.mkdtemp()
        a.modelpth = a.wrkspc
        a.agentMode = 'genetic'
        a.rewards = [10, 20, 30]
        a.bestAgentReward = 30.0
        a.geneticGeneration = 0
        with patch('flopyarcade.agent.load_model', return_value=MagicMock()) as lm:
            FloPyAgent.saveBestAgent(a, 'test')
            lm.assert_called_once()
        shutil.rmtree(a.wrkspc, ignore_errors=True)

    def test_create_nn_model_discrete(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.actionSpaceSize = 5
        a.observationsVector = np.array([1.0, 2.0, 3.0])
        with patch('flopyarcade.agent.Sequential', return_value=MagicMock()):
            with patch('flopyarcade.agent.Dense', return_value=MagicMock()):
                with patch('flopyarcade.agent.Activation', return_value=MagicMock()):
                    with patch('flopyarcade.agent.random_uniform', return_value=MagicMock()):
                        with patch('flopyarcade.agent.SeedGenerator', return_value=MagicMock()):
                            with patch('flopyarcade.agent.Adam', return_value=MagicMock()):
                                m = FloPyAgent.createNNModel(a, 'discrete', seed=42)
                                assert m is not None

    def test_create_nn_model_continuous(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.actionSpaceSize = 3
        a.observationsVector = np.array([1.0, 2.0, 3.0])
        with patch('flopyarcade.agent.Sequential', return_value=MagicMock()):
            with patch('flopyarcade.agent.Dense', return_value=MagicMock()):
                with patch('flopyarcade.agent.Activation', return_value=MagicMock()):
                    with patch('flopyarcade.agent.random_uniform', return_value=MagicMock()):
                        with patch('flopyarcade.agent.SeedGenerator', return_value=MagicMock()):
                            with patch('flopyarcade.agent.Adam', return_value=MagicMock()):
                                m = FloPyAgent.createNNModel(a, 'continuous', seed=42)
                                assert m is not None

    def test_train(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.replayMemory = []
        for i in range(16):
            a.replayMemory.append([np.random.rand(3), 0, np.random.rand(), np.random.rand(3), False])
        a.mainModel = MagicMock()
        a.targetModel = MagicMock()
        a.mainModel.predict.return_value = np.array([[1.0, 2.0, 3.0]] * 16)
        a.targetModel.predict.return_value = np.array([[1.1, 2.1, 3.1]] * 16)
        a.mainModel.fit = MagicMock()
        a.targetUpdateCount = 0
        a.hyParams['REPLAYMEMORYSIZEMIN'] = 10
        a.hyParams['MINIBATCHSIZE'] = 8
        a.hyParams['DISCOUNT'] = 0.99
        a.hyParams['UPDATEPREDICTIVEMODELEVERY'] = 100
        FloPyAgent.train(a, False, 1)
        a.mainModel.fit.assert_called_once()

    def test_train_insufficient_memory(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.replayMemory = [1, 2, 3]
        a.mainModel = MagicMock()
        a.targetModel = MagicMock()
        a.targetUpdateCount = 0
        a.hyParams['REPLAYMEMORYSIZEMIN'] = 10
        FloPyAgent.train(a, False, 1)
        a.mainModel.fit.assert_not_called()

    def test_take_actions_dqn(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.hyParams['NAGENTSTEPS'] = 3
        a.epsilon = 0.5
        a.gameReward = 0
        a.gameStep = 1
        a.mainModel = MagicMock()
        a.replayMemory = []
        env = MagicMock()
        env.actionType = 'discrete'
        env.observationsVectorNormalized = np.array([1.0, 2.0])
        env.step.return_value = (np.array([1.0]), 0.5, False, False, {})
        env.ENVTYPE = '1s-d'
        a.envSettings['RENDER'] = False
        with patch('flopyarcade.agent.argmax', return_value=1):
            with patch('flopyarcade.agent.random', return_value=0.9):
                with patch.object(FloPyAgent, 'train'):
                    FloPyAgent.takeActionsUpdateAndTrainDQN(a, env)

    def test_crossvalidate_dqn(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.hyParams['NGAMESCROSSVALIDATED'] = 1
        a.seedsCV = [42]
        a.mainModel = MagicMock()
        env = MagicMock()
        env.actionType = 'discrete'
        env.observationsVectorNormalized = np.array([1.0, 2.0])
        env.step.return_value = (np.array([1.0]), 0.5, True, False, {})
        env.ENVTYPE = '1s-d'
        env.initWithSolution = True
        a.envSettings['RENDER'] = False
        with patch('flopyarcade.agent.argmax', return_value=1):
            with patch.object(FloPyAgent, 'getqsGivenAgentModel', return_value=np.array([0.1, 0.5, 0.2, 0.1, 0.1])):
                FloPyAgent.crossvalidateDQN(a, env)

    def test_action_novelty_metric_discrete(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a1 = ['up', 'down', 'left']
        a2 = ['up', 'left', 'right']
        r = FloPyAgent.actionNoveltyMetric(a, a1, a2, 'discrete')
        assert 0 <= r <= 1

    def test_action_novelty_metric_continuous(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        a2 = np.array([[0.5, 0.6], [0.7, 0.8]])
        r = FloPyAgent.actionNoveltyMetric(a, a1, a2, 'continuous')
        assert 0 <= r <= 1

    def test_calculate_novelty_per_pair(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.actionType = 'discrete'
        actions1 = [['up', 'down'], ['left']]
        actions2 = [['up', 'right'], ['keep']]
        r = FloPyAgent.calculateNoveltyPerPair(a, [actions1, actions2])
        assert r >= 0

    def test_random_agent_genetic(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.tempModelpth = tempfile.mkdtemp()
        with patch('flopyarcade.agent.FloPyEnv') as FE:
            FE.return_value.getActionType.return_value = 'discrete'
            with patch.object(FloPyAgent, 'createNNModel', return_value=MagicMock()) as cnm:
                cnm.return_value.save = MagicMock()
                FloPyAgent.randomAgentGenetic(a, 0, 1)
        shutil.rmtree(a.tempModelpth, ignore_errors=True)

    def test_run_agents_genetic_single(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.hyParams['NAGENTS'] = 2
        a.mutationHistory = {
            'agent1': {'creationSeed': 1, 'mutationSeeds': []},
            'agent2': {'creationSeed': 2, 'mutationSeeds': []}
        }
        a.envSettings['KEEPMODELHISTORY'] = False
        a.envSettings['SEEDSRANDOM'] = False
        a.envSettings['ENVTYPE'] = '1s-d'
        a.envSettings['SURROGATESIMULATOR'] = None
        a.envSettings['RENDER'] = False
        env = MagicMock()
        env.actionType = 'discrete'
        env.observationsVectorNormalized = np.array([1.0, 2.0])
        env.step.return_value = (np.array([1.0]), 0.5, True, False, {})
        env.ENVTYPE = '1s-d'
        a.env = env
        a.geneticGeneration = 0
        a.currentGame = 1
        with patch.object(FloPyAgent, 'createNNModel', return_value=MagicMock()):
            with patch.object(FloPyAgent, 'mutateGenetic', side_effect=lambda agent, s: agent):
                with patch.object(FloPyAgent, 'getqsGivenAgentModel', return_value=np.array([0.1, 0.5, 0.2, 0.1, 0.1])):
                    with patch('flopyarcade.agent.argmax', return_value=1):
                        with patch.object(FloPyAgent, 'pickleDump'):
                            with patch.object(FloPyAgent, 'pickleLoad', return_value={'trajectories': [[]], 'actions': [[]], 'rewards': [[]], 'wellCoords': [[]]}):
                                r = FloPyAgent.runAgentsGeneticSingleRun(a, 0)
                                assert r == 0.5

    def test_return_children_genetic(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        a.hyParams['NAGENTS'] = 2
        a.mutationHistory = {
            'agent1': {'creationSeed': 1, 'mutationSeeds': [2]},
            'agent2': {'creationSeed': 2, 'mutationSeeds': []}
        }
        a.rewards = [10, 20]
        a.rereturnChildrenGenetic = False
        a.searchNovelty = False
        a.geneticGeneration = 0
        a.envSettings['KEEPMODELHISTORY'] = False
        a.envSettings['NAGENTSPARALLEL'] = 1
        a.maxTasksPerWorkerMutate = 100
        with patch.object(FloPyAgent, 'pickleDump'):
            with patch.object(FloPyAgent, 'createNNModel', return_value=MagicMock()):
                with patch.object(FloPyAgent, 'mutateGenetic', side_effect=lambda agent, s: agent):
                    with patch.object(FloPyAgent, 'multiprocessChunks', return_value=[{'creationSeed': 1, 'mutationSeeds': [2]}]):
                        with patch.object(FloPyAgent, 'yieldChunks', side_effect=lambda x, n: [list(x)]):
                            with patch('flopyarcade.agent.argmax', return_value=1):
                                with patch('flopyarcade.agent.randint', return_value=0):
                                    FloPyAgent.returnChildrenGenetic(a, [1, 0])

    def test_gpu_memory_growth(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        with patch('tensorflow.config.list_physical_devices', return_value=[]):
            FloPyAgent.GPUAllowMemoryGrowth(a)

    def test_suppress_tf_warnings(self):
        from flopyarcade.agent import FloPyAgent
        a = make_agent()
        FloPyAgent.suppressTensorFlowWarnings(a)
