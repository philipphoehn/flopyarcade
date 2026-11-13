"""Tests to increase agent.py coverage from 50% to 85%+.
All heavy dependencies (tensorflow, flopy, pathos, gymnasium) are mocked.
No actual training or MODFLOW is run.
"""
import os
import pickle
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from collections import deque, defaultdict
from itertools import count
from math import ceil, floor

from flopyarcade.agent import FloPyAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(**overrides):
    """Create a FloPyAgent instance bypassing __init__ and set attributes."""
    a = object.__new__(FloPyAgent)
    a.wrkspc = tempfile.mkdtemp()
    a.modelpth = os.path.join(a.wrkspc, 'models')
    os.makedirs(a.modelpth, exist_ok=True)
    a.actionSpace = ['keep', 'left', 'right', 'up', 'down']
    a.actionSpaceSize = len(a.actionSpace)
    a.actionType = 'discrete'
    a.agentMode = 'genetic'
    a.hyParams = {
        'NAGENTS': 10,
        'NNOVELTYELITES': 3,
        'NNOVELTYAGENTS': 2,
        'NAGENTELITES': 5,
        'NGENERATIONS': 3,
        'NGAMESAVERAGED': 2,
        'NAGENTSTEPS': 10,
        'NHIDDENNODES': [16, 8],
        'NNTYPE': 'perceptron',
        'BATCHNORMALIZATION': False,
        'HIDDENACTIVATIONS': ['relu', 'relu'],
        'DROPOUTS': [0.0, 0.0],
        'MUTATIONPROBABILITY': 0.1,
        'MUTATIONPOWER': 0.1,
        'NOVELTYSEARCH': True,
        'NNOVELTYNEIGHBORS': 4,
        'ADDNOVELTYEVERY': 2,
        'REPLAYMEMORYSIZE': 100,
        'REPLAYMEMORYSIZEMIN': 4,
        'MINIBATCHSIZE': 4,
        'EPSILONINITIAL': 1.0,
        'EPSILONMIN': 0.01,
        'EPSILONDECAY': 0.95,
        'DISCOUNT': 0.99,
        'UPDATEPREDICTIVEMODELEVERY': 10,
        'NGAMESCROSSVALIDATED': 2,
        'CROSSVALIDATEEVERY': 5,
        'NGAMES': 10,
        'ARCHITECTUREVARY': False,
    }
    a.envSettings = {
        'ENVTYPE': '0s-d',
        'MODELNAME': 'test_model',
        'NAGENTSPARALLEL': 1,
        'RESUME': False,
        'KEEPMODELHISTORY': False,
        'BESTAGENTANIMATION': False,
        'RENDER': False,
        'RENDEREVERY': 5,
        'SEEDSRANDOM': False,
        'SURROGATESIMULATOR': None,
        'PATHMF2005': '/tmp/mf2005',
        'PATHMP6': '/tmp/mp6',
        'NLAY': 1,
        'NROW': 4,
        'NCOL': 4,
        'REWARDMINTOSAVE': 0.5,
        'SEEDAGENT': 42,
        'SEEDENV': 123,
    }
    a.maxTasksPerWorker = 1000
    a.maxTasksPerWorkerMutate = 1000
    a.maxTasksPerWorkerNoveltySearch = 10000
    a.zFill = 6
    a.SEED = 42
    a.tempModelpth = os.path.join(a.wrkspc, 'temp', 'test_model')
    os.makedirs(a.tempModelpth, exist_ok=True)
    a.tempNoveltypth = os.path.join(a.tempModelpth, 'novelties')
    os.makedirs(a.tempNoveltypth, exist_ok=True)
    a.tempModelPrefix = os.path.join(a.tempModelpth, 'test_model_gen000001')
    a.tempNextModelPrefix = os.path.join(a.tempModelpth, 'test_model_gen000002')
    a.tempPrevModelPrefix = os.path.join(a.tempModelpth, 'test_model_gen000000')
    a.geneticGeneration = 0
    a.noveltyArchive = {}
    a.noveltyItemCount = 0
    a.agentsUnique = []
    a.agentsUniqueIDs = []
    a.agentsDuplicate = []
    a.agentsDuplicateID = []
    a.novelties = []
    a.noveltyFilenames = []
    a.mutationHistory = {}
    a.searchNovelty = False
    a.rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    a.currentGame = 1
    a.rereturnChildrenGenetic = False
    a.cores = 1
    a.pid = 'abc123def456'
    a.candidateParentIdxs = [0, 1, 2]
    a.candidateNoveltyParentIdxs = [0, 1]
    a.neighborLimitReached = False
    a.mutationSeeds = [[42, 43, 44, 45, 46, 47, 48, 49, 50, 51] for _ in range(3)]
    a.gamesSeeds = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for _ in range(3)]
    a.flagSkipGeneration = False
    a.bestRewards = []
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_init_random_mode(self):
        """Test __init__ with mode='random'."""
        with patch('flopyarcade.agent.FloPyEnv') as MockEnv:
            with patch('flopyarcade.agent.cpu_count', return_value=4):
                a = FloPyAgent(
                    observationsVector=np.zeros((4, 4, 1)),
                    actionSpace=['keep', 'left', 'right'],
                    hyParams={'NAGENTS': 4, 'NNOVELTYELITES': 2, 'NNOVELTYAGENTS': 1,
                              'NAGENTELITES': 2, 'NGENERATIONS': 1, 'NGAMESAVERAGED': 1,
                              'NAGENTSTEPS': 5, 'NHIDDENNODES': [8], 'NNTYPE': 'perceptron',
                              'BATCHNORMALIZATION': False, 'HIDDENACTIVATIONS': ['relu'],
                              'MUTATIONPROBABILITY': 0.1, 'MUTATIONPOWER': 0.1,
                              'NOVELTYSEARCH': False, 'NNOVELTYNEIGHBORS': 2,
                              'ADDNOVELTYEVERY': 1},
                    envSettings={'ENVTYPE': '0s-d', 'MODELNAME': 'init_test',
                                 'NAGENTSPARALLEL': 1, 'RESUME': False,
                                 'KEEPMODELHISTORY': False, 'BESTAGENTANIMATION': False,
                                 'RENDER': False, 'RENDEREVERY': 5, 'SEEDSRANDOM': False,
                                 'SURROGATESIMULATOR': None, 'PATHMF2005': '/tmp',
                                 'PATHMP6': '/tmp', 'NLAY': 1, 'NROW': 2, 'NCOL': 2,
                                 'REWARDMINTOSAVE': 0.5, 'SEEDAGENT': 42, 'SEEDENV': 123},
                    mode='random',
                )
        assert a.agentMode == 'random'
        assert a.actionSpaceSize == 3

    def test_init_dqn_mode(self):
        """Test __init__ with mode='DQN' triggers initializeDQNAgent."""
        with patch('flopyarcade.agent.FloPyEnv') as MockEnv:
            MockEnv.return_value.getActionType.return_value = 'discrete'
            with patch.object(FloPyAgent, 'createNNModel') as mock_nn:
                mock_model = MagicMock()
                mock_model.get_weights.return_value = [np.zeros(4)]
                mock_nn.return_value = mock_model
                with patch('flopyarcade.agent.numpySeed'), \
                     patch('flopyarcade.agent.randomSeed'), \
                     patch('flopyarcade.agent.set_random_seed'):
                    a = FloPyAgent(
                        observationsVector=np.zeros((4,)),
                        actionSpace=['keep', 'left'],
                        hyParams={'NAGENTS': 2, 'NNOVELTYELITES': 1, 'NNOVELTYAGENTS': 1,
                                  'NAGENTELITES': 1, 'NGENERATIONS': 1, 'NGAMESAVERAGED': 1,
                                  'NAGENTSTEPS': 5, 'NHIDDENNODES': [8], 'NNTYPE': 'perceptron',
                                  'BATCHNORMALIZATION': False, 'HIDDENACTIVATIONS': ['relu'],
                                  'MUTATIONPROBABILITY': 0.1, 'MUTATIONPOWER': 0.1,
                                  'NOVELTYSEARCH': False, 'NNOVELTYNEIGHBORS': 2,
                                  'ADDNOVELTYEVERY': 1,
                                  'REPLAYMEMORYSIZE': 50, 'REPLAYMEMORYSIZEMIN': 4,
                                  'MINIBATCHSIZE': 4, 'EPSILONINITIAL': 1.0,
                                  'EPSILONMIN': 0.01, 'EPSILONDECAY': 0.95,
                                  'DISCOUNT': 0.99, 'UPDATEPREDICTIVEMODELEVERY': 5},
                        envSettings={'ENVTYPE': '0s-d', 'MODELNAME': 'dqn_test',
                                     'NAGENTSPARALLEL': 1, 'RESUME': False,
                                     'KEEPMODELHISTORY': False, 'BESTAGENTANIMATION': False,
                                     'RENDER': False, 'RENDEREVERY': 5, 'SEEDSRANDOM': False,
                                     'SURROGATESIMULATOR': None, 'PATHMF2005': '/tmp',
                                     'PATHMP6': '/tmp', 'NLAY': 1, 'NROW': 2, 'NCOL': 2,
                                     'REWARDMINTOSAVE': 0.5, 'SEEDAGENT': 42, 'SEEDENV': 123},
                        mode='DQN',
                    )
        assert a.agentMode == 'DQN'
        assert a.mainModel is not None
        assert a.targetModel is not None
        assert isinstance(a.replayMemory, deque)


# ---------------------------------------------------------------------------
# getActionType
# ---------------------------------------------------------------------------

class TestGetActionType:
    def test_discrete(self):
        a = make_agent()
        assert a.getActionType('0s-d') == 'discrete'

    def test_continuous(self):
        a = make_agent()
        assert a.getActionType('0s-c') == 'continuous'

    def test_unknown(self):
        a = make_agent()
        with pytest.raises(SystemExit):
            a.getActionType('unknown-type')


# ---------------------------------------------------------------------------
# _decay_epsilon
# ---------------------------------------------------------------------------

class TestDecayEpsilon:
    def test_decay_above_min(self):
        a = make_agent(epsilon=0.5)
        a._decay_epsilon()
        assert a.epsilon == pytest.approx(max(0.01, 0.5 * 0.95))

    def test_decay_at_min(self):
        a = make_agent(epsilon=0.01)
        a._decay_epsilon()
        assert a.epsilon == 0.01  # epsilon equals min, guard prevents decay

    def test_decay_below_min(self):
        a = make_agent(epsilon=0.005)
        a._decay_epsilon()
        assert a.epsilon == 0.005  # guard prevents decay when below min


# ---------------------------------------------------------------------------
# _validate_hyperparams
# ---------------------------------------------------------------------------

class TestValidateHyperparams:
    def test_valid(self):
        a = make_agent()
        a.hyParams['NAGENTS'] = 10
        a.hyParams['NNOVELTYELITES'] = 3
        a._validate_hyperparams()  # should not raise

    def test_invalid(self):
        a = make_agent()
        a.hyParams['NAGENTS'] = 2
        a.hyParams['NNOVELTYELITES'] = 3
        with pytest.raises(ValueError, match='novelty elites'):
            a._validate_hyperparams()


# ---------------------------------------------------------------------------
# _initialize_environment
# ---------------------------------------------------------------------------

class TestInitializeEnvironment:
    def test_discrete(self):
        a = make_agent()
        env = MagicMock()
        env.actionType = 'discrete'
        a.hyParams['NGAMESAVERAGED'] = 2
        a._initialize_environment(env, True)
        assert a.searchNovelty is True
        assert a.noveltyArchive == {}
        assert a.noveltyItemCount == 0
        assert isinstance(a.actionsUniqueIDMapping, defaultdict)

    def test_continuous(self):
        a = make_agent()
        env = MagicMock()
        env.actionType = 'continuous'
        a._initialize_environment(env, False)
        assert a.searchNovelty is False


# ---------------------------------------------------------------------------
# _initialize_parallel_settings
# ---------------------------------------------------------------------------

class TestInitializeParallelSettings:
    def test_with_cores(self):
        a = make_agent()
        a.envSettings['NAGENTSPARALLEL'] = 4
        a._initialize_parallel_settings()
        assert a.cores == 4

    def test_default_cores(self):
        a = make_agent()
        a.envSettings['NAGENTSPARALLEL'] = None
        a._initialize_parallel_settings()
        assert a.cores is None


# ---------------------------------------------------------------------------
# _generate_unique_pid
# ---------------------------------------------------------------------------

class TestGenerateUniquePid:
    def test_generates_pid(self):
        a = make_agent()
        a._generate_unique_pid()
        assert len(a.pid) == 36  # uuid4 string is 36 chars (32 hex + 4 dashes)


# ---------------------------------------------------------------------------
# _log_generation_rewards
# ---------------------------------------------------------------------------

class TestLogGenerationRewards:
    def test_seeds_random(self):
        a = make_agent()
        a.envSettings['SEEDSRANDOM'] = True
        a.rewards = [1.0, 2.0, 3.0]
        a._log_generation_rewards()
        assert a.bestRewards == [3.0]

    def test_no_seeds_random(self):
        a = make_agent()
        a.envSettings['SEEDSRANDOM'] = False
        a.rewards = [1.0, 2.0, 3.0]
        a._log_generation_rewards()  # should not raise


# ---------------------------------------------------------------------------
# _sort_and_save_top_agents
# ---------------------------------------------------------------------------

class TestSortAndSaveTopAgents:
    def test_sorts_correctly(self):
        a = make_agent()
        a.rewards = [3.0, 1.0, 4.0, 1.0, 5.0, 2.0, 3.0, 2.0, 4.0, 5.0]
        with patch.object(FloPyAgent, 'pickleDump'):
            result = a._sort_and_save_top_agents()
        assert len(result) == a.hyParams['NAGENTELITES']
        # sorted descending, top 5
        assert result[0] in [4, 9]  # indices with reward 5.0


# ---------------------------------------------------------------------------
# _update_discrete_uniques
# ---------------------------------------------------------------------------

class TestUpdateDiscreteUniques:
    def test_new_unique(self):
        a = make_agent(noveltyItemCount=1)
        a.agentsUnique = []
        a.agentsUniqueIDs = []
        a.agentsDuplicate = []
        a._update_discrete_uniques(99)
        assert 99 in a.agentsUniqueIDs

    def test_duplicate(self):
        a = make_agent(noveltyItemCount=1)
        a.agentsUnique = [0]
        a.agentsUniqueIDs = [5]
        a.agentsDuplicate = []
        a._update_discrete_uniques(5)
        assert 1 in a.agentsDuplicate


# ---------------------------------------------------------------------------
# _reset_continuous_uniques
# ---------------------------------------------------------------------------

class TestResetContinuousUniques:
    def test_resets(self):
        a = make_agent(noveltyItemCount=3)
        a.noveltyArchive = {
            'agent1': {}, 'agent2': {}, 'agent3': {}, 'agent4': {}
        }
        a._reset_continuous_uniques()
        assert a.agentsUnique == [0, 1, 2, 3]
        assert a.agentsUniqueIDs == []  # source never appends to agentsUniqueIDs
        assert a.agentsDuplicate == []


# ---------------------------------------------------------------------------
# _print_novelty_stats
# ---------------------------------------------------------------------------

class TestPrintNoveltyStats:
    def test_below_neighbor(self, capsys):
        a = make_agent()
        a.agentsUnique = [0, 1, 2]
        a.agentsDuplicate = [3]
        a.hyParams['NNOVELTYNEIGHBORS'] = 10
        a.noveltyItemCount = 5
        a._print_novelty_stats()
        out = capsys.readouterr().out
        assert '3 unique agents' in out

    def test_above_neighbor(self, capsys):
        a = make_agent()
        a.agentsUnique = [0, 1, 2, 3, 4]
        a.agentsDuplicate = [5]
        a.hyParams['NNOVELTYNEIGHBORS'] = 4
        a.noveltyItemCount = 10
        a._print_novelty_stats()
        out = capsys.readouterr().out
        assert 'neighbor level reached' in out


# ---------------------------------------------------------------------------
# _build_novelty_args
# ---------------------------------------------------------------------------

class TestBuildNoveltyArgs:
    def test_below_limit(self):
        a = make_agent()
        a.agentsUnique = [0, 1, 2]
        a.hyParams['NNOVELTYNEIGHBORS'] = 10
        a.neighborLimitReached = False
        args, flags = a._build_novelty_args()
        assert len(args) == 3
        assert all(flags)

    def test_above_limit(self):
        a = make_agent()
        a.agentsUnique = [0, 1, 2, 3, 4]
        a.hyParams['NNOVELTYNEIGHBORS'] = 4
        a.neighborLimitReached = True
        args, flags = a._build_novelty_args()
        assert len(args) == 5


# ---------------------------------------------------------------------------
# perform_novelty_search
# ---------------------------------------------------------------------------

class TestPerformNoveltySearch:
    def test_not_searching(self):
        a = make_agent(searchNovelty=False)
        a.perform_novelty_search(MagicMock())  # should return early

    def test_full_flow(self):
        a = make_agent(searchNovelty=True)
        a.agentsUnique = [0]
        a.noveltyItemCount = 1
        a.noveltyArchive = {'agent1': {'novelty': 0.5, 'modelFile': '/tmp/f.keras'}}
        a.hyParams['NNOVELTYNEIGHBORS'] = 10
        a.neighborLimitReached = False
        env = MagicMock()
        env.actionType = 'discrete'
        with patch.object(a, '_prepare_novelty_archive'), \
             patch.object(a, '_build_novelty_args', return_value=([0], [True])), \
             patch.object(a, '_run_novelty_search', return_value=[0.7]), \
             patch.object(a, '_update_agent_novelties'), \
             patch.object(a, '_save_novelties'):
            a.perform_novelty_search(env)


# ---------------------------------------------------------------------------
# _update_agent_novelties
# ---------------------------------------------------------------------------

class TestUpdateAgentNovelties:
    def test_update(self):
        a = make_agent(noveltyItemCount=3)
        a.agentsUnique = [0, 1]
        a.agentsDuplicate = [2]
        a.noveltyArchive = {
            'agent1': {'actionsUniqueID': 0},
            'agent2': {'actionsUniqueID': 1},
            'agent3': {'actionsUniqueID': 0},
        }
        a._update_agent_novelties([0.5, 0.3], [True, True])
        assert a.noveltyArchive['agent1']['novelty'] == 0.5
        assert a.noveltyArchive['agent2']['novelty'] == 0.3
        assert a.noveltyArchive['agent3']['novelty'] == 0.5  # maps to unique 0


# ---------------------------------------------------------------------------
# _prepare_novelty_archive
# ---------------------------------------------------------------------------

class TestPrepareNoveltyArchive:
    def test_discrete(self):
        a = make_agent()
        a.noveltyItemCount = 0
        a.hyParams['NAGENTS'] = 2
        a.actionsUniqueIDMapping = defaultdict(count().__next__)
        env = MagicMock()
        env.actionType = 'discrete'
        with patch.object(a, 'pickleLoad') as mock_load:
            mock_load.return_value = {'actions': [['keep', 'left'], ['right', 'up']]}
            a._prepare_novelty_archive(env)
        assert a.noveltyItemCount == 2
        assert 'agent1' in a.noveltyArchive

    def test_continuous(self):
        a = make_agent()
        a.noveltyItemCount = 0
        a.hyParams['NAGENTS'] = 1
        env = MagicMock()
        env.actionType = 'continuous'
        a._prepare_novelty_archive(env)
        assert a.noveltyItemCount == 1


# ---------------------------------------------------------------------------
# _save_novelties
# ---------------------------------------------------------------------------

class TestSaveNovelties:
    def test_save(self):
        a = make_agent(noveltyItemCount=2)
        a.noveltyArchive = {
            'agent1': {'novelty': 0.5, 'modelFile': '/tmp/a.keras'},
            'agent2': {'novelty': 0.3, 'modelFile': '/tmp/b.keras'},
        }
        with patch('flopyarcade.archive_hdf5.NoveltyArchiveHDF5') as MockArch:
            a._save_novelties()
        assert a.novelties == [0.5, 0.3]
        assert a.noveltyFilenames == ['/tmp/a.keras', '/tmp/b.keras']


# ---------------------------------------------------------------------------
# _load_novelty_archive
# ---------------------------------------------------------------------------

class TestLoadNoveltyArchive:
    def test_h5_exists(self):
        a = make_agent()
        a.tempPrevModelPrefix = '/tmp/test_gen'
        a.envSettings['KEEPMODELHISTORY'] = True
        archive_path = a.tempPrevModelPrefix + '_noveltyArchive.h5'
        with patch('flopyarcade.agent.exists', return_value=True), \
             patch('flopyarcade.archive_hdf5.NoveltyArchiveHDF5') as MockArch:
            MockArch.return_value.load.return_value = {'agent1': {}}
            a._load_novelty_archive()
        assert a.noveltyArchive == {'agent1': {}}

    def test_fallback_pickle(self):
        a = make_agent()
        a.tempPrevModelPrefix = '/tmp/test_gen'
        a.envSettings['KEEPMODELHISTORY'] = True
        with patch('os.path.exists', side_effect=[False, False, False]), \
             patch.object(a, 'pickleLoad', return_value={'agent1': {}}):
            a._load_novelty_archive()
        assert a.noveltyArchive == {'agent1': {}}


# ---------------------------------------------------------------------------
# countParameters
# ---------------------------------------------------------------------------

class TestCountParameters:
    def test_count(self):
        a = make_agent()
        agent = MagicMock()
        w1 = np.ones((4, 8))
        w2 = np.ones((8, 4))
        agent.get_weights.return_value = [w1, w2]
        assert a.countParameters(agent) == 32 + 32


# ---------------------------------------------------------------------------
# mutateDecision / mutateDecisions
# ---------------------------------------------------------------------------

class TestMutateDecision:
    def test_mutate_decision_high_prob(self):
        a = make_agent()
        with patch('flopyarcade.agent.random', return_value=0.05):
            assert a.mutateDecision(0.1) is True

    def test_mutate_decision_low_prob(self):
        a = make_agent()
        with patch('flopyarcade.agent.random', return_value=0.95):
            assert a.mutateDecision(0.1) is False

    def test_mutate_decisions(self):
        a = make_agent()
        with patch('flopyarcade.agent.random', return_value=np.array([0.05, 0.5, 0.95])):
            result = a.mutateDecisions(0.1, (3,))
        assert result[0] == True
        assert result[1] == False
        assert result[2] == False


# ---------------------------------------------------------------------------
# getqsGivenAgentModel
# ---------------------------------------------------------------------------

class TestGetqsGivenAgentModel:
    def test_discrete(self):
        a = make_agent()
        agent = MagicMock()
        agent.predict_on_batch.return_value = [np.array([1.0, 2.0, 3.0, 0.5, 0.1])]
        state = np.array([0.1, 0.2, 0.3, 0.4])
        result = a.getqsGivenAgentModel(agent, state)
        assert np.allclose(result, [1.0, 2.0, 3.0, 0.5, 0.1])


# ---------------------------------------------------------------------------
# checkParameterMean
# ---------------------------------------------------------------------------

class TestCheckParameterMean:
    def test_mean(self):
        a = make_agent()
        agent = MagicMock()
        w1 = np.ones((4, 4)) * 2.0
        w2 = np.ones((4, 4)) * 4.0
        agent.get_weights.return_value = [w1, w2]
        assert a.checkParameterMean(agent) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# loadModelFromMutationHistory
# ---------------------------------------------------------------------------

class TestLoadModelFromMutationHistory:
    def test_no_mutations(self):
        a = make_agent()
        with patch.object(a, 'createNNModel', return_value=MagicMock()) as mock_nn:
            result = a.loadModelFromMutationHistory(42, [])
            mock_nn.assert_called_once_with('discrete', seed=42)

    def test_with_mutations(self):
        a = make_agent()
        mock_agent = MagicMock()
        with patch.object(a, 'createNNModel', return_value=mock_agent), \
             patch.object(a, 'mutateGenetic', return_value=mock_agent):
            result = a.loadModelFromMutationHistory(42, [1, 2, 3])
            assert a.mutateGenetic.call_count == 3


# ---------------------------------------------------------------------------
# updateMutationHistory
# ---------------------------------------------------------------------------

class TestUpdateMutationHistory:
    def test_basic(self):
        a = make_agent()
        mh = {}
        mh = a.updateMutationHistory(mh, 1, 42)
        assert mh['agent1']['creationSeed'] == 42
        assert mh['agent1']['mutationSeeds'] == []

    def test_with_seed(self):
        a = make_agent()
        mh = {}
        mh = a.updateMutationHistory(mh, 1, 42, mutationSeeds=[1], mutationSeed=99)
        assert mh['agent1']['mutationSeeds'] == [1, 99]


# ---------------------------------------------------------------------------
# createMutationRecord
# ---------------------------------------------------------------------------

class TestCreateMutationRecord:
    def test_basic(self):
        a = make_agent()
        record = a.createMutationRecord(42)
        assert record['creationSeed'] == 42
        assert record['mutationSeeds'] == []

    def test_with_seed(self):
        a = make_agent()
        record = a.createMutationRecord(42, mutationSeeds=[1, 2], mutationSeed=99)
        assert record['mutationSeeds'] == [1, 2, 99]


# ---------------------------------------------------------------------------
# updateReplayMemory / updateReplayMemoryZeroReward
# ---------------------------------------------------------------------------

class TestUpdateReplayMemory:
    def test_append(self):
        a = make_agent()
        a.replayMemory = deque(maxlen=100)
        a.updateReplayMemory([1, 2, 3, 4, False])
        assert len(a.replayMemory) == 1

    def test_zero_reward(self):
        a = make_agent()
        a.replayMemory = deque(maxlen=100)
        a.replayMemory.append([1, 2, 5.0, 4, False])
        a.replayMemory.append([1, 2, 5.0, 4, False])
        a.updateReplayMemoryZeroReward(2)
        assert a.replayMemory[0][2] == 0.0
        assert a.replayMemory[1][2] == 0.0


# ---------------------------------------------------------------------------
# getAction (all modes)
# ---------------------------------------------------------------------------

class TestGetAction:
    def test_random_discrete(self):
        a = make_agent()
        action = a.getAction(mode='random', actionType='discrete')
        assert action in a.actionSpace

    def test_manual_discrete_valid(self):
        a = make_agent()
        action = a.getAction(mode='manual', keyPressed='left', actionType='discrete')
        assert action == 'left'

    def test_manual_discrete_invalid(self):
        a = make_agent()
        action = a.getAction(mode='manual', keyPressed='bogus', actionType='discrete')
        assert action == 'keep'

    def test_model_discrete(self):
        a = make_agent()
        agent = MagicMock()
        agent.predict_on_batch.return_value = [np.array([0.1, 0.9, 0.1, 0.1, 0.1])]
        state = np.array([0.5, 0.5])
        action = a.getAction(mode='model', agent=agent, state=state, actionType='discrete')
        assert action == 'left'  # argmax index 1

    def test_random_continuous(self):
        a = make_agent(actionSpace=[0, 1, 2], actionSpaceSize=3)
        action = a.getAction(mode='random', actionType='continuous')
        assert len(action) == 3
        assert all(0.0 <= x <= 1.0 for x in action)

    def test_manual_continuous_quit(self):
        a = make_agent()
        with pytest.raises(SystemExit):
            a.getAction(mode='manual', actionType='continuous')

    def test_model_continuous(self):
        a = make_agent(actionSpace=[0, 1, 2], actionSpaceSize=3)
        agent = MagicMock()
        agent.predict_on_batch.return_value = [np.array([0.3, 0.6, 0.1])]
        state = np.array([0.5, 0.5])
        action = a.getAction(mode='model', agent=agent, state=state, actionType='continuous')
        assert np.allclose(action, [0.3, 0.6, 0.1])


# ---------------------------------------------------------------------------
# loadActions
# ---------------------------------------------------------------------------

class TestLoadActions:
    def test_discrete(self):
        a = make_agent()
        a.hyParams['NGAMESAVERAGED'] = 2
        a.agentsUnique = [0, 1]
        a.noveltyArchive = {
            'agent1': {'resultsFile': '/tmp/r1.p'},
            'agent2': {'resultsFile': '/tmp/r2.p'},
        }
        with patch.object(a, 'pickleLoad') as mock_load:
            mock_load.return_value = {'actions': [['keep', 'left'], ['right', 'up']]}
            result = a.loadActions(agents=[0, 1])
        assert result is not None

    def test_continuous(self):
        a = make_agent(actionType='continuous')
        a.hyParams['NGAMESAVERAGED'] = 1
        a.agentsUnique = [0]
        a.noveltyArchive = {'agent1': {'resultsFile': '/tmp/r1.p'}}
        with patch.object(a, 'pickleLoad') as mock_load:
            mock_load.return_value = {'actions': [[0.1, 0.2, 0.3]]}
            result = a.loadActions(agents=[0])
        assert result is not None


# ---------------------------------------------------------------------------
# multiprocessChunks
# ---------------------------------------------------------------------------

class TestMultiprocessChunks:
    def test_basic(self):
        a = make_agent()
        with patch('flopyarcade.agent.Pool') as MockPool:
            mock_p = MagicMock()
            MockPool.return_value = mock_p
            mock_p.map_async.return_value.get.return_value = [1, 2, 3]
            result = a.multiprocessChunks(lambda x: x + 1, [0, 1, 2])
            assert result == [1, 2, 3]

    def test_async_false(self):
        a = make_agent()
        with patch('flopyarcade.agent.Pool') as MockPool:
            mock_p = MagicMock()
            MockPool.return_value = mock_p
            mock_p.map.return_value = [4, 5]
            result = a.multiprocessChunks(lambda x: x + 4, [0, 1], async_=False)
            assert result == [4, 5]


# ---------------------------------------------------------------------------
# yieldChunks
# ---------------------------------------------------------------------------

class TestYieldChunks:
    def test_even_split(self):
        a = make_agent()
        chunks = list(a.yieldChunks([1, 2, 3, 4, 5, 6], 3))
        assert chunks == [[1, 2, 3], [4, 5, 6]]

    def test_uneven_split(self):
        a = make_agent()
        chunks = list(a.yieldChunks([1, 2, 3, 4, 5], 3))
        assert chunks == [[1, 2, 3], [4, 5]]

    def test_single_chunk(self):
        a = make_agent()
        chunks = list(a.yieldChunks([1, 2, 3], 10))
        assert chunks == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# pickleLoad / pickleDump
# ---------------------------------------------------------------------------

class TestPickleLoad:
    def test_plain(self):
        a = make_agent()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.p') as f:
            pickle.dump({'key': 'value'}, f)
            path = f.name
        try:
            result = a.pickleLoad(path)
            assert result == {'key': 'value'}
        finally:
            os.unlink(path)

    def test_lz4(self):
        a = make_agent()
        with patch('flopyarcade.agent.joblibLoad', return_value={'x': 1}) as mock_jl:
            result = a.pickleLoad('/tmp/fake.p', compressed='lz4')
            assert result == {'x': 1}
            mock_jl.assert_called_once()

    def test_lzma(self):
        a = make_agent()
        with patch('flopyarcade.agent.joblibLoad', return_value={'x': 2}) as mock_jl:
            result = a.pickleLoad('/tmp/fake.p', compressed='lzma')
            assert result == {'x': 2}

    def test_invalid_compression(self):
        a = make_agent()
        with pytest.raises(UnboundLocalError):
            a.pickleLoad('/tmp/fake.p', compressed='bzip9')


class TestPickleDump:
    def test_plain(self):
        a = make_agent()
        path = os.path.join(tempfile.mkdtemp(), 'test.p')
        a.pickleDump(path, {'data': [1, 2, 3]})
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        assert loaded == {'data': [1, 2, 3]}
        os.unlink(path)

    def test_lz4(self):
        a = make_agent()
        with patch('flopyarcade.agent.joblibDump') as mock_jd:
            a.pickleDump('/tmp/fake.p', {'x': 1}, compress='lz4')
            mock_jd.assert_called_once()

    def test_lzma(self):
        a = make_agent()
        with patch('flopyarcade.agent.joblibDump') as mock_jd:
            a.pickleDump('/tmp/fake.p', {'x': 1}, compress='lzma')
            mock_jd.assert_called_once()

    def test_invalid(self):
        a = make_agent()
        a.pickleDump('/tmp/fake.p', {'x': 1}, compress='bzip9')


# ---------------------------------------------------------------------------
# init_arr
# ---------------------------------------------------------------------------

class TestInitArr:
    def test_init(self):
        a = make_agent()
        import flopyarcade.agent as agent_mod
        a.init_arr([1.0, 2.0, 3.0, 4.0], shape=(2, 2))
        assert 'arr' in agent_mod.__dict__
        assert agent_mod.__dict__['arr'].shape == (2, 2)


# ---------------------------------------------------------------------------
# generatePathPrefixes
# ---------------------------------------------------------------------------

class TestGeneratePathPrefixes:
    def test_generates(self):
        a = make_agent(geneticGeneration=0)
        a.generatePathPrefixes()
        assert 'gen000001' in a.tempModelPrefix
        assert 'gen000002' in a.tempNextModelPrefix
        assert 'gen000000' in a.tempPrevModelPrefix


# ---------------------------------------------------------------------------
# saveBestAgent
# ---------------------------------------------------------------------------

class TestSaveBestAgent:
    def test_save(self):
        a = make_agent(geneticGeneration=0, bestAgentReward=42.5)
        with patch('flopyarcade.agent.load_model') as mock_lm:
            mock_model = MagicMock()
            mock_lm.return_value = mock_model
            a.saveBestAgent('test_model')
            assert 'test_model_gen000001' in a.bestAgentFileName
            mock_model.save.assert_called_once()


# ---------------------------------------------------------------------------
# saveBestAgentAnimation
# ---------------------------------------------------------------------------

class TestSaveBestAgentAnimation:
    def test_save(self):
        a = make_agent(currentGame=1)
        env = MagicMock()
        env.ENVTYPE = '0s-d'
        env.nLay = 1
        env.nRow = 4
        env.nCol = 4
        with patch('flopyarcade.agent.FloPyArcade') as MockArcade:
            a.saveBestAgentAnimation(env, 'best_agent', 'gen_count', 'model')
            MockArcade.assert_called_once()


# ---------------------------------------------------------------------------
# createNNModel
# ---------------------------------------------------------------------------

class TestCreateNNModel:
    def test_discrete_perceptron(self):
        a = make_agent()
        a.observationsVector = np.zeros((4, 4, 1))
        with patch('flopyarcade.agent.Sequential') as MockSeq, \
             patch('flopyarcade.agent.Dense') as MockDense, \
             patch('flopyarcade.agent.Activation') as MockAct, \
             patch('flopyarcade.agent.random_uniform', return_value='init'), \
             patch('flopyarcade.agent.SeedGenerator', return_value='seed_gen'), \
             patch('flopyarcade.agent.numpySeed'), \
             patch('flopyarcade.agent.Adam', return_value='optimizer'), \
             patch('flopyarcade.agent.randint', return_value=16):
            model = a.createNNModel('discrete', seed=42)
            assert model is not None
            model.compile.assert_called_once()

    def test_continuous(self):
        a = make_agent()
        a.observationsVector = np.zeros((4, 4, 1))
        with patch('flopyarcade.agent.Sequential') as MockSeq, \
             patch('flopyarcade.agent.Dense') as MockDense, \
             patch('flopyarcade.agent.Activation') as MockAct, \
             patch('flopyarcade.agent.random_uniform', return_value='init'), \
             patch('flopyarcade.agent.SeedGenerator', return_value='seed_gen'), \
             patch('flopyarcade.agent.numpySeed'), \
             patch('flopyarcade.agent.Adam', return_value='optimizer'), \
             patch('flopyarcade.agent.randint', return_value=16):
            model = a.createNNModel('continuous', seed=42)
            assert model is not None

    def test_convolution(self):
        a = make_agent()
        a.hyParams['NNTYPE'] = 'convolution'
        a.observationsVector = np.zeros((16, 16, 1))
        with patch('flopyarcade.agent.Sequential') as MockSeq, \
             patch('flopyarcade.agent.Conv2D') as MockConv, \
             patch('flopyarcade.agent.Dense') as MockDense, \
             patch('flopyarcade.agent.Flatten') as MockFlat, \
             patch('flopyarcade.agent.random_uniform', return_value='init'), \
             patch('flopyarcade.agent.SeedGenerator', return_value='seed_gen'), \
             patch('flopyarcade.agent.numpySeed'), \
             patch('flopyarcade.agent.Adam', return_value='optimizer'), \
             patch('flopyarcade.agent.randint', return_value=16):
            model = a.createNNModel('discrete', seed=42)
            assert model is not None


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

class TestTrain:
    def test_insufficient_memory(self):
        a = make_agent()
        a.replayMemory = deque(maxlen=100)
        a.replayMemory.append([np.array([1.0]), 0, 1.0, np.array([2.0]), False])
        a.train(True, 1)
        # should return early, no fit called

    def test_full_train(self):
        a = make_agent()
        a.replayMemory = deque(maxlen=100)
        # Add 4 transitions (MINIBATCHSIZE=4)
        for i in range(4):
            a.replayMemory.append([np.array([1.0, 2.0]), 1, 1.0, np.array([3.0, 4.0]), False])
        a.mainModel = MagicMock()
        a.mainModel.predict.return_value = np.array([[1.0, 2.0, 3.0, 0.5, 0.1]] * 4)
        a.targetModel = MagicMock()
        a.targetModel.predict.return_value = np.array([[0.5, 1.5, 2.5, 0.3, 0.2]] * 4)
        a.targetUpdateCount = 0
        a.train(True, 4)
        a.mainModel.fit.assert_called_once()
        assert a.targetUpdateCount == 1


# ---------------------------------------------------------------------------
# takeActionsUpdateAndTrainDQN
# ---------------------------------------------------------------------------

class TestTakeActionsUpdateAndTrainDQN:
    def test_basic(self):
        a = make_agent(epsilon=0.0)
        a.hyParams['NAGENTSTEPS'] = 2
        a.replayMemory = deque(maxlen=100)
        a.mainModel = MagicMock()
        a.mainModel.predict_on_batch.return_value = [np.array([0.1, 0.9, 0.1, 0.1, 0.1])]
        a.targetModel = MagicMock()
        a.targetModel.predict.return_value = np.array([[0.5]])
        env = MagicMock()
        env.actionType = 'discrete'
        env.observationsVectorNormalized = np.array([0.5, 0.5])
        env.step.return_value = (np.array([0.5, 0.5]), 1.0, True, False, {})
        env.ENVTYPE = '0s-d'
        a.takeActionsUpdateAndTrainDQN(env)
        assert a.gameStep > 0


# ---------------------------------------------------------------------------
# crossvalidateDQN
# ---------------------------------------------------------------------------

class TestCrossvalidateDQN:
    def test_basic(self):
        a = make_agent()
        a.hyParams['NGAMESCROSSVALIDATED'] = 2
        a.hyParams['NAGENTSTEPS'] = 2
        a.seedsCV = [100, 200]
        a.mainModel = MagicMock()
        a.mainModel.predict_on_batch.return_value = [np.array([0.1, 0.9, 0.1, 0.1, 0.1])]
        env = MagicMock()
        env.actionType = 'discrete'
        env.observationsVectorNormalized = np.array([0.5, 0.5])
        env.step.return_value = (np.array([0.5, 0.5]), 1.0, True, False, {})
        env.ENVTYPE = '0s-d'
        env.initWithSolution = True
        a.crossvalidateDQN(env)
        assert a.gameRewardsCV is not None
        assert len(a.gameRewardsCV) == 2


# ---------------------------------------------------------------------------
# actionNoveltyMetric
# ---------------------------------------------------------------------------

class TestActionNoveltyMetric:
    def test_discrete_same(self):
        a = make_agent()
        actions1 = ['keep', 'left', 'right']
        actions2 = ['keep', 'left', 'right']
        result = a.actionNoveltyMetric(actions1, actions2, 'discrete')
        assert result == 0.0

    def test_discrete_different(self):
        a = make_agent()
        actions1 = ['keep', 'left', 'right']
        actions2 = ['left', 'right', 'up']
        result = a.actionNoveltyMetric(actions1, actions2, 'discrete')
        assert result > 0.0

    def test_continuous(self):
        a = make_agent()
        actions1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        actions2 = np.array([[0.2, 0.3], [0.4, 0.5]])
        result = a.actionNoveltyMetric(actions1, actions2, 'continuous')
        assert result > 0.0

    def test_continuous_identical(self):
        a = make_agent()
        actions1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        actions2 = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = a.actionNoveltyMetric(actions1, actions2, 'continuous')
        assert result == 0.0


# ---------------------------------------------------------------------------
# calculateNoveltyPerPair
# ---------------------------------------------------------------------------

class TestCalculateNoveltyPerPair:
    def test_discrete(self):
        a = make_agent()
        actions1 = [['keep', 'left'], ['right', 'up']]
        actions2 = [['left', 'right'], ['up', 'down']]
        result = a.calculateNoveltyPerPair([actions1, actions2])
        assert result >= 0.0

    def test_empty(self):
        a = make_agent()
        result = a.calculateNoveltyPerPair([[], []])
        assert result == 0.0


# ---------------------------------------------------------------------------
# randomAgentGenetic
# ---------------------------------------------------------------------------

class TestRandomAgentGenetic:
    def test_creates_agent(self):
        a = make_agent()
        a.envSettings['SEEDAGENT'] = 42
        with patch('flopyarcade.agent.FloPyEnv') as MockEnv:
            MockEnv.return_value.getActionType.return_value = 'discrete'
            with patch.object(a, 'createNNModel') as mock_nn:
                mock_agent = MagicMock()
                mock_nn.return_value = mock_agent
                a.randomAgentGenetic(0)
                mock_agent.save.assert_called_once()


# ---------------------------------------------------------------------------
# runAgentsGeneticSingleRun
# ---------------------------------------------------------------------------

class TestRunAgentsGeneticSingleRun:
    def test_basic(self):
        a = make_agent(currentGame=1)
        a.envSettings['KEEPMODELHISTORY'] = False
        a.envSettings['SURROGATESIMULATOR'] = None
        a.envSettings['ENVTYPE'] = '0s-d'
        a.mutationHistory = {'agent1': {'creationSeed': 42, 'mutationSeeds': []}}
        mock_env = MagicMock()
        mock_env.actionType = 'discrete'
        mock_env.observationsVectorNormalized = np.array([0.5, 0.5])
        mock_env.step.return_value = (np.array([0.5, 0.5]), 1.0, True, False, {})
        mock_env.ENVTYPE = '0s-d'
        a.env = mock_env
        mock_agent = MagicMock()
        mock_agent.predict_on_batch.return_value = [np.array([0.1, 0.9, 0.1, 0.1, 0.1])]
        with patch.object(a, 'loadModelFromMutationHistory', return_value=mock_agent), \
             patch('flopyarcade.agent.load_model', return_value=mock_agent), \
             patch.object(a, 'pickleDump'):
            result = a.runAgentsGeneticSingleRun(0)
            assert result == 1.0  # reward from one step


# ---------------------------------------------------------------------------
# runAgentsRepeatedlyGenetic
# ---------------------------------------------------------------------------

class TestRunAgentsRepeatedlyGenetic:
    def test_basic(self):
        a = make_agent()
        a.hyParams['NGAMESAVERAGED'] = 2
        a.hyParams['NGENERATIONS'] = 3
        agent_counts = [0, 1, 2]
        with patch.object(a, 'runAgentsGenetic', return_value=np.array([1.0, 2.0, 3.0])), \
             patch.object(a, 'pickleDump'):
            result = a.runAgentsRepeatedlyGenetic(agent_counts, 2, MagicMock())
        assert np.allclose(result, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# runAgentsGenetic (with mocked Pool)
# ---------------------------------------------------------------------------

class TestRunAgentsGenetic:
    def test_basic(self):
        a = make_agent(geneticGeneration=0, currentGame=1)
        a.envSettings['NAGENTSPARALLEL'] = 1
        a.envSettings['SURROGATESIMULATOR'] = None
        a.envSettings['ENVTYPE'] = '0s-d'
        with patch('flopyarcade.agent.time', return_value=0.0), \
             patch.object(a, 'multiprocessChunks', return_value=[1.0, 2.0, 3.0]), \
             patch('flopyarcade.agent.__name__', 'flopyarcade'):
            result = a.runAgentsGenetic([0, 1, 2], MagicMock())
            assert result == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# returnChildrenGenetic
# ---------------------------------------------------------------------------

class TestReturnChildrenGenetic:
    def test_basic(self):
        a = make_agent(geneticGeneration=0)
        a.hyParams['NGENERATIONS'] = 3
        a.hyParams['NAGENTS'] = 4
        a.hyParams['NAGENTELITES'] = 2
        a.hyParams['NNOVELTYAGENTS'] = 1
        a.rereturnChildrenGenetic = False
        a.searchNovelty = False
        sortedParentIdxs = np.array([0, 1, 2, 3])
        mock_agent = MagicMock()
        a.mutationHistory = {
            'agent1': {'creationSeed': 1, 'mutationSeeds': []},
            'agent2': {'creationSeed': 2, 'mutationSeeds': []},
            'agent3': {'creationSeed': 3, 'mutationSeeds': []},
            'agent4': {'creationSeed': 4, 'mutationSeeds': []},
        }
        with patch('flopyarcade.agent.load_model', return_value=mock_agent), \
             patch.object(a, 'loadModelFromMutationHistory', return_value=mock_agent), \
             patch.object(a, 'multiprocessChunks', return_value=[{'creationSeed': 1, 'mutationSeeds': []}]), \
             patch.object(a, 'pickleLoad', return_value=np.array([1.0, 2.0, 3.0, 4.0])):
            a.returnChildrenGenetic(sortedParentIdxs)


# ---------------------------------------------------------------------------
# resumeGenetic
# ---------------------------------------------------------------------------

class TestResumeGenetic:
    def test_best_exists(self):
        a = make_agent()
        a.envSettings['KEEPMODELHISTORY'] = False
        a.noveltyItemCount = 0
        best_pth = a.tempModelPrefix + '_agentBest.keras'
        with patch('flopyarcade.agent.exists', return_value=True), \
             patch.object(a, 'pickleLoad', side_effect=[
                 np.array([0, 1]),  # sortedParentIdxs
                 {'agent1': {}},  # noveltyArchive
                 {'agent1': {}},  # mutationHistory (KEEPMODELHISTORY=False)
             ]):
            sortedIdxs, cont, brk = a.resumeGenetic()
            assert cont is True
            assert brk is False

    def test_no_best_no_history(self):
        a = make_agent()
        a.envSettings['KEEPMODELHISTORY'] = False
        a.sortedParentIdxs = None
        with patch('flopyarcade.agent.exists', return_value=False):
            sortedIdxs, cont, brk = a.resumeGenetic()
            assert brk is True


# ---------------------------------------------------------------------------
# loadAgentModel
# ---------------------------------------------------------------------------

class TestLoadAgentModel:
    def test_json_exists(self):
        a = make_agent()
        model_prefix = os.path.join(a.modelpth, 'test_model')
        json_path = model_prefix + '.json'
        # Create fake json file
        with open(json_path, 'w') as f:
            f.write('{"class_name": "Sequential"}')
        # Create fake weights file
        with open(model_prefix + '.weights.h5', 'w') as f:
            f.write('fake')
        with patch('flopyarcade.agent.model_from_json') as mock_mfj:
            mock_model = MagicMock()
            mock_mfj.return_value = mock_model
            result = a.loadAgentModel('test_model')
            mock_model.load_weights.assert_called_once()
        os.unlink(json_path)
        os.unlink(model_prefix + '.weights.h5')

    def test_no_json(self):
        a = make_agent()
        model_prefix = os.path.join(a.modelpth, 'test_model2')
        with patch('os.path.exists', return_value=False), \
             patch('flopyarcade.agent.load_model') as mock_lm:
            mock_model = MagicMock()
            mock_model.to_json.return_value = '{}'
            mock_lm.return_value = mock_model
            with patch('flopyarcade.agent.model_from_json'):
                result = a.loadAgentModel('test_model2', compiled=False)
            mock_model.save_weights.assert_called_once()


# ---------------------------------------------------------------------------
# GPUAllowMemoryGrowth
# ---------------------------------------------------------------------------

class TestGPUAllowMemoryGrowth:
    def test_no_gpus(self):
        a = make_agent()
        with patch('tensorflow.config.list_physical_devices', return_value=[]):
            a.GPUAllowMemoryGrowth()

    def test_with_gpus(self):
        a = make_agent()
        gpu = MagicMock()
        with patch('tensorflow.config.list_physical_devices', return_value=[gpu]), \
             patch('tensorflow.config.experimental.set_memory_growth'):
            a.GPUAllowMemoryGrowth()


# ---------------------------------------------------------------------------
# calculateNoveltyNeighborBounds
# ---------------------------------------------------------------------------

class TestCalculateNoveltyNeighborBounds:
    def test_middle(self):
        a = make_agent()
        a.agentsUnique = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        a.hyParams['NNOVELTYNEIGHBORS'] = 4
        a.hyParams['NAGENTS'] = 10
        lower, higher, idx, needs_update = a.calculateNoveltyNeighborBounds(5)
        assert higher > lower

    def test_bottom(self):
        a = make_agent()
        a.agentsUnique = [0, 1, 2, 3, 4]
        a.hyParams['NNOVELTYNEIGHBORS'] = 4
        a.hyParams['NAGENTS'] = 10
        lower, higher, idx, needs_update = a.calculateNoveltyNeighborBounds(1)
        assert lower == 0

    def test_top(self):
        a = make_agent()
        a.agentsUnique = [0, 1, 2, 3, 4]
        a.hyParams['NNOVELTYNEIGHBORS'] = 4
        a.hyParams['NAGENTS'] = 10
        lower, higher, idx, needs_update = a.calculateNoveltyNeighborBounds(4)
        assert higher == 5


# ---------------------------------------------------------------------------
# _run_novelty_search
# ---------------------------------------------------------------------------

class TestRunNoveltySearch:
    def test_basic(self):
        a = make_agent()
        a.cores = 1
        a.maxTasksPerWorkerNoveltySearch = 10000
        with patch.object(a, 'multiprocessChunks', return_value=[0.5, 0.3]):
            result = a._run_novelty_search([[0], [1]])
            assert result == [0.5, 0.3]


# ---------------------------------------------------------------------------
# calculateNoveltyPerAgent
# ---------------------------------------------------------------------------

class TestCalculateNoveltyPerAgent:
    def test_basic(self):
        a = make_agent()
        a.neighborLimitReached = False
        a.tempNoveltypth = os.path.join(tempfile.mkdtemp(), 'nov')
        os.makedirs(a.tempNoveltypth, exist_ok=True)
        with patch.object(a, 'loadActions', return_value=np.array([np.array(['a', 'b']), np.array(['c', 'd'])])), \
             patch('os.path.exists', return_value=False), \
             patch.object(a, 'calculateNoveltyPerPair', return_value=0.5), \
             patch.object(a, 'pickleDump'):
            result = a.calculateNoveltyPerAgent([0, None, 0, 2, 0])
            assert result is not None


# ---------------------------------------------------------------------------
# _regenerate_unique_duplicate_lists
# ---------------------------------------------------------------------------

class TestRegenerateUniqueDuplicateLists:
    def test_discrete(self):
        a = make_agent(noveltyItemCount=2)
        a.noveltyArchive = {
            'agent1': {'modelFile': '/tmp/a1.keras'},
            'agent2': {'modelFile': '/tmp/a2.keras'},
        }
        a.actionsUniqueIDMapping = defaultdict(count().__next__)
        env = MagicMock()
        env.actionType = 'discrete'
        with patch.object(a, 'pickleLoad', side_effect=[
            {'actions': [['keep'], ['left']]},
            {'actions': [['up'], ['down']]},
        ]):
            a._regenerate_unique_duplicate_lists(env)
        assert a.agentsUnique == [0, 1]


# ---------------------------------------------------------------------------
# _cleanup_old_models
# ---------------------------------------------------------------------------

class TestCleanupOldModels:
    def test_cleanup(self):
        a = make_agent()
        with patch('flopyarcade.agent.glob', return_value=[]), \
             patch('flopyarcade.agent.rmtree'), \
             patch('flopyarcade.agent.remove'):
            a._cleanup_old_models()


# ---------------------------------------------------------------------------
# _setup_novelty_path
# ---------------------------------------------------------------------------

class TestSetupNoveltyPath:
    def test_creates_dir(self):
        a = make_agent()
        a.envSettings['RESUME'] = False
        a._setup_novelty_path()
        assert a.tempNoveltypth is not None

    def test_resume_cleans(self):
        a = make_agent()
        a.envSettings['RESUME'] = False
        a.tempNoveltypth = os.path.join(tempfile.mkdtemp(), 'nov')
        os.makedirs(a.tempNoveltypth, exist_ok=True)
        test_file = os.path.join(a.tempNoveltypth, 'old.p')
        with open(test_file, 'w') as f:
            f.write('x')
        with patch('flopyarcade.agent.exists', return_value=True), \
             patch('flopyarcade.agent.listdir', return_value=['old.p']), \
             patch('flopyarcade.agent.remove'):
            a._setup_novelty_path()


# ---------------------------------------------------------------------------
# _initialize_seeds
# ---------------------------------------------------------------------------

class TestInitializeSeeds:
    def test_with_seeds_random(self):
        a = make_agent()
        a.envSettings['SEEDSRANDOM'] = True
        a.hyParams['NAGENTS'] = 2
        a.hyParams['NGENERATIONS'] = 2
        a.hyParams['NGAMESAVERAGED'] = 2
        a._initialize_seeds()
        assert len(a.mutationSeeds) == 2
        assert len(a.gamesSeeds) == 2

    def test_without_seeds_random(self):
        a = make_agent()
        a.envSettings['SEEDSRANDOM'] = False
        a.hyParams['NAGENTS'] = 2
        a.hyParams['NGENERATIONS'] = 2
        del a.gamesSeeds
        a._initialize_seeds()
        assert len(a.mutationSeeds) == 2
        assert not hasattr(a, 'gamesSeeds')


# ---------------------------------------------------------------------------
# _setup_genetic_agent
# ---------------------------------------------------------------------------

class TestSetupGeneticAgent:
    def test_basic(self):
        a = make_agent()
        a.envSettings['NAGENTSPARALLEL'] = None
        a.envSettings['RESUME'] = False
        a.envSettings['BESTAGENTANIMATION'] = False
        a.hyParams['NOVELTYSEARCH'] = False
        with patch('flopyarcade.agent.cpu_count', return_value=4), \
             patch.object(a, '_cleanup_old_models'), \
             patch.object(a, '_initialize_seeds'), \
             patch.object(a, 'initializeGeneticAgents'), \
             patch.object(a, 'pickleDump'):
            a._setup_genetic_agent()
        assert a.envSettings['NAGENTSPARALLEL'] == 4


# ---------------------------------------------------------------------------
# initializeGeneticAgents
# ---------------------------------------------------------------------------

class TestInitializeGeneticAgents:
    def test_with_history(self):
        a = make_agent()
        a.envSettings['KEEPMODELHISTORY'] = True
        a.hyParams['NAGENTS'] = 2
        with patch.object(a, 'multiprocessChunks'):
            a.initializeGeneticAgents()

    def test_without_history(self):
        a = make_agent()
        a.envSettings['KEEPMODELHISTORY'] = False
        a.hyParams['NAGENTS'] = 3
        a.initializeGeneticAgents()
        assert len(a.mutationHistory) == 3


# ---------------------------------------------------------------------------
# setSeeds
# ---------------------------------------------------------------------------

class TestSetSeeds:
    def test_set(self, capsys):
        a = make_agent()
        with patch('flopyarcade.agent.numpySeed'), \
             patch('flopyarcade.agent.randomSeed'), \
             patch('flopyarcade.agent.set_random_seed'):
            a.setSeeds()
        assert a.SEED == 42


# ---------------------------------------------------------------------------
# _init_seeds
# ---------------------------------------------------------------------------

class TestInitSeeds:
    def test_init(self):
        a = make_agent()
        with patch('flopyarcade.agent.numpySeed'), \
             patch('flopyarcade.agent.randomSeed'), \
             patch('flopyarcade.agent.set_random_seed'):
            a._init_seeds()
        assert a.SEED == 42


# ---------------------------------------------------------------------------
# _play_one_game
# ---------------------------------------------------------------------------

class TestPlayOneGame:
    def test_with_success(self):
        a = make_agent()
        a.gameStep = 5
        env = MagicMock()
        env.success = True
        with patch.object(a, 'takeActionsUpdateAndTrainDQN'):
            a._play_one_game(env)

    def test_without_success(self):
        a = make_agent()
        a.gameStep = 1
        env = MagicMock()
        env.success = False
        a.replayMemory = deque(maxlen=100)
        a.replayMemory.append([1, 2, 3.0, 4, False])
        with patch.object(a, 'takeActionsUpdateAndTrainDQN'):
            a._play_one_game(env)
        assert a.replayMemory[0][2] == 0.0


# ---------------------------------------------------------------------------
# _crossvalidate_and_save
# ---------------------------------------------------------------------------

class TestCrossvalidateAndSave:
    def test_save_triggered(self):
        a = make_agent()
        a.average_rewardCV = 1.0
        a.max_rewardCV = 2.0
        a.min_rewardCV = 0.5
        a.envSettings['REWARDMINTOSAVE'] = 0.5
        env = MagicMock()
        with patch.object(a, 'crossvalidateDQN'), \
             patch.object(a, 'mainModel', create=True) as mock_model:
            a._crossvalidate_and_save(env, 1)


# ---------------------------------------------------------------------------
# _create_folder
# ---------------------------------------------------------------------------

class TestCreateFolder:
    def test_creates(self):
        a = make_agent()
        test_path = os.path.join(tempfile.mkdtemp(), 'new_dir')
        a._create_folder(test_path, assign_attr='testattr')
        assert os.path.exists(test_path)
        assert a.testattr == test_path

    def test_exists(self):
        a = make_agent()
        a._create_folder(a.wrkspc)  # already exists


# ---------------------------------------------------------------------------
# mutateGenetic
# ---------------------------------------------------------------------------

class TestMutateGenetic:
    def test_mutate(self):
        a = make_agent()
        mock_agent = MagicMock()
        w = np.ones((4, 4))
        mock_agent.get_weights.return_value = [w]
        with patch('flopyarcade.agent.numpySeed'), \
             patch('flopyarcade.agent.randn', return_value=np.ones((4, 4)) * 0.1), \
             patch('flopyarcade.agent.random', return_value=np.ones((4, 4))):
            result = a.mutateGenetic(mock_agent, 42)
            mock_agent.set_weights.assert_called_once()


# ---------------------------------------------------------------------------
# runDQN
# ---------------------------------------------------------------------------

class TestRunDQN:
    def test_basic(self):
        a = make_agent()
        a.hyParams['NGAMES'] = 2
        a.hyParams['CROSSVALIDATEEVERY'] = 2
        a.gameSeeds = [100, 200, 300]
        a.seedsCV = np.array([1, 2])
        env = MagicMock()
        env.actionType = 'discrete'
        env.MODELNAME = 'test'
        a.gameReward = 0.0
        with patch('flopyarcade.agent.tqdm', side_effect=lambda x, **kw: x), \
             patch.object(a, '_play_one_game'), \
             patch.object(a, '_crossvalidate_and_save'), \
             patch.object(a, '_decay_epsilon'), \
             patch('flopyarcade.agent.randint', side_effect=lambda **kw: 999):
            a.runDQN(env)


# ---------------------------------------------------------------------------
# runGenetic
# ---------------------------------------------------------------------------

class TestRunGenetic:
    def test_basic(self):
        a = make_agent(geneticGeneration=0)
        a.hyParams['NGENERATIONS'] = 1
        a.hyParams['NAGENTS'] = 2
        a.envSettings['RESUME'] = False
        a.envSettings['ENVTYPE'] = '0s-d'
        a.envSettings['BESTAGENTANIMATION'] = False
        a.n = 1
        a.searchNovelty = False
        with patch.object(a, '_validate_hyperparams'), \
             patch.object(a, '_initialize_environment'), \
             patch.object(a, '_initialize_parallel_settings'), \
             patch.object(a, '_generate_unique_pid'), \
             patch.object(a, 'generatePathPrefixes'), \
             patch.object(a, 'runAgentsRepeatedlyGenetic', return_value=np.array([1.0, 2.0])), \
             patch.object(a, '_log_generation_rewards'), \
             patch.object(a, '_sort_and_save_top_agents', return_value=np.array([0, 1])), \
             patch.object(a, 'perform_novelty_search'), \
             patch.object(a, 'returnChildrenGenetic'), \
             patch.object(a, 'saveBestAgent'), \
             patch.object(a, 'pickleDump'):
            a.runGenetic(MagicMock(), searchNovelty=False)


# ---------------------------------------------------------------------------
# returnChildrenGeneticSingleRun
# ---------------------------------------------------------------------------

class TestReturnChildrenGeneticSingleRun:
    def test_basic(self):
        a = make_agent(geneticGeneration=0)
        a.envSettings['KEEPMODELHISTORY'] = False
        a.searchNovelty = False
        a.candidateParentIdxs = [0, 1, 2]
        a.mutationHistory = {'agent1': {'creationSeed': 42, 'mutationSeeds': []}}
        mock_agent = MagicMock()
        with patch('flopyarcade.agent.randint', return_value=0), \
             patch.object(a, 'loadModelFromMutationHistory', return_value=mock_agent), \
             patch.object(a, 'mutateGenetic', return_value=mock_agent), \
             patch.object(a, 'createMutationRecord', return_value={'creationSeed': 42, 'mutationSeeds': [1]}):
            result = a.returnChildrenGeneticSingleRun(0)
            assert result['creationSeed'] == 42


# ---------------------------------------------------------------------------
# loadModelFromMutationHistory edge
# ---------------------------------------------------------------------------

class TestSuppressTensorFlowWarnings:
    def test_module_level(self):
        """Verify the module-level TF warning suppression ran successfully."""
        import flopyarcade.agent as agent_mod
        from logging import getLogger, FATAL
        tf_logger = getLogger('tensorflow')
        assert tf_logger.level == FATAL
