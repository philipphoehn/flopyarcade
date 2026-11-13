
"""Coverage tests for flopyarcade: agent, benchmark, actions, observations.
Uses object.__new__ to bypass heavy __init__ and unbound mixin calls."""
import sys, os, json, pickle
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from collections import defaultdict, deque


# ===================== AGENT =====================

def make_agent():
    from flopyarcade.agent import FloPyAgent
    a = object.__new__(FloPyAgent)
    a.hyParams = {
        'NAGENTS': 5, 'NNOVELTYELITES': 2, 'NGENERATIONS': 3,
        'NGAMESAVERAGED': 2, 'NGAMES': 3, 'NGAMESCROSSVALIDATED': 2,
        'CROSSVALIDATEEVERY': 1, 'REPLAYMEMORYSIZE': 100,
        'REPLAYMEMORYSIZEMIN': 10, 'MINIBATCHSIZE': 4, 'DISCOUNT': 0.99,
        'EPSILONINITIAL': 1.0, 'EPSILONMIN': 0.05, 'EPSILONDECAY': 0.95,
        'UPDATEPREDICTIVEMODELEVERY': 10, 'NHIDDENNODES': [128, 64],
        'NNTYPE': 'perceptron', 'ARCHITECTUREVARY': False,
        'BATCHNORMALIZATION': False, 'HIDDENACTIVATIONS': ['relu', 'relu'],
        'MUTATIONPROBABILITY': 0.1, 'MUTATIONPOWER': 0.1,
        'NAGENTELITES': 2, 'NNOVELTYAGENTS': 1, 'NNOVELTYNEIGHBORS': 3,
        'ADDNOVELTYEVERY': 2, 'NOVELTYSEARCH': False, 'DROPOUTS': [0.0, 0.0]
    }
    a.envSettings = {
        'ENVTYPE': '1s-d', 'MODELNAME': 'test', 'NAGENTSPARALLEL': 2,
        'RESUME': False, 'BESTAGENTANIMATION': False, 'KEEPMODELHISTORY': False,
        'SEEDSRANDOM': False, 'SEEDAGENT': 42, 'SEEDENV': 100,
        'RENDER': False, 'RENDEREVERY': 5, 'REWARDMINTOSAVE': 50.0,
        'SURROGATESIMULATOR': None, 'PATHMF2005': None, 'PATHMP6': None,
        'NLAY': 1, 'NROW': 10, 'NCOL': 10
    }
    a.epsilon = 1.0
    a.replayMemory = deque(maxlen=100)
    a.agentsUnique = []
    a.agentsUniqueIDs = []
    a.agentsDuplicate = []
    a.noveltyItemCount = 0
    a.noveltyArchive = {}
    a.mutationHistory = {}
    a.cores = 1
    a.geneticGeneration = 0
    a.zFill = 6
    a.tempModelPrefix = '/tmp/a0test'
    a.tempNextModelPrefix = '/tmp/a0next'
    a.searchNovelty = False
    a.mainModel = MagicMock()
    a.targetModel = MagicMock()
    a.targetUpdateCount = 0
    a.actionSpace = ['up', 'down', 'keep']
    return a


class TestAgentBasic:
    def test_action_type(self):
        a = make_agent()
        assert a.getActionType('1s-d') == 'discrete'
        assert a.getActionType('1s-c') == 'continuous'

    def test_decay_epsilon(self):
        a = make_agent()
        a.epsilon = 1.0
        a._decay_epsilon()
        assert abs(a.epsilon - 0.95) < 1e-9

    def test_decay_epsilon_min(self):
        a = make_agent()
        a.epsilon = 0.01
        a._decay_epsilon()
        assert a.epsilon == 0.01

    def test_validate_ok(self):
        a = make_agent()
        a._validate_hyperparams()

    def test_validate_fail(self):
        a = make_agent()
        a.hyParams['NAGENTS'] = 2
        a.hyParams['NNOVELTYELITES'] = 3
        with pytest.raises(ValueError):
            a._validate_hyperparams()

    def test_parallel_settings(self):
        a = make_agent()
        a._initialize_parallel_settings()
        assert a.cores == 2

    def test_yield_chunks(self):
        a = make_agent()
        assert list(a.yieldChunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    def test_count_parameters(self):
        a = make_agent()
        m = MagicMock()
        m.get_weights.return_value = [np.ones((2, 3)), np.ones(3)]
        assert a.countParameters(m) == 9

    def test_check_parameter_mean(self):
        a = make_agent()
        m = MagicMock()
        m.get_weights.return_value = [np.ones((2, 3)), np.full(3, 2.0)]
        assert a.checkParameterMean(m) == 1.5

    def test_mutate_decision(self):
        a = make_agent()
        res = a.mutateDecision(1.0)
        assert res

    def test_mutate_decisions(self):
        a = make_agent()
        res = a.mutateDecisions(1.0, (3, 3))
        assert res.shape == (3, 3)

    def test_update_replay_memory(self):
        a = make_agent()
        a.updateReplayMemory([np.random.randn(4), 0, 1.0, np.random.randn(4), False])
        assert len(a.replayMemory) == 1

    def test_update_replay_zero(self):
        a = make_agent()
        a.replayMemory = deque(maxlen=100)
        a.replayMemory.append([np.random.randn(4), 0, 1.0, np.random.randn(4), False])
        a.updateReplayMemoryZeroReward(1)
        assert a.replayMemory[0][2] == 0.0

    def test_mutation_record(self):
        a = make_agent()
        rec = a.createMutationRecord(42, [1, 2], 3)
        assert rec['creationSeed'] == 42
        assert rec['mutationSeeds'] == [1, 2, 3]

    def test_update_mutation_history(self):
        a = make_agent()
        h = a.updateMutationHistory({}, 5, 10, [], 20)
        assert h['agent5']['creationSeed'] == 10

    def test_getqs(self):
        a = make_agent()
        m = MagicMock()
        m.predict_on_batch.return_value = [np.array([0.1, 0.5, 0.4])]
        res = a.getqsGivenAgentModel(m, np.random.randn(10))
        assert len(res) == 3

    def test_get_action_random_discrete(self):
        a = make_agent()
        act = a.getAction(mode='random', actionType='discrete')
        assert act in a.actionSpace

    def test_get_action_random_continuous(self):
        a = make_agent()
        a.actionSpaceSize = 3
        act = a.getAction(mode='random', actionType='continuous')
        assert len(act) == 3

    def test_get_action_manual(self):
        a = make_agent()
        assert a.getAction(mode='manual', keyPressed='up', actionType='discrete') == 'up'
        assert a.getAction(mode='manual', keyPressed='x', actionType='discrete') == 'keep'

    def test_novelty_metric_discrete(self):
        a = make_agent()
        a1 = ['up', 'down', 'keep', 'up']
        a2 = ['up', 'down', 'keep', 'up']
        assert a.actionNoveltyMetric(a1, a2, 'discrete') == 0.0

    def test_novelty_metric_continuous(self):
        a = make_agent()
        a1 = [[0.1, 0.2], [0.3, 0.4]]
        a2 = [[0.5, 0.6], [0.7, 0.8]]
        assert a.actionNoveltyMetric(a1, a2, 'continuous') > 0





    def test_pickle_roundtrip(self, tmp_path):
        a = make_agent()
        p = str(tmp_path / 'x.p')
        a.pickleDump(p, {'k': 'v'})
        assert a.pickleLoad(p) == {'k': 'v'}


class TestAgentNovelty:
    def test_update_discrete_uniques(self):
        a = make_agent()
        a._update_discrete_uniques(42)
        assert 0 in a.agentsUnique

    def test_update_discrete_dup(self):
        a = make_agent()
        a.noveltyItemCount = 1
        a.agentsUnique = [0]
        a.agentsUniqueIDs = [42]
        a._update_discrete_uniques(42)
        assert 1 in a.agentsDuplicate

    def test_reset_continuous(self):
        a = make_agent()
        a.noveltyItemCount = 3
        a.noveltyArchive = {f'agent{i+1}': {} for i in range(4)}
        a._reset_continuous_uniques()
        assert a.agentsUnique == [0, 1, 2, 3]

    def test_print_stats(self, capsys):
        a = make_agent()
        a.agentsUnique = [0, 1]
        a.agentsDuplicate = [2]
        a.noveltyItemCount = 10
        a._print_novelty_stats()
        assert '2 unique' in capsys.readouterr().out

    def test_build_novelty_args(self):
        a = make_agent()
        a.agentsUnique = [0, 1, 2]
        a.noveltyItemCount = 10
        args, flags = a._build_novelty_args()
        assert len(args) == 3 and len(flags) == 3

    def test_update_novelties(self):
        a = make_agent()
        a.agentsUnique = [0, 1]
        a.agentsDuplicate = [2]
        a.noveltyArchive = {
            'agent1': {'novelty': 0}, 'agent2': {'novelty': 0},
            'agent3': {'actionsUniqueID': 0, 'novelty': 0},
        }
        a._update_agent_novelties([5.0, 7.0], [True, True])
        assert a.noveltyArchive['agent1']['novelty'] == 5.0

    def test_neighbor_bounds(self):
        a = make_agent()
        a.agentsUnique = list(range(10))
        a.noveltyItemCount = 10
        res = a.calculateNoveltyNeighborBounds(4)
        assert len(res) == 4

    def test_save_novelties(self, tmp_path):
        a = make_agent()
        a.noveltyItemCount = 2
        a.noveltyArchive = {
            'agent1': {'novelty': 1.0, 'modelFile': '/tmp/a1'},
            'agent2': {'novelty': 2.0, 'modelFile': '/tmp/a2'},
        }
        a.tempModelPrefix = str(tmp_path / 't')
        with patch('flopyarcade.archive_hdf5.NoveltyArchiveHDF5') as M:
            a._save_novelties()
        assert a.novelties == [1.0, 2.0]


class TestAgentGenetic:
    def test_mutate_genetic(self):
        a = make_agent()
        m = MagicMock()
        w = [np.ones((2, 3)), np.ones(3)]
        m.get_weights.return_value = [x.copy() for x in w]
        a.mutateGenetic(m, 42)
        m.set_weights.assert_called()




class TestAgentTrain:
    def test_train_below_min(self):
        a = make_agent()
        for i in range(5):
            a.replayMemory.append([np.random.randn(4), 0, 1.0, np.random.randn(4), False])
        a.train(False, 1)

    def test_train_full(self):
        a = make_agent()
        for i in range(12):
            a.replayMemory.append([np.random.randn(4), np.random.randint(3), np.random.randn(), np.random.randn(4), False])
        a.mainModel.predict.return_value = np.random.rand(12, 3)
        a.targetModel.predict.return_value = np.random.rand(12, 3)
        a.train(True, 12)
        a.mainModel.fit.assert_called()

