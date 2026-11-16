#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com

# currently on tested stable using python 3.8
# set PYTHONWARNINGS=ignore::ResourceWarning
# python -m coverage run --source=flopyarcade -m unittest

import flopyarcade.examples.train_dqn as train_dqn
from flopyarcade import FloPyArcade
from glob import glob
from os import remove
from os.path import abspath, dirname, isdir, isfile, join
from shutil import rmtree
import unittest


class TestFloPyAgentDQN(unittest.TestCase):

    def test_FloPyAgentDQN_noExceptionRaised(self):
        """Test an agent using the deep Q-Learning algorithm on a
        computationally simple case.
        """

        # from train_dqn import envSettings, hyParams
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams

        failed = []
        # for ENVTYPE in FloPyArcade().ENVTYPES:
        for ENVTYPE in ['1s-d']:
            raised = False
            # try:
            # environment settings
            envSettings['ENVTYPE'] = '1s-d'
            envSettings['MODELNAME'] = 'unittest'
            envSettings['RENDER'] = False
            # hyperparameters
            hyParams['NGAMES'] = 4
            hyParams['NAGENTSTEPS'] = 3
            hyParams['REPLAYMEMORYSIZE'] = 10000
            hyParams['REPLAYMEMORYSIZEMIN'] = 3
            hyParams['MINIBATCHSIZE'] = 3
            hyParams['UPDATEPREDICTIVEMODELEVERY'] = 1
            hyParams['NHIDDENNODES'] = [5] * 2
            hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
            hyParams['DROPOUTS'] = [0.0] * 2
            hyParams['CROSSVALIDATEEVERY'] = 2
            hyParams['NGAMESCROSSVALIDATED'] = 2
            train_dqn.main(envSettings, hyParams)
            # FloPyArcadeDQN.main(envSettings, hyParams)
            print('Test of deep Q-network agent in environment ' + ENVTYPE + ' succeeded.')
            # except Exception as e:
            #     raised = True
            #     failed.append(ENVTYPE)
            #     print('error train_dqn:', e)
        for ENVTYPE in failed:
            self.assertFalse(ENVTYPE, 'Test of deep Q-network agent in environment ' + ENVTYPE + ' failed.')


    def test_targetModel_exists_after_init(self):
        """Verify that targetModel is created during DQN agent initialization."""
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams
        envSettings['ENVTYPE'] = '1s-d'
        envSettings['MODELNAME'] = 'unittest_ddqn'
        envSettings['RENDER'] = False
        hyParams['NHIDDENNODES'] = [5] * 2
        hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
        hyParams['DROPOUTS'] = [0.0] * 2
        hyParams['REPLAYMEMORYSIZE'] = 100
        hyParams['EPSILONINITIAL'] = 1.0

        from flopyarcade import FloPyAgent, FloPyEnv
        env = FloPyEnv(
            envSettings['ENVTYPE'], None, None,
            envSettings['MODELNAME'], flagRender=False,
            NAGENTSTEPS=3, OBSPREP='perceptron')
        agent = FloPyAgent(
            env.observationsVectorNormalized,
            env.actionSpace, hyParams, envSettings, 'DQN')

        self.assertTrue(hasattr(agent, 'targetModel'),
                        'targetModel attribute missing after DQN init')
        self.assertIsNotNone(agent.targetModel)

    def test_targetModel_same_architecture_as_mainModel(self):
        """Verify targetModel has the same architecture as mainModel."""
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams
        envSettings['ENVTYPE'] = '1s-d'
        envSettings['MODELNAME'] = 'unittest_ddqn'
        envSettings['RENDER'] = False
        hyParams['NHIDDENNODES'] = [5] * 2
        hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
        hyParams['DROPOUTS'] = [0.0] * 2
        hyParams['REPLAYMEMORYSIZE'] = 100
        hyParams['EPSILONINITIAL'] = 1.0

        from flopyarcade import FloPyAgent, FloPyEnv
        env = FloPyEnv(
            envSettings['ENVTYPE'], None, None,
            envSettings['MODELNAME'], flagRender=False,
            NAGENTSTEPS=3, OBSPREP='perceptron')
        agent = FloPyAgent(
            env.observationsVectorNormalized,
            env.actionSpace, hyParams, envSettings, 'DQN')

        # Compare layer-by-layer architecture
        main_layers = agent.mainModel.layers
        target_layers = agent.targetModel.layers
        self.assertEqual(len(main_layers), len(target_layers),
                         'Layer count mismatch between mainModel and targetModel')
        for i, (ml, tl) in enumerate(zip(main_layers, target_layers)):
            self.assertEqual(ml.__class__.__name__, tl.__class__.__name__,
                             f'Layer {i} class mismatch: {ml.__class__.__name__} vs {tl.__class__.__name__}')
            self.assertEqual(ml.count_params(), tl.count_params(),
                             f'Layer {i} param count mismatch: {ml.count_params()} vs {tl.count_params()}')

    def test_train_uses_targetNetwork_for_bootstrapping(self):
        """Verify that train() uses targetModel (not mainModel) for Q-target computation."""
        import numpy as np
        from unittest.mock import patch
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams
        envSettings['ENVTYPE'] = '1s-d'
        envSettings['MODELNAME'] = 'unittest_ddqn'
        envSettings['RENDER'] = False
        hyParams['NHIDDENNODES'] = [5] * 2
        hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
        hyParams['DROPOUTS'] = [0.0] * 2
        hyParams['REPLAYMEMORYSIZE'] = 100
        hyParams['REPLAYMEMORYSIZEMIN'] = 3
        hyParams['MINIBATCHSIZE'] = 3
        hyParams['EPSILONINITIAL'] = 1.0
        hyParams['UPDATEPREDICTIVEMODELEVERY'] = 100

        from flopyarcade import FloPyAgent, FloPyEnv
        env = FloPyEnv(
            envSettings['ENVTYPE'], None, None,
            envSettings['MODELNAME'], flagRender=False,
            NAGENTSTEPS=3, OBSPREP='perceptron')
        agent = FloPyAgent(
            env.observationsVectorNormalized,
            env.actionSpace, hyParams, envSettings, 'DQN')

        # Create fake replay memory entries with correct observation shape
        obs_len = len(env.observationsVectorNormalized)
        fake_transitions = []
        for _ in range(hyParams['REPLAYMEMORYSIZEMIN']):
            state = np.random.rand(obs_len).astype(np.float32)
            action = 0
            reward = 1.0
            next_state = np.random.rand(obs_len).astype(np.float32)
            done = False
            fake_transitions.append([state, action, reward, next_state, done])
        agent.replayMemory.extend(fake_transitions)

        # Patch targetModel.predict to track calls
        with patch.object(agent.targetModel, 'predict', wraps=agent.targetModel.predict) as target_predict_mock:
            with patch.object(agent.mainModel, 'predict', wraps=agent.mainModel.predict):
                with patch.object(agent.mainModel, 'fit', wraps=agent.mainModel.fit):
                    agent.train(terminal_state=True, step=1)

        self.assertTrue(target_predict_mock.called,
                        'targetModel.predict was not called during train() - '
                        'Double-DQN bootstrapping is not using the target network')

    def test_double_dqn_mainModel_predict_called_for_next_states(self):
        """Verify Double-DQN: mainModel.predict is called for next-state action selection."""
        import numpy as np
        from unittest.mock import patch
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams
        envSettings['ENVTYPE'] = '1s-d'
        envSettings['MODELNAME'] = 'unittest_ddqn'
        envSettings['RENDER'] = False
        hyParams['NHIDDENNODES'] = [5] * 2
        hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
        hyParams['DROPOUTS'] = [0.0] * 2
        hyParams['REPLAYMEMORYSIZE'] = 100
        hyParams['REPLAYMEMORYSIZEMIN'] = 3
        hyParams['MINIBATCHSIZE'] = 3
        hyParams['EPSILONINITIAL'] = 1.0
        hyParams['UPDATEPREDICTIVEMODELEVERY'] = 100

        from flopyarcade import FloPyAgent, FloPyEnv
        env = FloPyEnv(
            envSettings['ENVTYPE'], None, None,
            envSettings['MODELNAME'], flagRender=False,
            NAGENTSTEPS=3, OBSPREP='perceptron')
        agent = FloPyAgent(
            env.observationsVectorNormalized,
            env.actionSpace, hyParams, envSettings, 'DQN')

        obs_len = len(env.observationsVectorNormalized)
        for _ in range(hyParams['REPLAYMEMORYSIZEMIN']):
            state = np.random.rand(obs_len).astype(np.float32)
            next_state = np.random.rand(obs_len).astype(np.float32)
            agent.replayMemory.append([state, 0, 1.0, next_state, False])

        with patch.object(agent.mainModel, 'predict', wraps=agent.mainModel.predict) as main_predict_mock:
            with patch.object(agent.targetModel, 'predict', wraps=agent.targetModel.predict):
                with patch.object(agent.mainModel, 'fit', wraps=agent.mainModel.fit):
                    agent.train(terminal_state=True, step=1)

        # mainModel.predict should be called at least twice:
        # once for current states (Q-values) and once for next states (action selection)
        self.assertGreaterEqual(
            main_predict_mock.call_count, 2,
            f'mainModel.predict called {main_predict_mock.call_count} times; '
            'expected >=2 (current states + next-state action selection for Double-DQN)')

    def test_double_dqn_argmax_from_online_not_target(self):
        """Verify the selected next action is argmax of online Q, not target Q."""
        import numpy as np
        from unittest.mock import patch
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams
        envSettings['ENVTYPE'] = '1s-d'
        envSettings['MODELNAME'] = 'unittest_ddqn'
        envSettings['RENDER'] = False
        hyParams['NHIDDENNODES'] = [5] * 2
        hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
        hyParams['DROPOUTS'] = [0.0] * 2
        hyParams['REPLAYMEMORYSIZE'] = 100
        hyParams['REPLAYMEMORYSIZEMIN'] = 3
        hyParams['MINIBATCHSIZE'] = 3
        hyParams['EPSILONINITIAL'] = 1.0
        hyParams['UPDATEPREDICTIVEMODELEVERY'] = 100
        hyParams['DISCOUNT'] = 0.9

        from flopyarcade import FloPyAgent, FloPyEnv
        env = FloPyEnv(
            envSettings['ENVTYPE'], None, None,
            envSettings['MODELNAME'], flagRender=False,
            NAGENTSTEPS=3, OBSPREP='perceptron')
        agent = FloPyAgent(
            env.observationsVectorNormalized,
            env.actionSpace, hyParams, envSettings, 'DQN')

        obs_len = len(env.observationsVectorNormalized)
        n_actions = len(env.actionSpace)

        # Set up networks to produce KNOWN outputs:
        # online: action 0 best (Q=[5.0, 1.0, 1.0, ...])
        # target: action 1 best (Q=[1.0, 9.0, 1.0, ...])
        # In standard DQN (target argmax): would pick action 1
        # In Double-DQN (online argmax): should pick action 0

        online_q = np.zeros((3, n_actions), dtype=np.float32)
        online_q[:, 0] = 5.0  # action 0 is best for online
        target_q = np.zeros((3, n_actions), dtype=np.float32)
        target_q[:, 1] = 9.0  # action 1 is best for target

        state = np.random.rand(obs_len).astype(np.float32)
        next_state = np.random.rand(obs_len).astype(np.float32)
        for _ in range(hyParams['REPLAYMEMORYSIZEMIN']):
            agent.replayMemory.append([state.copy(), 0, 1.0, next_state.copy(), False])

        with patch.object(agent.mainModel, 'predict') as main_pred:
            with patch.object(agent.targetModel, 'predict') as target_pred:
                with patch.object(agent.mainModel, 'fit') as fit_mock:
                    # mainModel called first for current states, then for next states
                    main_pred.side_effect = lambda x: (online_q if x.shape[0] == 3 else online_q)
                    target_pred.return_value = target_q
                    agent.train(terminal_state=True, step=1)

        # Capture the target (y) passed to fit
        fit_call = fit_mock.call_args
        y_array = np.array(fit_call[0][1])  # second positional arg

        # For Double-DQN: new_q = reward + discount * target_q[best_online_action]
        # best_online_action = argmax(online_q) = 0  (online says action 0 is best)
        # target_q[0] = 0.0  (target has 0.0 at action 0, 9.0 at action 1)
        # new_q = 1.0 + 0.9 * 0.0 = 1.0
        # For standard DQN: best = argmax(target_q) = 1, new_q = 1.0 + 0.9*9.0 = 9.1
        expected_new_q = 1.0 + 0.9 * 0.0  # = 1.0
        actual_new_q = y_array[0][0]  # Q-value at action 0

        self.assertAlmostEqual(
            float(actual_new_q), expected_new_q, places=3,
            msg=(f'Expected new_q={expected_new_q} (Double-DQN: online argmax→action 0, target eval→0.0), '
                 f'got {actual_new_q}. '
                 f'If it were standard DQN (target argmax→action 1), it would be 1.0+0.9*9.0=9.1'))

    def test_double_dqn_terminal_state_skips_next_eval(self):
        """Verify terminal states use reward directly without next-state evaluation."""
        import numpy as np
        from unittest.mock import patch
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams
        envSettings['ENVTYPE'] = '1s-d'
        envSettings['MODELNAME'] = 'unittest_ddqn'
        envSettings['RENDER'] = False
        hyParams['NHIDDENNODES'] = [5] * 2
        hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
        hyParams['DROPOUTS'] = [0.0] * 2
        hyParams['REPLAYMEMORYSIZE'] = 100
        hyParams['REPLAYMEMORYSIZEMIN'] = 3
        hyParams['MINIBATCHSIZE'] = 3
        hyParams['EPSILONINITIAL'] = 1.0
        hyParams['UPDATEPREDICTIVEMODELEVERY'] = 100

        from flopyarcade import FloPyAgent, FloPyEnv
        env = FloPyEnv(
            envSettings['ENVTYPE'], None, None,
            envSettings['MODELNAME'], flagRender=False,
            NAGENTSTEPS=3, OBSPREP='perceptron')
        agent = FloPyAgent(
            env.observationsVectorNormalized,
            env.actionSpace, hyParams, envSettings, 'DQN')

        obs_len = len(env.observationsVectorNormalized)
        state = np.random.rand(obs_len).astype(np.float32)
        next_state = np.random.rand(obs_len).astype(np.float32)
        # All transitions are terminal (done=True)
        for _ in range(hyParams['REPLAYMEMORYSIZEMIN']):
            agent.replayMemory.append([state.copy(), 2, 7.5, next_state.copy(), True])

        with patch.object(agent.mainModel, 'fit') as fit_mock:
            with patch.object(agent.mainModel, 'predict', return_value=np.zeros((3, len(env.actionSpace)), dtype=np.float32)):
                with patch.object(agent.targetModel, 'predict', return_value=np.ones((3, len(env.actionSpace)), dtype=np.float32)):
                    agent.train(terminal_state=True, step=1)

        y_array = np.array(fit_mock.call_args[0][1])
        # Terminal: new_q = reward = 7.5 (no discount, no next-state term)
        self.assertAlmostEqual(float(y_array[0][2]), 7.5, places=4,
                              msg='Terminal state should set new_q = reward directly')


    def test_dqn_checkpoint_save_load(self):
        """Verify save_dqn_checkpoint and load_dqn_checkpoint restore all state."""
        import numpy as np
        import tempfile
        import shutil

        train_dqn = self._get_train_dqn()
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams
        envSettings['ENVTYPE'] = '1s-d'
        envSettings['MODELNAME'] = 'unittest_ckpt'
        envSettings['RENDER'] = False
        hyParams['NHIDDENNODES'] = [5] * 2
        hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
        hyParams['DROPOUTS'] = [0.0] * 2
        hyParams['REPLAYMEMORYSIZE'] = 100
        hyParams['REPLAYMEMORYSIZEMIN'] = 3
        hyParams['MINIBATCHSIZE'] = 3
        hyParams['EPSILONINITIAL'] = 0.5

        from flopyarcade import FloPyAgent, FloPyEnv
        env = FloPyEnv(
            envSettings['ENVTYPE'], None, None,
            envSettings['MODELNAME'], flagRender=False,
            NAGENTSTEPS=3, OBSPREP='perceptron')
        agent = FloPyAgent(
            env.observationsVectorNormalized,
            env.actionSpace, hyParams, envSettings, 'DQN')

        agent.epsilon = 0.42
        agent.dqnGameCounter = 42
        agent.targetUpdateCount = 7
        obs_len = len(env.observationsVectorNormalized)
        fake_transitions = []
        for _ in range(3):
            state = np.random.rand(obs_len).astype(np.float32)
            fake_transitions.append([state, 0, 1.0, state.copy(), False])
        agent.replayMemory.extend(fake_transitions)

        orig_main_weights = [w.copy() for w in agent.mainModel.get_weights()]
        orig_target_weights = [w.copy() for w in agent.targetModel.get_weights()]
        orig_replay = [list(t) for t in list(agent.replayMemory)]

        tmpdir = tempfile.mkdtemp(prefix='dqn_ckpt_test_')
        ckpt_dir = join(tmpdir, 'test_checkpoint')
        agent.save_dqn_checkpoint(ckpt_dir)

        agent.epsilon = 0.99
        agent.dqnGameCounter = 0
        agent.targetUpdateCount = 0
        agent.replayMemory.clear()
        perturbed = [np.zeros_like(w) for w in agent.mainModel.get_weights()]
        agent.mainModel.set_weights(perturbed)

        agent.load_dqn_checkpoint(ckpt_dir)

        self.assertAlmostEqual(agent.epsilon, 0.42, msg='epsilon not restored')
        self.assertEqual(agent.dqnGameCounter, 42, msg='game counter not restored')
        self.assertEqual(agent.targetUpdateCount, 7, msg='targetUpdateCount not restored')
        self.assertEqual(len(agent.replayMemory), 3, msg='replay memory length not restored')
        for i, (orig, restored) in enumerate(zip(orig_replay, list(agent.replayMemory))):
            np.testing.assert_array_equal(np.array(orig[0]), np.array(restored[0]),
                err_msg=f'replay entry {i} state mismatch')
        restored_main = agent.mainModel.get_weights()
        for i, (orig_w, rest_w) in enumerate(zip(orig_main_weights, restored_main)):
            np.testing.assert_array_equal(orig_w, rest_w, err_msg=f'main weight {i} mismatch')
        restored_target = agent.targetModel.get_weights()
        for i, (orig_w, rest_w) in enumerate(zip(orig_target_weights, restored_target)):
            np.testing.assert_array_equal(orig_w, rest_w, err_msg=f'target weight {i} mismatch')

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dqn_checkpoint_periodic_in_runDQN(self):
        """Verify periodic checkpointing is triggered during runDQN."""
        import unittest.mock as mock

        train_dqn = self._get_train_dqn()
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams
        envSettings['ENVTYPE'] = '1s-d'
        envSettings['MODELNAME'] = 'unittest_ckptperiodic'
        envSettings['RENDER'] = False
        hyParams['NGAMES'] = 3
        hyParams['NAGENTSTEPS'] = 2
        hyParams['REPLAYMEMORYSIZE'] = 100
        hyParams['REPLAYMEMORYSIZEMIN'] = 3
        hyParams['MINIBATCHSIZE'] = 3
        hyParams['NHIDDENNODES'] = [5] * 2
        hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
        hyParams['DROPOUTS'] = [0.0] * 2
        hyParams['CROSSVALIDATEEVERY'] = 10
        hyParams['NGAMESCROSSVALIDATED'] = 1
        hyParams['CHECKPOINTEVERY'] = 2

        from flopyarcade import FloPyAgent, FloPyEnv
        env = FloPyEnv(
            envSettings['ENVTYPE'], None, None,
            envSettings['MODELNAME'], flagRender=False,
            NAGENTSTEPS=2, OBSPREP='perceptron')
        agent = FloPyAgent(
            env.observationsVectorNormalized,
            env.actionSpace, hyParams, envSettings, 'DQN')

        with mock.patch.object(agent, 'save_dqn_checkpoint', wraps=agent.save_dqn_checkpoint) as mock_save:
            agent.runDQN(env)

        self.assertEqual(mock_save.call_count, 1,
                         f'Expected 1 checkpoint, got {mock_save.call_count}')



if __name__ == '__main__':
    unittest.main(exit=False)

    # detected bug: with gameSettings['NAGENTSTEPS'] = 2 reward may be gained from lost games

    # deleting all test-related files, except unittestAgent.h5
    wrkspc = dirname(abspath(__file__))
    unittestFilesAndFolders = glob(join(wrkspc, 'models', 'unittest*'))
    for obj in unittestFilesAndFolders:
        if isdir(obj):
            rmtree(obj)
        elif isfile(obj):
            remove(obj)
    unittestFilesAndFolders = glob(join(wrkspc, 'temp', 'unittest*'))
    for obj in unittestFilesAndFolders:
        if isdir(obj):
            rmtree(obj)
        elif isfile(obj):
            remove(obj)