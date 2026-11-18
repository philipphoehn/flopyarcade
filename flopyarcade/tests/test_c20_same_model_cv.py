"""C20: Verify that training and cross-validation use the same model.

Ensures that FloPyAgent.crossvalidateDQN operates on the identical
self.mainModel instance that is used during training, preventing
accidental model divergence.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def _make_dqn_agent():
    """Create a minimal FloPyAgent DQN instance using train_dqn settings."""
    from flopyarcade import FloPyAgent, FloPyEnv
    import flopyarcade.examples.train_dqn as train_dqn

    envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams
    envSettings['ENVTYPE'] = '1s-d'
    envSettings['MODELNAME'] = 'c20_test'
    envSettings['RENDER'] = False
    hyParams['NHIDDENNODES'] = [5] * 2
    hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
    hyParams['DROPOUTS'] = [0.0] * 2
    hyParams['REPLAYMEMORYSIZE'] = 100
    hyParams['EPSILONINITIAL'] = 1.0
    hyParams['NAGENTSTEPS'] = 3
    hyParams['NGAMESCROSSVALIDATED'] = 1

    env = FloPyEnv(
        envSettings['ENVTYPE'], None, None,
        envSettings['MODELNAME'], flagRender=False,
        NAGENTSTEPS=3, OBSPREP='perceptron')
    agent = FloPyAgent(
        env.observationsVectorNormalized,
        env.actionSpace, hyParams, envSettings, 'DQN')
    return agent, env


def test_cv_uses_same_main_model():
    """crossvalidateDQN must reference the same mainModel used in training."""
    agent, env = _make_dqn_agent()
    original_model = agent.mainModel

    assert original_model is not None
    assert hasattr(original_model, 'predict')

    # Patch getqsGivenAgentModel to track which model is passed
    call_log = []
    orig_getqs = agent.getqsGivenAgentModel

    def tracking_getqs(model, obs):
        call_log.append(id(model))
        return np.random.rand(1, len(agent.actionSpace)).astype(np.float32)

    with patch.object(agent, 'getqsGivenAgentModel', side_effect=tracking_getqs):
        agent.seedsCV = np.array([42])
        agent.crossvalidateDQN(env)

    assert len(call_log) > 0, "crossvalidateDQN did not query the model"
    assert all(mid == id(original_model) for mid in call_log), \
        "CV used a different model instance than training"


def test_model_architecture_consistent():
    """The mainModel and targetModel have identical architecture."""
    agent, _ = _make_dqn_agent()

    main_layers = len(agent.mainModel.layers)
    target_layers = len(agent.targetModel.layers)
    assert main_layers == target_layers, \
        f"mainModel has {main_layers} layers, targetModel has {target_layers}"

    # Verify output shape matches action space size
    out_shape = agent.mainModel.output_shape[-1]
    assert out_shape == len(agent.actionSpace), \
        f"Output shape {out_shape} != action space size {len(agent.actionSpace)}"


def test_main_model_not_replaced_during_cv():
    """crossvalidateDQN must not reassign self.mainModel."""
    agent, env = _make_dqn_agent()
    model_id_before = id(agent.mainModel)

    with patch.object(agent.mainModel, 'predict',
                      return_value=np.random.rand(1, len(agent.actionSpace)).astype(np.float32)):
        agent.seedsCV = np.array([42])
        agent.crossvalidateDQN(env)

    assert id(agent.mainModel) == model_id_before, \
        "mainModel was replaced during cross-validation"


def test_cv_does_not_modify_weights():
    """crossvalidateDQN must not update model weights (it only evaluates)."""
    agent, env = _make_dqn_agent()
    weights_before = [w.copy() for w in agent.mainModel.get_weights()]

    with patch.object(agent.mainModel, 'predict',
                      return_value=np.random.rand(1, len(agent.actionSpace)).astype(np.float32)):
        agent.seedsCV = np.array([123])
        agent.crossvalidateDQN(env)

    weights_after = agent.mainModel.get_weights()
    for i, (w_before, w_after) in enumerate(zip(weights_before, weights_after)):
        np.testing.assert_array_equal(w_before, w_after, 
            f"Weight tensor {i} changed during CV")


def test_cv_results_computed():
    """crossvalidateDQN computes mean/min/max reward statistics."""
    agent, env = _make_dqn_agent()

    with patch.object(agent.mainModel, 'predict',
                      return_value=np.random.rand(1, len(agent.actionSpace)).astype(np.float32)):
        agent.seedsCV = np.array([42])
        agent.crossvalidateDQN(env)

    assert hasattr(agent, 'average_rewardCV')
    assert hasattr(agent, 'min_rewardCV')
    assert hasattr(agent, 'max_rewardCV')
    assert agent.min_rewardCV <= agent.average_rewardCV <= agent.max_rewardCV
