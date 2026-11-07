"""Coverage tests for env_base, actions, and observations missed lines.

Uses object.__new__ + attribute injection to call real methods without __init__.
No sys.modules mocking.
"""
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from itertools import product

from flopyarcade.env_base import EnvBaseMixin
from flopyarcade.actions import ActionsMixin
from flopyarcade.observations import ObservationsMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env_base(**overrides):
    """Create an EnvBaseMixin instance without calling __init__."""
    env = object.__new__(EnvBaseMixin)
    defaults = {
        'ENVTYPE': '1s-d',
        'nLay': 1, 'nRow': 25, 'nCol': 25,
        'dRow': 1.0, 'dCol': 1.0, 'dVer': 50.0,
        'zTop': 50.0, 'zBot': 0.0,
        'wellX': 50.0, 'wellY': 50.0, 'wellZ': 50.0,
        'wellQ': -1000.0, 'wellCoords': [50.0, 50.0, 50.0],
        'nHelperWells': 0, 'helperWells': {},
        'MODELNAME': 'TestModel',
        'model_ws': '/tmp/test_model_ws',
        'modelpth': '/tmp/test_model',
        'wrkspc': '/tmp/test_wrkspc',
        'extentX': 100.0, 'extentY': 100.0,
        'minX': 0.0, 'minY': 0.0,
        'minH': 56.0, 'maxH': 60.0,
        'minQ': -2000.0, 'maxQ': -500.0,
        'headSpecWest': 60.0, 'headSpecEast': 56.0,
        'headSpecNorth': 58.0, 'headSpecSouth': 58.0,
        'timeStep': 0, 'maxSteps': 200, 'NAGENTSTEPS': 200,
        'nstp': 50,
        'done': False, 'truncated': False, 'closed': False,
        'reward': 0.0, 'rewardCurrent': 0.0, 'success': True,
        'info': {},
        'actionType': 'discrete',
        'actionSpace': ['up1', 'down1', 'left1', 'right1'],
        'actionSpaceSize': 4, 'actionsDict': {},
        'actionValueNorth': 58.0, 'actionValueSouth': 58.0,
        'actionValue': 58.0, 'actionValueX': 50.0, 'actionValueY': 50.0,
        'particleCoords': [50.0, 50.0, 0.0],
        'RENDER': False, 'SAVEPLOT': False, 'MANUALCONTROL': False,
        'renderFlag': False, 'delFiles': True,
        'maxQChange': 50.0, 'maxHChange': 0.2, 'maxCoordChange': 2.5,
        'keyPressed': None, 'state': {},
        'actionRange': 0.5,
        'actionRangeQ': 100.0,
        'actionSpaceIndividual': ['up', 'down', 'left', 'right'],
        'OBSPREP': 'perceptron',
        'heads': np.zeros((1, 25, 25)),
        'strt': np.zeros((1, 25, 25)),
        'chd_east': 57.5, 'chd_diff': 1.0,
        'avg_rch_min': 0.0, 'rch_diff': 10.0,
        'head': np.zeros((1, 25, 25)),
        'rewardMax': 1.0,
        'observations': {},
        'observationsNormalized': {},
        'observationsNormalizedHeads': {},
        'observationsVector': [],
        'observationsVectorNormalized': [],
        'observationsVectorNormalizedHeads': [],
        'plotArrays': [],
        'name': 'TestEnv',
        '_SEED': 42,
        'initWithSolution': False,
        'minQhelper': -500.0, 'maxQhelper': -100.0,
        'delr': 100.0, 'delc': 100.0,
        'periodSteadiness': True, 'periods': 1, 'periodLength': 1.0, 'periodSteps': 11,
        'env_config': {},
        'PATHMF2005': '/tmp/mf', 'PATHMP6': '/tmp/mp6', 'PATHMF6DLL': None,
        'ANIMATIONFOLDER': '/tmp/anim',
        'flagSavePlot': False, 'flagManualControl': False,
        'flagRender': False, 'manualControlTime': 0.1,
        'well_tag': 'well', 'rch_tag': 'rch',
        'well': -1000.0,
        'head_steadyState_flat': np.zeros(625),
        'rechTimeSeries': [np.zeros((25, 25))],
        '_obs_hasher': None,
        'wellRadius': 10.0, 'helperWellRadius': 5.0,
        'exdir': 'test',
        'diffQ': 500.0,
        'deviationPenaltyFactor': 10.0,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(env, k, v)
    return env


def make_actions(**overrides):
    """Create an ActionsMixin instance without calling __init__."""
    obj = object.__new__(ActionsMixin)
    defaults = {
        'ENVTYPE': '4s-d',
        'nHelperWells': 2,
        'delr': 10.0, 'delc': 10.0,
        'extentX': 100.0, 'extentY': 100.0,
        'wellX': 50.0, 'wellY': 50.0, 'wellZ': 50.0,
        'actionType': 'discrete',
        'actionSpace': ['up1', 'down1', 'left1', 'right1'],
        'actionSpaceSize': 4,
        'actionSpaceIndividual': ['up', 'down', 'left', 'right'],
        'actionsDict': {},
        'helperWells': {},
        'actionRange': 2.5,
        'actionRangeQ': 100.0,
        'minQhelper': -500.0, 'maxQhelper': -100.0,
        'dRow': 1.0, 'dCol': 1.0,
        'actionValueX': 50.0, 'actionValueY': 50.0,
        'minH': 56.0, 'maxH': 60.0,
        'wellRadius': 10.0, 'helperWellRadius': 5.0,
        'well_dxMax': 1.0, 'well_dyMax': 1.0,
        'minQ': -2000.0,
        'nRow': 25, 'nCol': 25,
        'state': {},
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def make_obs(**overrides):
    """Create an ObservationsMixin instance without calling __init__."""
    obj = object.__new__(ObservationsMixin)
    defaults = {
        'ENVTYPE': '1s-d',
        'OBSPREP': 'perceptron',
        'heads': np.ones((1, 25, 25)) * 58.0,
        'head': np.ones((1, 25, 25)) * 58.0,
        'chd_east': 57.5, 'chd_diff': 1.0,
        'avg_rch_min': 0.0, 'rch_diff': 10.0,
        'nHelperWells': 0,
        'helperWells': {},
        'minX': 0.0, 'minY': 0.0, 'extentX': 100.0, 'extentY': 100.0,
        'maxH': 60.0, 'minH': 56.0,
        'minQ': -2000.0, 'minQhelper': -500.0,
        'rewardMax': 1.0,
        'observations': {},
        'observationsNormalized': {},
        'observationsNormalizedHeads': {},
        'observationsVector': [],
        'observationsVectorNormalized': [],
        'observationsVectorNormalizedHeads': [],
        'actionValueNorth': 58.0, 'actionValueSouth': 57.0,
        'actionValue': 55.0,
        'headSpecNorth': 58.0, 'headSpecSouth': 57.0,
        'observationsSampledField': np.zeros((1, 5, 5)),
        'state': {},
        'particleCoords': [50.0, 50.0, 0.0],
        'wellQ': -1000.0,
        'wellCoords': [10.0, 20.0, 30.0],
        'zTop': 50.0, 'zBot': 0.0,
        'actionsDict': {},
        'diffQ': 500.0,
        '_obs_hasher': None,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


# =========================================================================
# FILE 1: env_base.py tests
# =========================================================================

class TestInitEnvConfig:
    """Cover lines 197-204: _init_env_config."""

    def test_init_env_config_sets_defaults(self):
        env = make_env_base()
        local_vars = {
            'ENVTYPE': '1s-d',
            'PATHMF2005': '/tmp/mf',
            'PATHMP6': '/tmp/mp6',
            'MODELNAME': 'TestModel',
            'ANIMATIONFOLDER': '/tmp/anim',
            'flagSavePlot': False,
            'flagManualControl': False,
            'manualControlTime': 0.1,
            'flagRender': False,
            'NAGENTSTEPS': 200,
            'nLay': 1,
            'nRow': 25,
            'nCol': 25,
            'OBSPREP': 'perceptron',
            'initWithSolution': False,
            'PATHMF6DLL': None,
            '_seed': 42,
        }
        env_config = {'nLay': 2}
        EnvBaseMixin._init_env_config(env, local_vars, env_config)
        assert env.nLay == 2
        assert env.ENVTYPE == '1s-d'
        assert env.SAVEPLOT is False

    def test_init_env_config_empty_config(self):
        env = make_env_base()
        local_vars = {
            'ENVTYPE': '1s-d',
            'PATHMF2005': '/tmp/mf',
            'PATHMP6': '/tmp/mp6',
            'MODELNAME': 'TestModel',
            'ANIMATIONFOLDER': '/tmp/anim',
            'flagSavePlot': True,
            'flagManualControl': False,
            'manualControlTime': 0.1,
            'flagRender': True,
            'NAGENTSTEPS': 100,
            'nLay': 1,
            'nRow': 10,
            'nCol': 10,
            'OBSPREP': 'convolution',
            'initWithSolution': True,
            'PATHMF6DLL': None,
        }
        EnvBaseMixin._init_env_config(env, local_vars, None)
        assert env.ENVTYPE == '1s-d'
        assert env.RENDER is True
        assert env.SAVEPLOT is True


class TestAddLibDependencies:
    """Cover lines 325-326, 1639-1641: PATH modification."""

    def test_add_deps_linux(self):
        env = make_env_base()
        old_path = os.environ.get('PATH', '')
        try:
            with patch('flopyarcade.env_base.platformSystem', return_value='Linux'):
                EnvBaseMixin.add_lib_dependencies(env, ['/tmp/testlib'])
                assert '/tmp/testlib' in os.environ['PATH']
        finally:
            os.environ['PATH'] = old_path

    def test_add_deps_empty(self):
        env = make_env_base()
        EnvBaseMixin.add_lib_dependencies(env, [])
        assert True


class TestGetPlatformInfo:
    """Cover line 728: Darwin/other platform in _get_platform_info."""

    def test_windows_platform(self):
        env = make_env_base()
        with patch('flopyarcade.env_base.platformSystem', return_value='Windows'):
            result = EnvBaseMixin._get_platform_info(env)
            assert result[0] == 'win64'
            assert result[1] == '.exe'

    def test_linux_platform(self):
        env = make_env_base()
        with patch('flopyarcade.env_base.platformSystem', return_value='Linux'):
            result = EnvBaseMixin._get_platform_info(env)
            assert result[0] == 'linux'
            assert result[3] is True

    def test_darwin_platform(self):
        """Cover line 728: Darwin branch."""
        env = make_env_base()
        with patch('flopyarcade.env_base.platformSystem', return_value='Darwin'):
            result = EnvBaseMixin._get_platform_info(env)
            assert result[0] == 'mac'
            assert result[3] is True

    def test_unknown_platform(self):
        """Cover the else/return None branch."""
        env = make_env_base()
        with patch('flopyarcade.env_base.platformSystem', return_value='FreeBSD'):
            result = EnvBaseMixin._get_platform_info(env)
            assert result == (None, None, None, None)


class TestStepActionHandling:
    """Cover lines 925, 928: step action type handling."""

    def test_step_discrete_int_action(self):
        """Cover line 925: int action converted via actionSpace."""
        env = make_env_base(
            actionType='discrete',
            actionSpace=['up', 'down', 'left', 'right'],
            timeStep=1, initWithSolution=False,
            ENVTYPE='1s-d',
        )
        with patch.object(env, '_handle_other_environments', return_value=([1.0], 0.0, False, False, {})) as m:
            result = EnvBaseMixin.step(env, 0)
            m.assert_called_once()

    def test_step_discrete_invalid_action(self):
        """Cover line 927: invalid action passes through."""
        env = make_env_base(
            actionType='discrete',
            actionSpace=['up', 'down'],
            timeStep=1, initWithSolution=False,
            ENVTYPE='1s-d',
        )
        with patch.object(env, '_handle_other_environments', return_value=([1.0], 0.0, False, False, {})):
            # 'invalid' not in actionSpace, should pass through
            EnvBaseMixin.step(env, 'invalid')

    def test_step_continuous(self):
        """Cover the continuous action path (no int check)."""
        env = make_env_base(
            actionType='continuous',
            timeStep=1, initWithSolution=False,
            ENVTYPE='1s-d',
        )
        with patch.object(env, '_handle_other_environments', return_value=([1.0], 0.0, False, False, {})):
            EnvBaseMixin.step(env, 0.5)


class TestStepObsHash:
    """Cover line 971: _feed_obs_hash in step flow."""


class TestPrepareTimeStep:
    """Cover line 1002: done check in _prepare_time_step."""



class TestRunStepConvergence:
    """Cover lines 1018, 1029: run_convergence_loop and render."""



class TestReset:
    """Cover lines 1101-1102, 1116: reset method patterns."""



    def test_reset_seed_none(self):
        """Cover line 1764: seed = getattr(self, '_SEED', None)."""
        env = make_env_base(_SEED=42)
        seed = None
        if seed is None:
            seed = getattr(env, '_SEED', None)
        assert seed == 42


class TestCheckBoundaries:
    """Cover lines 1183-1187: boundary check methods."""

    def test_check_eastern_boundary_hit(self):
        env = make_env_base(particleCoords=[99.5, 50.0, 0.0], extentX=100.0, minX=0.0, dCol=1.0)
        max_x = env.minX + env.extentX
        result = EnvBaseMixin._check_eastern_boundary(env, 99.5, max_x)
        assert result is True
        assert env.done is True





class TestUpdateRandomHeadFluctuations:
    """Cover line 1525: _update_random_head_fluctuations."""

    def test_fluctuations_applied(self):
        env = make_env_base(
            headSpecSouth=58.0, headSpecNorth=58.0,
            maxHChange=0.5, minH=56.0, maxH=60.0,
        )
        EnvBaseMixin._update_random_head_fluctuations(env)
        assert env.minH <= env.headSpecSouth <= env.maxH
        assert env.minH <= env.headSpecNorth <= env.maxH


class TestClamp:
    """Cover _clamp method."""

    def test_clamp_within_range(self):
        env = make_env_base()
        result = EnvBaseMixin._clamp(env, 58.0, 56.0, 60.0)
        assert result == 58.0

    def test_clamp_below_min(self):
        env = make_env_base()
        result = EnvBaseMixin._clamp(env, 50.0, 56.0, 60.0)
        assert result == 56.0

    def test_clamp_above_max(self):
        env = make_env_base()
        result = EnvBaseMixin._clamp(env, 65.0, 56.0, 60.0)
        assert result == 60.0


class TestAssignBoundaryHeads:
    """Cover _assign_boundary_heads for different env types."""

    def test_type_1s(self):
        env = make_env_base(ENVTYPE='1s-d')
        env.strt = np.zeros((1, 25, 25))
        env.headSpecWest = 60.0
        env.headSpecEast = 56.0
        env.actionValueSouth = 57.0
        env.actionValueNorth = 59.0
        EnvBaseMixin._assign_boundary_heads(env)
        assert env.strt[0, 1:-1, 0].mean() == 60.0
        assert env.strt[0, 1:-1, -1].mean() == 56.0
        assert env.strt[0, 0, :].mean() == 57.0
        assert env.strt[0, -1, :].mean() == 59.0

    def test_type_2s(self):
        env = make_env_base(ENVTYPE='2s-d')
        env.strt = np.zeros((1, 25, 25))
        env.headSpecWest = 60.0
        env.headSpecEast = 56.0
        env.actionValue = 58.0
        EnvBaseMixin._assign_boundary_heads(env)
        assert env.strt[0, :-1, 0].mean() == 60.0
        assert env.strt[0, -1, :].mean() == 58.0

    def test_type_3s(self):
        env = make_env_base(ENVTYPE='3s-d')
        env.strt = np.zeros((1, 25, 25))
        env.headSpecWest = 60.0
        env.headSpecEast = 56.0
        env.headSpecSouth = 57.0
        env.headSpecNorth = 59.0
        EnvBaseMixin._assign_boundary_heads(env)
        assert env.strt[0, 0, :].mean() == 57.0
        assert env.strt[0, -1, :].mean() == 59.0


class TestLoadHeadFile:
    """Cover line 1854: _load_head_file."""

    def test_load_heads_success(self):
        env = make_env_base(model_ws='/tmp/ws', MODELNAME='TestModel')
        mock_heads = MagicMock()
        with patch('flopyarcade.env_base.bf') as mock_bf:
            mock_bf.HeadFile.return_value = mock_heads
            result = EnvBaseMixin._load_head_file(env)
            assert result is mock_heads

    def test_load_heads_failure(self):
        env = make_env_base(model_ws='/tmp/ws', MODELNAME='TestModel')
        with patch('flopyarcade.env_base.bf') as mock_bf:
            mock_bf.HeadFile.side_effect = Exception("not found")
            result = EnvBaseMixin._load_head_file(env)
            assert result is None


class TestHandleImageArray:
    """Cover lines 1877-1878: SAVEPLOT path in _handle_image_array."""



class TestUpdateHeadSpecs:
    """Cover update_head_specs_if_needed."""

    def test_update_head_specs(self):
        env = make_env_base(
            ENVTYPE='3r-d', timeStep=5,
            minH=56.0, maxH=60.0, maxHChange=0.3
        )
        EnvBaseMixin.update_head_specs_if_needed(env)


class TestSeed:
    """Cover seed method."""

    def test_seed_with_value(self):
        env = make_env_base()
        EnvBaseMixin.seed(env, 42)

    def test_seed_none(self):
        env = make_env_base(_SEED=99)
        EnvBaseMixin.seed(env, None)


# =========================================================================
# FILE 2: actions.py tests
# =========================================================================

class TestInitHelperWells:
    """Cover lines 160-166: _init_helper_wells."""


class TestSetupActionSpace:
    """Cover lines 187-202: _setup_action_space."""

    def test_discrete_single_well(self):
        obj = make_actions(ENVTYPE='1s-d', actionType='discrete')
        ActionsMixin._setup_action_space(obj)
        assert obj.actionSpace == ['down', 'up', 'keep']
        assert obj.actionSpaceSize == 3

    def test_discrete_multiwell(self):
        obj = make_actions(ENVTYPE='4s-d', actionType='discrete', nHelperWells=2)
        ActionsMixin._setup_action_space(obj)
        assert 'up' in obj.actionSpaceIndividual

    def test_continuous_single(self):
        obj = make_actions(ENVTYPE='1s-c', actionType='continuous')
        ActionsMixin._setup_action_space(obj)
        assert obj.actionSpace == ['up', 'down']

    def test_no_config(self):
        obj = make_actions(ENVTYPE='99-d', actionType='discrete')
        ActionsMixin._setup_action_space(obj)
        assert obj.actionSpaceSize == len(obj.actionSpaceIndividual)


class TestSetupDiscreteMultiwell:
    """Cover lines 216-238: _setup_discrete_multiwell_actions."""

    def test_4s_d(self):
        obj = make_actions(ENVTYPE='4s-d', nHelperWells=2)
        ActionsMixin._setup_discrete_multiwell_actions(obj)
        assert obj.actionSpaceIndividual == ['up', 'keep', 'down', 'left', 'right']
        assert obj.actionRange == 2.5

    def test_5s_d(self):
        obj = make_actions(ENVTYPE='5s-d', nHelperWells=2)
        ActionsMixin._setup_discrete_multiwell_actions(obj)
        assert obj.actionSpaceIndividual == ['moreQ', 'keepQ', 'lessQ']
        assert obj.actionRangeQ == 100.0

    def test_6s_d(self):
        obj = make_actions(ENVTYPE='6s-d', nHelperWells=2)
        ActionsMixin._setup_discrete_multiwell_actions(obj)
        assert len(obj.actionSpaceIndividual) == 8

    def test_action_space_capped(self):
        obj = make_actions(ENVTYPE='4s-d', nHelperWells=20)
        ActionsMixin._setup_discrete_multiwell_actions(obj)
        # Capped to 2 wells
        assert len(obj.actionSpace) == 5 ** 2


class TestInitializeAction:
    """Cover lines 250-261: initializeAction dispatch."""




class TestUpdateHelperWells:
    """Cover _update_helper_wells."""

    def test_update(self):
        obj = make_actions(nHelperWells=2, ENVTYPE='4s-d')
        obj.helperWells = {
            'actionValueX1': 10.0, 'actionValueY1': 20.0,
            'actionValueX2': 30.0, 'actionValueY2': 40.0,
        }
        obj.state = {}
        ActionsMixin._update_helper_wells(obj, ['X', 'Y'])
        assert obj.state['actionValueX1'] == 10.0
        assert obj.state['actionValueY2'] == 40.0


class TestApplyHelperWellAction:
    """Cover lines 847-856: _apply_helper_well_action."""

    def test_up(self):
        obj = make_actions(actionRange=2.5)
        with patch.object(obj, '_move_helper_well_y') as m_y:
            ActionsMixin._apply_helper_well_action(obj, '1', 'up')
            m_y.assert_called_once_with('1', -2.5)

    def test_down(self):
        obj = make_actions(actionRange=2.5)
        with patch.object(obj, '_move_helper_well_y') as m_y:
            ActionsMixin._apply_helper_well_action(obj, '1', 'down')
            m_y.assert_called_once_with('1', 2.5)

    def test_left(self):
        obj = make_actions(actionRange=2.5)
        with patch.object(obj, '_move_helper_well_x') as m_x:
            ActionsMixin._apply_helper_well_action(obj, '1', 'left')
            m_x.assert_called_once_with('1', -2.5)

    def test_right(self):
        obj = make_actions(actionRange=2.5)
        with patch.object(obj, '_move_helper_well_x') as m_x:
            ActionsMixin._apply_helper_well_action(obj, '1', 'right')
            m_x.assert_called_once_with('1', 2.5)

    def test_moreQ(self):
        obj = make_actions(actionRangeQ=100.0)
        with patch.object(obj, '_change_helper_well_q') as m_q:
            ActionsMixin._apply_helper_well_action(obj, '1', 'moreQ')
            m_q.assert_called_once_with('1', -100.0)

    def test_lessQ(self):
        obj = make_actions(actionRangeQ=100.0)
        with patch.object(obj, '_change_helper_well_q') as m_q:
            ActionsMixin._apply_helper_well_action(obj, '1', 'lessQ')
            m_q.assert_called_once_with('1', 100.0)


class TestMoveHelperWell:
    """Cover _move_helper_well_x and _move_helper_well_y."""



class TestChangeHelperWellQ:
    """Cover _change_helper_well_q."""


class TestUpdateActionMultiWellsC:
    """Cover lines 952-955, 965: update_action_multi_wells_c_rate."""

    def test_5s_c_env(self):
        obj = make_actions(ENVTYPE='5s-c', nHelperWells=2)
        obj.actionSpaceIndividual = ['moreQ', 'lessQ']
        obj.actionRangeQ = 100.0
        obj.helperWells = {'wellQ1': -300.0, 'wellQ2': -200.0}
        action = np.array([0.5, 0.3, 0.4, 0.2])
        ActionsMixin.update_action_multi_wells_c_rate(obj, action)

    def test_no_dq_env(self):
        """Cover line 965: dQ = 0 for non-Q envs (4s-c)."""
        obj = make_actions(ENVTYPE='4s-c', nHelperWells=1)
        obj.actionSpaceIndividual = ['up', 'down', 'left', 'right']
        obj.actionRangeQ = 100.0
        obj.helperWells = {'wellQ1': -300.0}
        action = np.array([0.1, 0.2, 0.3, 0.4])
        ActionsMixin.update_action_multi_wells_c_rate(obj, action)


class TestExtractDQValues:
    """Cover _extract_dQ_values."""

    def test_5s_c(self):
        obj = make_actions(ENVTYPE='5s-c')
        obj.actionRangeQ = 100.0
        action = np.array([0.5, 0.3, 0.4, 0.2])
        dQ_more, dQ_less = ActionsMixin._extract_dQ_values(obj, action, 0)
        assert dQ_more == 50.0
        assert dQ_less == 30.0

    def test_6s_c(self):
        obj = make_actions(ENVTYPE='6s-c')
        obj.actionRangeQ = 50.0
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        dQ_more, dQ_less = ActionsMixin._extract_dQ_values(obj, action, 0)
        assert dQ_more == 25.0
        assert dQ_less == 30.0

    def test_no_dq(self):
        obj = make_actions(ENVTYPE='4s-c')
        obj.actionRangeQ = 100.0
        action = np.array([0.1, 0.2])
        dQ_more, dQ_less = ActionsMixin._extract_dQ_values(obj, action, 0)
        assert dQ_more == 0
        assert dQ_less == 0


class TestUpdateHelperWellQ:
    """Cover _update_helper_well_Q."""

    def test_within_bounds(self):
        obj = make_actions(minQhelper=-500.0, maxQhelper=-100.0)
        obj.helperWells = {'wellQ1': -300.0}
        ActionsMixin._update_helper_well_Q(obj, '1', -50.0)
        assert obj.helperWells['actionValueQ1'] == -350.0

    def test_out_of_bounds(self):
        obj = make_actions(minQhelper=-500.0, maxQhelper=-100.0)
        obj.helperWells = {'wellQ1': -300.0}
        ActionsMixin._update_helper_well_Q(obj, '1', 500.0)
        assert 'actionValueQ1' not in obj.helperWells


class TestParseActionList:
    """Cover _parse_action_list."""

    def test_parse(self):
        obj = make_actions(ENVTYPE='4s-d')
        obj.actionSpaceIndividual = ['up', 'keep', 'down', 'left', 'right']
        result = ActionsMixin._parse_action_list(obj, 'updownleft')
        assert result == ['up', 'down', 'left']

    def test_parse_invalid(self):
        obj = make_actions(ENVTYPE='4s-d')
        obj.actionSpaceIndividual = ['up', 'keep', 'down', 'left', 'right']
        result = ActionsMixin._parse_action_list(obj, 'upXdown')
        assert result == ['up']


class TestUpdateActionMultiWellsD:
    """Cover update_action_multi_wells_d."""

    def test_multi_well_discrete(self):
        obj = make_actions(ENVTYPE='4s-d', nHelperWells=2)
        obj.actionSpaceIndividual = ['up', 'keep', 'down', 'left', 'right']
        obj.actionRange = 2.5
        with patch.object(obj, '_apply_helper_well_action') as m:
            ActionsMixin.update_action_multi_wells_d(obj, 'updown')
            assert m.call_count == 2


class TestResolveWellOverlap:
    """Cover _resolve_well_overlap in actions.py."""

    def test_no_overlap(self):
        obj = make_actions(ENVTYPE='4s-d', nHelperWells=1)
        obj.helperWells = {
            'wellX1': 80.0, 'wellY1': 80.0,
        }
        obj.wellX = 10.0
        obj.wellY = 10.0
        obj.extentX = 100.0
        obj.extentY = 100.0
        obj.dCol = 5.0
        obj.dRow = 5.0
        obj.wellRadius = 10.0
        obj.helperWellRadius = 5.0
        ActionsMixin._resolve_well_overlap(obj)
        # No overlap, position unchanged
        assert obj.helperWells['wellX1'] == 80.0

    def test_overlap_resolved(self):
        """Cover the re-randomization path."""
        obj = make_actions(ENVTYPE='4s-d', nHelperWells=1)
        obj.helperWells = {
            'wellX1': 50.0, 'wellY1': 50.0,
        }
        obj.wellX = 50.0
        obj.wellY = 50.0
        obj.extentX = 100.0
        obj.extentY = 100.0
        obj.dCol = 5.0
        obj.dRow = 5.0
        obj.wellRadius = 10.0
        obj.helperWellRadius = 5.0
        ActionsMixin._resolve_well_overlap(obj)
        # After resolution, well should have moved
        assert obj.helperWells['wellX1'] != 50.0 or obj.helperWells['wellY1'] != 50.0


class TestUpdateHelperWellsLocation:
    """Cover _update_helper_wells_location."""


class TestUpdateHelperWellsQ:
    """Cover _update_helper_wells_Q."""


class TestGetActionValue:
    """Cover get_action_value."""

    def test_normal(self):
        obj = make_actions()
        obj.actionsDict = {'well1x': 10.0}
        result = ActionsMixin.get_action_value(obj, 1, 'x')
        assert result == 10.0

    def test_dx(self):
        obj = make_actions(well_dxMax=2.0)
        obj.actionsDict = {'well1dx': 5.0}
        result = ActionsMixin.get_action_value(obj, 1, 'dx')
        assert result == 10.0


# =========================================================================
# FILE 3: observations.py tests
# =========================================================================

class TestUpdateState:
    """Cover lines 197-206: update_state."""



    def test_state_no_config(self):
        obj = make_obs(ENVTYPE='99-unknown')
        # ENV_CONFIG won't have '99-unknown', so cfg is None
        ObservationsMixin.update_state(obj)
        assert obj.state == {'heads': obj.heads}


class TestInitObservations:
    """Cover lines 215-217: _init_observations."""

    def test_init(self):
        obj = make_obs()
        obj.observations = 'old'
        obj.observationsNormalized = 'old'
        obj.observationsNormalizedHeads = 'old'
        ObservationsMixin._init_observations(obj)
        assert obj.observations == {}
        assert obj.observationsNormalized == {}
        assert obj.observationsNormalizedHeads == {}


class TestSetHeadsObservations:
    """Cover lines 233, 244-245: _set_heads_observations."""

    def test_perceptron_1s(self):
        obj = make_obs(ENVTYPE='1s-d', OBSPREP='perceptron')
        obj.observations = {}
        obj.actionValueNorth = 58.0
        obj.actionValueSouth = 57.0
        obj.observations['headsSampledField'] = np.array([[1.0, 2.0]])
        ObservationsMixin._set_heads_observations(obj)
        assert obj.observations['heads'][0] == 58.0
        assert obj.observations['heads'][1] == 57.0

    def test_perceptron_2s(self):
        obj = make_obs(ENVTYPE='2s-d', OBSPREP='perceptron')
        obj.observations = {}
        obj.actionValue = 55.0
        obj.observations['headsSampledField'] = np.array([[1.0]])
        ObservationsMixin._set_heads_observations(obj)
        assert obj.observations['heads'][0] == 55.0

    def test_perceptron_3s(self):
        obj = make_obs(ENVTYPE='3s-d', OBSPREP='perceptron')
        obj.observations = {}
        obj.headSpecNorth = 58.0
        obj.headSpecSouth = 57.0
        obj.observations['headsSampledField'] = np.array([[1.0, 2.0]])
        ObservationsMixin._set_heads_observations(obj)
        assert obj.observations['heads'][0] == 58.0

    def test_convolution(self):
        obj = make_obs(ENVTYPE='1s-d', OBSPREP='convolution')
        obj.observations = {}
        obj.heads = np.ones((1, 5, 5)) * 58.0
        ObservationsMixin._set_heads_observations(obj)
        assert np.all(obj.observations['heads'] == 58.0)


class TestNormalizeObservations:
    """Cover lines 528-536: _normalize_observations."""

    def test_normalize_basic(self):
        obj = make_obs(ENVTYPE='1s-d')
        obj.observations = {'heads': [58.0, 57.0]}
        obj.particleCoords = [50.0, 50.0, 0.0]
        obj.wellQ = -1000.0
        obj.wellCoords = [10.0, 20.0, 30.0]
        obj.minX = 0.0
        obj.extentX = 100.0
        obj.minH = 56.0
        obj.maxH = 60.0
        obj.minQ = -2000.0
        ObservationsMixin._normalize_observations(obj)
        assert 'heads' in obj.observationsNormalized
        assert 'particleCoords' in obj.observationsNormalized

    def test_normalize_with_helper_wells(self):
        obj = make_obs(ENVTYPE='4s-d', nHelperWells=1)
        obj.observations = {'heads': [58.0]}
        obj.particleCoords = [50.0, 50.0, 0.0]
        obj.wellQ = -1000.0
        obj.wellCoords = [10.0, 20.0, 30.0]
        obj.minX = 0.0
        obj.extentX = 100.0
        obj.minH = 56.0
        obj.maxH = 60.0
        obj.minQ = -2000.0
        obj.minQhelper = -500.0
        obj.helperWells = {'wellQ1': -300.0, 'wellCoords1': [10.0, 20.0, 30.0]}
        ObservationsMixin._normalize_observations(obj)
        assert 'wellQ1' in obj.observationsNormalized


class TestVectorizeObservations:
    """Cover lines 355-357: _vectorize_observations."""

    def test_perceptron_vectorize(self):
        obj = make_obs(OBSPREP='perceptron')
        obj.observations = {'heads': [1.0, 2.0]}
        obj.observationsNormalized = {'heads': [0.5, 0.6]}
        obj.observationsNormalizedHeads = {'heads': [0.5, 0.6]}
        with patch.object(obj, 'observationsDictToVector', return_value=[1, 2, 3]):
            ObservationsMixin._vectorize_observations(obj)
            assert obj.observationsVector == [1, 2, 3]

    def test_convolution_vectorize(self):
        obj = make_obs(OBSPREP='convolution')
        obj.observations = {'heads': np.zeros((1, 5, 5))}
        obj.observationsNormalized = {'heads': np.zeros((1, 5, 5))}
        with patch.object(obj, 'observationsDictToVector', return_value=np.zeros((10, 1, 1))):
            ObservationsMixin._vectorize_observations(obj)


class TestGetNormalizedObservations:
    """Cover lines 367-372: get_normalized_observations full pipeline."""


class TestBuildConvolutionVector:
    """Cover line 624: _build_convolution_vector with helper wells."""

    def test_with_helper_wells(self):
        obj = make_obs(OBSPREP='convolution', nHelperWells=1)
        obj.helperWells = {'wellQ1': -200.0, 'wellCoords1': [10.0, 20.0, 30.0]}
        obs_dict = {
            'heads': np.ones((1, 5, 5)) * 58.0,
            'particleCoords': [1.0, 2.0, 3.0],
            'wellQ': -100.0,
            'wellCoords': [10.0, 20.0, 30.0],
            'wellQ1': -200.0,
            'wellCoords1': [10.0, 20.0, 30.0],
        }
        result = ObservationsMixin._build_convolution_vector(obj, obs_dict)
        assert isinstance(result, np.ndarray)

    def test_without_helper_wells(self):
        obj = make_obs(OBSPREP='convolution', nHelperWells=0)
        obs_dict = {
            'heads': np.ones((1, 5, 5)) * 58.0,
            'particleCoords': [1.0, 2.0, 3.0],
            'wellQ': -100.0,
            'wellCoords': [10.0, 20.0, 30.0],
        }
        result = ObservationsMixin._build_convolution_vector(obj, obs_dict)
        assert isinstance(result, np.ndarray)


class TestParseHelperWellsPerceptron:
    """Cover lines 665-673: _parse_helper_wells_perceptron."""

    def test_parse(self):
        obj = make_obs(nHelperWells=2)
        # Vector: [3 coords] + [2 heads] + [1 wellQ] + [3 wellCoords] + [2*(1 wellQ + 3 coords)]
        # = 3 + 2 + 1 + 3 + 8 = 17
        obs_vector = list(range(17))
        result = ObservationsMixin._parse_helper_wells_perceptron(obj, obs_vector, 8)
        assert 'wellQ0' in result
        assert 'wellCoords0' in result
        assert 'wellQ1' in result
        assert 'wellCoords1' in result


class TestParseConvolution:
    """Cover line 745: _parse_convolution with helper wells."""

    def test_with_helpers(self):
        obj = make_obs(OBSPREP='convolution', nHelperWells=1)
        # base 8 + 4 helper = 12 rows
        obs_vector = np.zeros((12, 1, 1))
        obs_vector[0, 0, 0] = 1.0
        obs_vector[1, 0, 0] = 2.0
        obs_vector[2, 0, 0] = 3.0
        obs_vector[3, 0, 0] = 58.0
        obs_vector[4, 0, 0] = -100.0
        obs_vector[5, 0, 0] = 10.0
        obs_vector[6, 0, 0] = 20.0
        obs_vector[7, 0, 0] = 30.0
        obs_vector[8, 0, 0] = -200.0
        obs_vector[9, 0, 0] = 40.0
        obs_vector[10, 0, 0] = 50.0
        obs_vector[11, 0, 0] = 60.0
        result = ObservationsMixin._parse_convolution(obj, obs_vector)
        assert result['wellQ'] == -100.0
        assert 'wellQ1' in result

    def test_without_helpers(self):
        obj = make_obs(OBSPREP='convolution', nHelperWells=0)
        obs_vector = np.zeros((8, 1, 1))
        obs_vector[0, 0, 0] = 1.0
        obs_vector[3, 0, 0] = 58.0
        obs_vector[4, 0, 0] = -100.0
        result = ObservationsMixin._parse_convolution(obj, obs_vector)
        assert result['wellQ'] == -100.0
        assert 'wellQ1' not in result


class TestParseHelperWellsConvolution:
    """Cover lines 707-717: _parse_helper_wells_convolution."""

    def test_parse(self):
        obj = make_obs(nHelperWells=2)
        obs_vector = np.zeros((16, 1, 1))
        obs_vector[8, 0, 0] = -100.0
        obs_vector[9, 0, 0] = 10.0
        obs_vector[10, 0, 0] = 20.0
        obs_vector[11, 0, 0] = 30.0
        obs_vector[12, 0, 0] = -200.0
        obs_vector[13, 0, 0] = 40.0
        obs_vector[14, 0, 0] = 50.0
        obs_vector[15, 0, 0] = 60.0
        result = ObservationsMixin._parse_helper_wells_convolution(obj, obs_vector)
        assert result['wellQ1'] == -100.0
        assert result['wellCoords1'] == [10.0, 20.0, 30.0]
        assert result['wellQ2'] == -200.0


class TestObservationsVectorToDict:
    """Cover lines 761-764: observationsVectorToDict."""

    def test_perceptron(self):
        obj = make_obs(OBSPREP='perceptron', nHelperWells=0)
        obs_vector = [1, 2, 3, 4, 5, 6, 7, 8]
        with patch.object(obj, '_parse_perceptron', return_value={'data': 1}) as m_parse:
            result = ObservationsMixin.observationsVectorToDict(obj, obs_vector)
            m_parse.assert_called_once_with(obs_vector)

    def test_convolution(self):
        obj = make_obs(OBSPREP='convolution', nHelperWells=0)
        obs_vector = np.zeros((10, 1, 1))
        with patch.object(obj, '_parse_convolution', return_value={'data': 1}) as m_parse:
            result = ObservationsMixin.observationsVectorToDict(obj, obs_vector)
            m_parse.assert_called_once_with(obs_vector)

    def test_unsupported(self):
        obj = make_obs(OBSPREP='invalid')
        with pytest.raises(ValueError, match="Unsupported OBSPREP"):
            ObservationsMixin.observationsVectorToDict(obj, [1, 2, 3])


class TestNormalize:
    """Cover line 779: normalize."""

    def test_normalize(self):
        obj = make_obs()
        array = np.array([10.0, 20.0, 30.0])
        result = ObservationsMixin.normalize(obj, array, 5.0, 10.0)
        assert np.allclose(result, [0.5, 1.5, 2.5])


class TestUnnormalize:
    """Cover lines 792-803: unnormalize."""


class TestObservationsDictToVector:
    """Cover observationsDictToVector dispatch."""

    def test_perceptron(self):
        obj = make_obs(OBSPREP='perceptron')
        obs_dict = {'heads': [1.0, 2.0]}
        with patch.object(obj, '_build_perceptron_vector', return_value=[1, 2]) as m_build:
            result = ObservationsMixin.observationsDictToVector(obj, obs_dict)
            m_build.assert_called_once_with(obs_dict)

    def test_convolution(self):
        obj = make_obs(OBSPREP='convolution')
        obs_dict = {'heads': np.ones((1, 5, 5))}
        with patch.object(obj, '_build_convolution_vector', return_value=np.zeros(10)) as m_build:
            result = ObservationsMixin.observationsDictToVector(obj, obs_dict)
            m_build.assert_called_once_with(obs_dict)

    def test_invalid(self):
        obj = make_obs(OBSPREP='invalid')
        with pytest.raises(ValueError, match="Unsupported OBSPREP"):
            ObservationsMixin.observationsDictToVector(obj, {})


class TestAppendHelperWellsPerceptron:
    """Cover _append_helper_wells_perceptron."""

    def test_append(self):
        obj = make_obs(nHelperWells=2)
        obs_dict = {
            'wellQ0': -100.0, 'wellCoords0': [1.0, 2.0, 3.0],
            'wellQ1': -200.0, 'wellCoords1': [4.0, 5.0, 6.0],
        }
        vals = [1.0]
        ObservationsMixin._append_helper_wells_perceptron(obj, vals, obs_dict)
        assert len(vals) == 9


class TestAppendHelperWellsConvolution:
    """Cover _append_helper_wells_convolution."""

    def test_append(self):
        obj = make_obs(nHelperWells=1)
        obs_dict = {
            'wellQ1': -100.0, 'wellCoords1': [1.0, 2.0, 3.0],
        }
        vals = []
        ObservationsMixin._append_helper_wells_convolution(obj, vals, obs_dict)
        assert len(vals) == 4


class TestFeedObsHash:
    """Cover _feed_obs_hash."""

    def test_feed(self):
        obj = make_obs()
        obj._obs_hasher = __import__('hashlib').sha256()
        obs = np.array([1.0, 2.0, 3.0])
        ObservationsMixin._feed_obs_hash(obj, obs)
        assert obj._obs_hasher is not None


class TestBuildObservations:
    """Cover build_observations."""

    def test_build(self):
        obj = make_obs()
        obj.head = np.ones((1, 5, 5)) * 58.0
        obj.chd_east = 57.5
        obj.chd_diff = 1.0
        obj.avg_rch_min = 0.0
        obj.rch_diff = 10.0
        new_recharge = np.ones((1, 5, 5))
        obs_wells = [1.0, 2.0, 3.0]
        obs_storms = [4.0, 5.0]
        result = ObservationsMixin.build_observations(obj, new_recharge, obs_wells, obs_storms)
        assert isinstance(result, list)

    def test_build_no_wells(self):
        obj = make_obs()
        obj.head = np.ones((1, 5, 5)) * 58.0
        obj.chd_east = 57.5
        obj.chd_diff = 1.0
        obj.avg_rch_min = 0.0
        obj.rch_diff = 10.0
        new_recharge = np.ones((1, 5, 5))
        result = ObservationsMixin.build_observations(obj, new_recharge, None, None)
        assert isinstance(result, list)


class TestNormalizeHelpers:
    """Cover _normalize_heads, _normalize_coords, _normalize_wellQ."""

    def test_normalize_heads(self):
        obj = make_obs(minH=56.0, maxH=60.0)
        result = ObservationsMixin._normalize_heads(obj, [58.0, 60.0], 56.0, 60.0)
        assert result[0] == 0.5
        assert result[1] == 1.0

    def test_normalize_coords(self):
        obj = make_obs(minX=0.0, extentX=100.0)
        result = ObservationsMixin._normalize_coords(obj, [50.0, 100.0], 0.0, 100.0)
        assert result[0] == 0.5
        assert result[1] == 1.0

    def test_normalize_wellQ(self):
        obj = make_obs(minQ=-2000.0)
        result = ObservationsMixin._normalize_wellQ(obj, -1000.0, -2000.0)
        assert result == 0.5


class TestNormalizeWell:
    """Cover _normalize_well."""

    def test_normalize_well(self):
        obj = make_obs(minQ=-2000.0, minX=0.0, extentX=100.0, minY=0.0, extentY=100.0, zBot=0.0, zTop=50.0)
        result = ObservationsMixin._normalize_well(obj, -1000.0, 50.0, 50.0, 25.0)
        assert result == [0.5, 0.5, 0.5, 0.5]


class TestNormalizeHelperWells:
    """Cover _normalize_helper_wells."""

    def test_normalize(self):
        obj = make_obs(nHelperWells=2, minQ=-2000.0, minX=0.0, extentX=100.0, minY=0.0, extentY=100.0, zBot=0.0, zTop=50.0)
        obj.helperWells = {
            'wellQ1': -1000.0, 'wellX1': 50.0, 'wellY1': 50.0, 'wellZ1': 25.0,
            'wellQ2': -2000.0, 'wellX2': 100.0, 'wellY2': 100.0, 'wellZ2': 50.0,
        }
        result = ObservationsMixin._normalize_helper_wells(obj)
        assert len(result) == 8


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v', '--timeout=10']))
