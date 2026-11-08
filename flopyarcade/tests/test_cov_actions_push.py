"""Coverage push tests for actions.py ActionsMixin methods."""
import sys
import numpy as np
from unittest.mock import MagicMock, patch
import pytest


def _mk(ENVTYPE='4s-d', nHelperWells=3, **kw):
    from flopyarcade.actions import ActionsMixin
    obj = object.__new__(ActionsMixin)
    obj.ENVTYPE = ENVTYPE
    obj.nHelperWells = nHelperWells
    obj.actionType = 'discrete' if '-d' in ENVTYPE else 'continuous'
    obj.delr = 10.0
    obj.delc = 10.0
    obj.extentX = 100.0
    obj.extentY = 100.0
    obj.actionsDict = {}
    obj.helperWells = {}
    obj.state = {}
    obj.minQhelper = -5.0
    obj.maxQhelper = 5.0
    obj.well_dxMax = 5.0
    obj.well_dyMax = 5.0
    obj.well = np.zeros((nHelperWells + 1, 1))
    obj.actionSpace = None
    obj.actionSpaceIndividual = None
    obj.actionSpaceSize = 0
    obj._init_type_1 = MagicMock()
    obj._init_type_2 = MagicMock()
    obj._init_type_3 = MagicMock()
    obj._init_type_4 = MagicMock()
    obj._init_type_5 = MagicMock()
    obj._init_type_6 = MagicMock()
    obj.get_cellID = MagicMock(return_value=0)
    obj.get_cellColRow = MagicMock(return_value=(1, 1))
    obj.collect_Qs = MagicMock(return_value=np.array([-1.0, -2.0, -3.0, -4.0]))
    obj.initializeWellRate = MagicMock(return_value=(10.0, 20.0, 0.0, (10.0, 20.0), -1.0))
    for k, v in kw.items():
        setattr(obj, k, v)
    return obj


class TestGetActionType:
    def test_discrete(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk()
        assert ActionsMixin.getActionType(obj, '1s-d') == 'discrete'

    def test_continuous(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk()
        assert ActionsMixin.getActionType(obj, '1s-c') == 'continuous'

    def test_invalid(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk()
        with pytest.raises(ValueError):
            ActionsMixin.getActionType(obj, 'xyz')


class TestGetActionValue:
    def test_scaled_dx(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk()
        obj.actionsDict = {'well1dx': 0.5}
        assert ActionsMixin.get_action_value(obj, 1, 'dx') == 2.5

    def test_scaled_dy(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk()
        obj.actionsDict = {'well2dy': 0.3}
        assert ActionsMixin.get_action_value(obj, 2, 'dy') == 1.5

    def test_unscaled_Q(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk()
        obj.actionsDict = {'well1Q': -10.0}
        assert ActionsMixin.get_action_value(obj, 1, 'Q') == -10.0


class TestUpdateWellQ:
    def test_basic(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(nHelperWells=3)
        ActionsMixin.update_well_Q(obj)
        assert obj.well[0, 0] == -1.0


class TestInitHelperWells:
    def test_creates_entries(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(nHelperWells=2)
        ActionsMixin._init_helper_wells(obj)
        assert 'well0x' in obj.actionsDict
        assert 'well1Q' in obj.actionsDict


class TestSetupActionSpace:
    def test_4s_discrete(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='4s-d', nHelperWells=2)
        with patch('flopyarcade.env_config.ENV_CONFIG', {'4s-d': {'action_discrete': None, 'action_continuous': None}}):
            ActionsMixin._setup_action_space(obj)
        assert obj.actionSpaceIndividual == ['up', 'keep', 'down', 'left', 'right']

    def test_5s_discrete(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='5s-d', nHelperWells=2)
        with patch('flopyarcade.env_config.ENV_CONFIG', {'5s-d': {'action_discrete': None, 'action_continuous': None}}):
            ActionsMixin._setup_action_space(obj)
        assert obj.actionSpaceIndividual == ['moreQ', 'keepQ', 'lessQ']

    def test_6s_discrete(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='6s-d', nHelperWells=2)
        with patch('flopyarcade.env_config.ENV_CONFIG', {'6s-d': {'action_discrete': None, 'action_continuous': None}}):
            ActionsMixin._setup_action_space(obj)
        assert len(obj.actionSpaceIndividual) == 8

    def test_4s_continuous(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='4s-c', nHelperWells=2)
        with patch('flopyarcade.env_config.ENV_CONFIG', {'4s-c': {'action_discrete': None, 'action_continuous': None}}):
            ActionsMixin._setup_action_space(obj)
        assert obj.actionSpaceIndividual == ['up', 'down', 'left', 'right']

    def test_5s_continuous(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='5s-c', nHelperWells=2)
        with patch('flopyarcade.env_config.ENV_CONFIG', {'5s-c': {'action_discrete': None, 'action_continuous': None}}):
            ActionsMixin._setup_action_space(obj)
        assert obj.actionSpaceIndividual == ['moreQ', 'lessQ']

    def test_6s_continuous(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='6s-c', nHelperWells=2)
        with patch('flopyarcade.env_config.ENV_CONFIG', {'6s-c': {'action_discrete': None, 'action_continuous': None}}):
            ActionsMixin._setup_action_space(obj)
        assert len(obj.actionSpaceIndividual) == 6


class TestInitializeHelperWells:
    def test_multiwell(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='4s-d', nHelperWells=2)
        ActionsMixin.initialize_helper_wells(obj)
        assert 'wellX1' in obj.helperWells
        assert 'wellQ2' in obj.helperWells

    def test_single_well_skips(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='1s-d', nHelperWells=0)
        ActionsMixin.initialize_helper_wells(obj)
        assert obj.helperWells == {}


class TestInitializeAction:
    def test_type_1(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='1s-d')
        ActionsMixin.initializeAction(obj)
        obj._init_type_1.assert_called_once()

    def test_type_4(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='4s-d')
        ActionsMixin.initializeAction(obj)
        obj._init_type_4.assert_called_once()

    def test_type_6(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='6s-c')
        ActionsMixin.initializeAction(obj)
        obj._init_type_6.assert_called_once()


class TestUpdateHelperWells:
    def test_basic(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='4s-d', nHelperWells=2)
        obj.helperWells = {'actionValueQ1': -1.0, 'actionValueQ2': -2.0}
        ActionsMixin._update_helper_wells(obj, ['Q'])
        assert obj.state['actionValueQ1'] == -1.0


class TestChangeActionDict:
    def test_discrete_up1(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='4s-d', nHelperWells=2)
        obj.actionsDict = {}
        # action is list of nHelperWells*6 values
        action = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ActionsMixin.changeActionDict(obj, obj.actionsDict, action)
        assert 'well0actiondxRight' in obj.actionsDict

    def test_discrete_keep1(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='4s-d', nHelperWells=1)
        obj.actionsDict = {}
        action = [0.5, 0.3, 0.1, 0.2, 0.0, 0.0]
        result = ActionsMixin.changeActionDict(obj, obj.actionsDict, action)
        assert result['well0actiondxRight'] == 0.5

    def test_continuous_moreQ(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='5s-c', nHelperWells=1)
        obj.actionsDict = {}
        action = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        result = ActionsMixin.changeActionDict(obj, obj.actionsDict, action)
        assert result['well0actiondQUp'] == 1.0


class TestGetActionSpaceSize:
    def test_basic(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='4s-d', nHelperWells=3)
        actionsDict = {'a': 1, 'b': 2, 'c': 3}
        result = ActionsMixin.getActionSpaceSize(obj, actionsDict)
        assert result == 3


class TestGetActionValues:
    def test_basic(self):
        from flopyarcade.actions import ActionsMixin
        obj = _mk(ENVTYPE='4s-d', nHelperWells=1)
        obj.actionsDict = {'well0x': 50.0, 'well0y': 60.0, 'well0Q': -1.0,
                          'well0actiondxRight': 0.0, 'well0actiondxLeft': 0.0,
                          'well0actiondyUp': 0.0, 'well0actiondyDown': 0.0,
                          'well0actiondQUp': 0.0, 'well0actiondQDown': 0.0}
        obj.get_cellID = MagicMock(return_value=42)
        obj._update_well_x = MagicMock(return_value=obj.actionsDict)
        obj._update_well_y = MagicMock(return_value=obj.actionsDict)
        obj._update_well_Q = MagicMock(return_value=obj.actionsDict)
        result = ActionsMixin.getActionValues(obj, obj.actionsDict)
        assert result['well0cellID'] == 42
