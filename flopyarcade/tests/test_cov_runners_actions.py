#!/usr/bin/env python3
"""Coverage tests for modflow_runner, modpath_runner, reward, actions, observations."""
import math
import hashlib
from builtins import min as py_min, max as py_max
from copy import copy, deepcopy
from itertools import product
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from flopyarcade.modflow_runner import ModflowMixin
from flopyarcade.modpath_runner import ModpathMixin
from flopyarcade.reward import RewardMixin
from flopyarcade.actions import ActionsMixin
from flopyarcade.observations import ObservationsMixin


class TestEnv(ModflowMixin, ModpathMixin, RewardMixin, ActionsMixin, ObservationsMixin):
    """Lightweight test harness combining all mixins plus env_base helpers."""

    def __init__(self):
        self.done = False
        self.reward = 0.0
        self.rewardCurrent = 0.0
        self.gameReward = 0.0

    def _calculate_index(self, coord, delta, max_index):
        max_index = int(max_index)
        idx = int(math.ceil(coord / delta))
        return py_max(1, py_min(idx, max_index))

    def distance_to_well(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _sum_flows(self):
        return 0.0, 0.0

    def check_boundaries(self):
        pass

    def check_max_steps(self):
        pass

    def _modify_mpsim_file(self):
        pass

    def _filterByTime(self, data):
        return np.ones(len(data), dtype=bool)

    def _appendTrajectories(self, data, valid):
        pass

    def calculatePathLength(self, x, y):
        if len(x) < 2:
            return 0.0
        dx = np.diff(x)
        dy = np.diff(y)
        return float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))

    def collect_Qs(self):
        return np.array([self.wellQ])

    def initializeWellRate(self, minQ, maxQ):
        return 50.0, 50.0, 0.0, [50.0, 50.0, 0.0]

    def _get_delta_1d(self, action):
        if action == 'up':
            return 1.0
        if action == 'down':
            return -1.0
        return 0.0

    def _get_delta_2d(self, action):
        if action == 'up':
            return 1.0
        if action == 'down':
            return -1.0
        return 0.0

    def _calc_delta(self, pos, neg):
        return pos - neg

    def _calculate_displacement(self, action, offset):
        dy = action[offset] - action[offset + 1]
        dx = action[offset + 2] - action[offset + 3]
        return dy, dx

    def _append_list(self, base_list, key, obsDict):
        if key in obsDict:
            base_list.extend(obsDict[key])

    def _update_1(self):
        self.state['actionValueNorth'] = self.actionValueNorth
        self.state['actionValueSouth'] = self.actionValueSouth

    def _update_2(self):
        self.state['actionValue'] = self.actionValue

    def _update_3(self):
        self.state['actionValueX'] = self.actionValueX
        self.state['actionValueY'] = self.actionValueY

    def teardown(self):
        pass

    def _set_basic_observations(self):
        pass

    def _set_heads_observations(self):
        pass

    def _set_well_observations(self):
        pass

    def _normalize_basic_stresses(self, vals):
        return [v / 100.0 for v in vals]

    def build_observations_storms(self):
        return []

    def build_observations_wells(self):
        return {}

    def _init_observations(self):
        self.observations = {'heads': np.ones(10), 'particleCoords': np.array([1.0, 1.0]),
                            'wellQ': 50.0, 'wellCoords': np.array([1.0, 1.0])}
        self.observationsNormalized = {}
        self.observationsNormalizedHeads = {}


# =========================================================================
# MODFLOW RUNNER
# =========================================================================

class TestModflowRunner:

    def _make_env(self):
        env = TestEnv()
        env.delc = 10.0
        env.delr = 10.0
        env.ncol = 10
        env.nrow = 10
        env.extentX = 100.0
        env.extentY = 100.0
        env.timeStep = 0
        env.ENVTYPE = '0s-c'
        env.zBot = 0.0
        env.dVer = 10.0
        env.nLay = 5
        env.minX = 0.0
        env.dCol = 10.0
        env.nCol = 10
        env.minY = 0.0
        env.dRow = 10.0
        env.nRow = 10
        return env

    def test_get_cellID(self):
        env = self._make_env()
        cellID = env.get_cellID(50.0, 50.0)
        assert cellID == 46

    def test_get_cellID_origin(self):
        env = self._make_env()
        cellID = env.get_cellID(0.0, 0.0)
        assert cellID == 91

    def test_get_cellID_out_of_bounds_x(self):
        env = self._make_env()
        with pytest.raises(ValueError, match="outside the model domain"):
            env.get_cellID(200.0, 50.0)

    def test_get_cellID_out_of_bounds_y(self):
        env = self._make_env()
        with pytest.raises(ValueError, match="outside the model domain"):
            env.get_cellID(50.0, 200.0)

    def test_get_cellColRow(self):
        env = self._make_env()
        iCol, iRow = env.get_cellColRow(50.0, 50.0)
        assert iCol == 5
        assert iRow == 4

    def test_get_cellColRow_out_of_bounds(self):
        env = self._make_env()
        with pytest.raises(ValueError):
            env.get_cellColRow(-1.0, 50.0)

    def test_runMODFLOW_check(self):
        env = self._make_env()
        with patch.object(env, '_write_input') as m_wi,              patch.object(env, '_check_model') as m_cm,              patch.object(env, '_time_run_model') as m_trm,              patch.object(env, '_load_simulation_results') as m_lsr:
            env.runMODFLOW(check=True)
            m_wi.assert_called_once()
            m_cm.assert_called_once()
            m_trm.assert_called_once()
            m_lsr.assert_called_once()

    def test_runMODFLOW_no_check(self):
        env = self._make_env()
        with patch.object(env, '_write_input') as m_wi,              patch.object(env, '_check_model') as m_cm,              patch.object(env, '_time_run_model') as m_trm,              patch.object(env, '_load_simulation_results') as m_lsr:
            env.runMODFLOW(check=False)
            m_wi.assert_called_once()
            m_cm.assert_not_called()

    def test__time_run_model_success(self):
        env = self._make_env()
        env.mf = MagicMock()
        env.mf.run_model.return_value = (True, "success output")
        env._time_run_model()
        assert env.success_modflow is True
        assert env.buff == "success output"
        assert env.t_simulate >= 0

    def test__time_run_model_failure_all_retries(self):
        env = self._make_env()
        env.mf = MagicMock()
        env.mf.run_model.return_value = (False, "failure output")
        with patch('flopyarcade.modflow_runner.gc') as mock_gc,              patch('flopyarcade.modflow_runner.sleep'):
            with pytest.raises(RuntimeError, match="MODFLOW did not terminate normally"):
                env._time_run_model()
        assert env.mf.run_model.call_count == 3

    def test__time_run_model_retry_success(self):
        env = self._make_env()
        env.mf = MagicMock()
        env.mf.run_model.side_effect = [
            (False, "first fail"),
            (True, "second ok"),
        ]
        with patch('flopyarcade.modflow_runner.gc') as mock_gc,              patch('flopyarcade.modflow_runner.sleep'):
            env._time_run_model()
        assert env.success_modflow is True
        assert env.buff == "second ok"

    def test_cellInfoFromCoordinates(self):
        env = self._make_env()
        result = env.cellInfoFromCoordinates([50.0, 50.0, 0.0])
        assert result == (1, 5, 5)

    def test_cellInfoFromCoordinates_boundary(self):
        env = self._make_env()
        result = env.cellInfoFromCoordinates([0.0, 0.0, 0.0])
        assert result[0] >= 1
        assert result[1] >= 1
        assert result[2] >= 1

    def test__write_input_timeStep_zero(self):
        env = self._make_env()
        env.timeStep = 0
        env.mf = MagicMock()
        env._write_input()
        env.mf.write_input.assert_called_once()
        assert env.debug_written == 'full'

    def test__write_input_timeStep_positive(self):
        env = self._make_env()
        env.timeStep = 5
        env.mf_bas = MagicMock()
        env.wel = MagicMock()
        env._write_input()
        env.mf_bas.write_file.assert_called_once_with(check=False)
        env.wel.write_file.assert_called_once()
        assert env.debug_written == 'partial'

    def test__check_model(self):
        env = self._make_env()
        env.mf = MagicMock()
        env.mf.check.return_value = True
        env._check_model()
        assert env.check is True

    def test_updateModel(self):
        env = self._make_env()
        with patch.object(env, 'constructModel') as m_cm,              patch.object(env, '_update_cell_info'),              patch.object(env, '_update_modflow_wel'):
            env.updateModel()
            m_cm.assert_called_once()

    def test_get_well_cells(self):
        env = self._make_env()
        env.nHelperWells = 2
        env.actionsDict = {'well0cellID': 10, 'well1cellID': 20}
        result = env.get_well_cells()
        np.testing.assert_array_equal(result, np.array([10, 20], dtype=np.int32))

    def test_get_wel_x(self):
        env = self._make_env()
        env.mf6 = MagicMock()
        env.nameUpper = 'MODEL1'
        env.mf6.get_var_address.return_value = ('tag', 'WEL_0')
        env.mf6.get_value.return_value = np.array([1, 2, 3])
        tag, val = env.get_wel_x()
        assert tag == ('tag', 'WEL_0')
        np.testing.assert_array_equal(val, [1, 2, 3])

    def test_update_wel_x(self):
        env = self._make_env()
        env.mf6 = MagicMock()
        new_vals = np.array([5, 6, 7])
        env.update_wel_x('some_tag', new_vals)
        env.mf6.set_value.assert_called_once_with('some_tag', new_vals)

    def test_update_well_cells(self):
        env = self._make_env()
        env.nHelperWells = 2
        env.actionsDict = {'well0cellID': 10, 'well1cellID': 20}
        with patch.object(env, 'get_wel_x', return_value=('tag', np.array([0, 0]))),              patch.object(env, 'update_wel_x') as m_ux:
            env.update_well_cells()
            m_ux.assert_called_once()
            args = m_ux.call_args[0]
            assert args[0] == 'tag'
            np.testing.assert_array_equal(args[1], [10, 20])


# =========================================================================
# MODPATH RUNNER
# =========================================================================

class TestModpathRunner:

    def _make_env(self):
        env = TestEnv()
        env.delc = 10.0
        env.delr = 10.0
        env.ncol = 10
        env.nrow = 10
        env.extentX = 100.0
        env.extentY = 100.0
        env.timeStep = 1
        env.ENVTYPE = '6s-c'
        env.particleCoords = np.array([50.0, 50.0])
        env.wellCoords = np.array([50.0, 50.0])
        env.wellRadius = 10.0
        env.particleID = 1
        env.modelpth = '/tmp/test_model'
        env.MODELNAME = 'TESTMODEL'
        env.nHelperWells = 0
        env.helperWells = {}
        env.helperWellRadius = 5.0
        return env

    def test_check_particle_within_well(self):
        env = self._make_env()
        env.particleCoords = np.array([50.0, 50.0])
        env.wellCoords = np.array([50.0, 50.0])
        env.wellRadius = 10.0
        env.done = False
        env.check_particle_within_well()
        assert env.done is True

    def test_check_particle_not_within_well(self):
        env = self._make_env()
        env.particleCoords = np.array([10.0, 10.0])
        env.wellCoords = np.array([50.0, 50.0])
        env.wellRadius = 5.0
        env.done = False
        env.check_particle_within_well()
        assert env.done is False

    def test_check_particle_within_helper_wells_no_helpers(self):
        env = self._make_env()
        env.ENVTYPE = '6s-c'
        env.nHelperWells = 0
        env.helperWells = {}
        env.done = False
        env.check_particle_within_helper_wells()
        assert env.done is False

    def test_check_particle_within_helper_wells_with_helper(self):
        env = self._make_env()
        env.ENVTYPE = '6s-c'
        env.nHelperWells = 1
        env.helperWells = {'wellCoords1': np.array([50.0, 50.0])}
        env.helperWellRadius = 10.0
        env.particleCoords = np.array([50.0, 50.0])
        env.done = False
        env.check_particle_within_helper_wells()
        assert env.done is True

    def test_check_particle_state(self):
        env = self._make_env()
        env.particleCoords = np.array([50.0, 50.0])
        env.wellCoords = np.array([50.0, 50.0])
        env.wellRadius = 10.0
        env.nHelperWells = 0
        env.helperWells = {}
        env.done = False
        env.check_particle_state()
        assert env.done is True

    def test_runMODPATH(self):
        env = self._make_env()
        env.mp = MagicMock()
        with patch.object(env, '_transform_particle_coords') as m_tp,              patch.object(env, '_initialize_modpath_objects') as m_init,              patch.object(env, '_write_modpath_input') as m_wi,              patch.object(env, '_modify_mpsim_file') as m_mpf,              patch.object(env, '_write_particle_location') as m_wpl:
            env.runMODPATH()
            m_tp.assert_called_once()
            m_init.assert_called_once()
            m_wi.assert_called_once()
            m_mpf.assert_called_once()
            m_wpl.assert_called_once()
            env.mp.run_model.assert_called_once_with(silent=True)

    def test_cleanup_temp_dir(self):
        env = self._make_env()
        env.tempDir = MagicMock()
        env.cleanup_temp_dir()
        env.tempDir.cleanup.assert_called_once()


# =========================================================================
# REWARD
# =========================================================================

class TestReward:

    def _make_env(self):
        env = TestEnv()
        env.ENVTYPE = '1s-c'
        env.rewardCurrent = 0.0
        env.costQFail = -10.0
        env.distanceMax = 100.0
        env.rewardMax = 1.0
        env.deviationPenaltyFactor = 2
        env.costQm3dPump = 0.01
        env.costQm3dRecharge = 0.005
        return env

    def test_update_reward_on_fail_cost_type(self):
        env = self._make_env()
        env.ENVTYPE = '5s-c-cost'
        env.rewardCurrent = 5.0
        env.costQFail = -10.0
        env.update_reward_on_fail()
        assert env.reward == -5.0
        assert env.rewardCurrent == 0.0

    def test_update_reward_on_fail_normal_positive(self):
        env = self._make_env()
        env.ENVTYPE = '1s-c'
        env.rewardCurrent = 3.0
        env.update_reward_on_fail()
        assert env.reward == -3.0
        assert env.rewardCurrent == 0.0

    def test_update_reward_on_fail_normal_negative(self):
        env = self._make_env()
        env.ENVTYPE = '1s-c'
        env.rewardCurrent = -2.0
        env.update_reward_on_fail()
        assert env.reward == 0.0
        assert env.rewardCurrent == 0.0

    def test_calculateGameRewardTrajectory_forward(self):
        env = self._make_env()
        env.trajectories = {'x': [np.array([0.0, 80.0])], 'y': [np.array([0.0, 5.0])]}
        reward = env.calculateGameRewardTrajectory()
        # length_shortest = 80, length_actual = sqrt(80^2+5^2) ~ 80.16
        # dist_frac = 0.8, rewardMaxSegment = 0.8
        # path_ratio ~ 0.998, reward ~ 0.8 * 0.996 ~ 0.797
        assert 0.7 < reward < 0.81

    def test_calculateGameRewardTrajectory_no_movement(self):
        env = self._make_env()
        env.trajectories = {'x': [np.array([50.0, 50.0])], 'y': [np.array([0.0, 0.0])]}
        reward = env.calculateGameRewardTrajectory()
        assert reward == 0.0

    def test_calculateGameRewardTrajectory_backward(self):
        env = self._make_env()
        env.trajectories = {'x': [np.array([100.0, 50.0])], 'y': [np.array([0.0, 0.0])]}
        reward = env.calculateGameRewardTrajectory()
        # length_shortest = -50, dist_frac = -0.5
        # rewardMaxSegment = -0.5, reward = -5*0.5 = -2.5
        assert reward == -2.5

    def test_calculateGameRewardHeadChange(self):
        env = self._make_env()
        target = np.array([1.0, 2.0, 3.0])
        realized = np.array([1.5, 2.5, 3.5])
        reward = env.calculateGameRewardHeadChange(target, realized)
        assert reward == pytest.approx(-1.5)

    def test_calculateGameRewardOperationCost(self):
        env = self._make_env()
        with patch.object(env, '_sum_flows', return_value=(100.0, 50.0)):
            reward = env.calculateGameRewardOperationCost()
        assert reward == pytest.approx(-(100.0 * 0.01 + 50.0 * 0.005))

    def test_compute_reward(self):
        env = self._make_env()
        result = env._compute_reward(100.0, 50.0)
        assert result == pytest.approx(-(100.0 * 0.01 + 50.0 * 0.005))

    def test_format_score(self):
        env = self._make_env()
        env.ENVTYPE = '1s-c'
        env.reward = 0.75
        result = env.format_score()
        assert isinstance(result, str)


# =========================================================================
# ACTIONS
# =========================================================================

class TestActions:

    def _make_env(self, n_helper=2):
        env = TestEnv()
        env.ENVTYPE = '6s-c'
        env.nHelperWells = n_helper
        env.well_dxMax = 5.0
        env.well_dyMax = 5.0
        env.well_dQMax = 10.0
        env.delc = 10.0
        env.delr = 10.0
        env.ncol = 10
        env.nrow = 10
        env.extentX = 100.0
        env.extentY = 100.0
        env.minQ = 10.0
        env.maxQ = 200.0
        env.minX = 0.0
        env.minY = 0.0
        env.zBot = 0.0
        env.zTop = 10.0
        env.wellQ = 50.0
        env.wellX = 50.0
        env.wellY = 50.0
        env.wellZ = 5.0
        env.headSpecSouth = 10.0
        env.headSpecNorth = 20.0
        env.actionValueNorth = 1.0
        env.actionValueSouth = 1.0
        env.actionValue = 1.0
        env.actionValueX = 1.0
        env.actionValueY = 1.0
        env.actionRange = 5.0
        env.state = {}
        env.helperWells = {}
        for i in range(n_helper):
            w = str(i + 1)
            env.helperWells[f'wellX{w}'] = 30.0 + i * 10
            env.helperWells[f'wellY{w}'] = 30.0 + i * 10
            env.helperWells[f'wellQ{w}'] = 50.0
            env.helperWells[f'wellZ{w}'] = 5.0
            env.helperWells[f'wellCoords{w}'] = np.array([30.0 + i * 10, 30.0 + i * 10])
            env.helperWells[f'actionValueX{w}'] = 30.0 + i * 10
            env.helperWells[f'actionValueY{w}'] = 30.0 + i * 10
            env.helperWells[f'actionValueQ{w}'] = 50.0
        return env

    def test_changeActionDict_valid(self):
        env = self._make_env(n_helper=2)
        actionsDict = {}
        action = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
        result = env.changeActionDict(actionsDict, action)
        assert result['well0actiondxRight'] == 1
        assert result['well0actiondxLeft'] == 0
        assert result['well0actiondyUp'] == 1
        assert result['well0actiondyDown'] == 0
        assert result['well0actiondQUp'] == 0
        assert result['well0actiondQDown'] == 1
        assert result['well1actiondxRight'] == 0
        assert result['well1actiondxLeft'] == 1
        assert result['well1actiondyUp'] == 0
        assert result['well1actiondyDown'] == 0
        assert result['well1actiondQUp'] == 1
        assert result['well1actiondQDown'] == 0

    def test_changeActionDict_wrong_length(self):
        env = self._make_env(n_helper=2)
        with pytest.raises(ValueError, match="does not match expected size"):
            env.changeActionDict({}, [1, 2, 3])

    def test_getActionSpaceSize(self):
        env = self._make_env(n_helper=2)
        actionsDict = {'a': 1, 'b': 2, 'c': 3}
        assert env.getActionSpaceSize(actionsDict) == 3

    def test_getActionType_discrete(self):
        env = self._make_env()
        assert env.getActionType('1s-d') == 'discrete'

    def test_getActionType_continuous(self):
        env = self._make_env()
        assert env.getActionType('1s-c') == 'continuous'

    def test_getActionType_invalid(self):
        env = self._make_env()
        with pytest.raises(ValueError, match="Unknown environment type"):
            env.getActionType('1s-x')

    def test_update_well_x(self):
        env = self._make_env()
        ad = {'well0x': 50.0, 'well0actiondxRight': 1, 'well0actiondxLeft': 0}
        result = env._update_well_x(ad, 0)
        assert result['well0x'] == 55.0

    def test_update_well_x_clamped(self):
        env = self._make_env()
        ad = {'well0x': 99.0, 'well0actiondxRight': 1, 'well0actiondxLeft': 0}
        result = env._update_well_x(ad, 0)
        max_x = 100.0 - 1.1 * 10.0
        assert result['well0x'] == max_x

    def test_update_well_y(self):
        env = self._make_env()
        ad = {'well0y': 50.0, 'well0actiondyUp': 1, 'well0actiondyDown': 0}
        result = env._update_well_y(ad, 0)
        assert result['well0y'] == 55.0

    def test_update_well_Q(self):
        env = self._make_env()
        ad = {'well0Q': 50.0, 'well0actiondQUp': 1, 'well0actiondQDown': 0}
        result = env._update_well_Q(ad, 0)
        assert result['well0Q'] == 60.0

    def test_update_well_Q_clamped(self):
        env = self._make_env()
        ad = {'well0Q': 195.0, 'well0actiondQUp': 1, 'well0actiondQDown': 0}
        result = env._update_well_Q(ad, 0)
        assert result['well0Q'] == 200.0

    def test_getActionValues(self):
        env = self._make_env(n_helper=1)
        ad = {
            'well0x': 50.0, 'well0y': 50.0, 'well0Q': 50.0,
            'well0actiondxRight': 1, 'well0actiondxLeft': 0,
            'well0actiondyUp': 0, 'well0actiondyDown': 1,
            'well0actiondQUp': 0, 'well0actiondQDown': 0,
        }
        result = env.getActionValues(ad)
        assert result['well0x'] == 55.0
        assert result['well0y'] == 45.0
        assert result['well0Q'] == 50.0
        assert 'well0cellID' in result

    def test_update_action_1d_up(self):
        env = self._make_env()
        env.actionValueNorth = 0.0
        env.actionValueSouth = 0.0
        env.update_action_1d('up')
        assert env.actionValueNorth == 1.0
        assert env.actionValueSouth == 1.0

    def test_update_action_1d_down(self):
        env = self._make_env()
        env.actionValueNorth = 0.0
        env.actionValueSouth = 0.0
        env.update_action_1d('down')
        assert env.actionValueNorth == -1.0
        assert env.actionValueSouth == -1.0

    def test_update_action_1d_keep(self):
        env = self._make_env()
        env.actionValueNorth = 5.0
        env.actionValueSouth = 5.0
        env.update_action_1d('keep')
        assert env.actionValueNorth == 5.0
        assert env.actionValueSouth == 5.0

    def test_update_action_1c(self):
        env = self._make_env()
        env.actionValueNorth = 0.0
        env.actionValueSouth = 0.0
        env.actionRange = 5.0
        env.update_action_1c([0.8, 0.2])
        # inc = 0.8*5=4, dec=0.2*5=1
        # north = 0+4-1=3, south=0+4-1=3
        assert env.actionValueNorth == pytest.approx(3.0)
        assert env.actionValueSouth == pytest.approx(3.0)

    def test_update_helper_wells_keys(self):
        env = self._make_env(n_helper=2)
        env._update_helper_wells(['X', 'Y'])
        assert env.state['actionValueX1'] == 30.0
        assert env.state['actionValueY1'] == 30.0
        assert env.state['actionValueX2'] == 40.0
        assert env.state['actionValueY2'] == 40.0

    def test_get_action_value_scaled_dx(self):
        env = self._make_env()
        env.actionsDict = {'well0dxRight': 1.0}
        result = env.get_action_value(0, 'dxRight')
        assert result == 5.0

    def test_get_action_value_scaled_dy(self):
        env = self._make_env()
        env.actionsDict = {'well0dyUp': 0.5}
        result = env.get_action_value(0, 'dyUp')
        assert result == 2.5

    def test_get_action_value_dQ_scaled(self):
        env = self._make_env()
        env.actionsDict = {'well0actiondQUp': 0.5}
        # suffix 'actiondQUp' contains 'dQ' -> scaled by well_dyMax
        result = env.get_action_value(0, 'actiondQUp')
        assert result == 2.5  # 5.0 * 0.5

    def test_update_well_Q_method(self):
        env = self._make_env()
        env.well = np.zeros((1, 3))
        env.wellQ = 50.0
        env.update_well_Q()
        assert env.well[0, 0] == 50.0

    def test_parse_action_list(self):
        env = self._make_env(n_helper=1)
        env.actionSpaceIndividual = ['up', 'down', 'left', 'right', 'moreQ', 'lessQ', 'keep']
        result = env._parse_action_list('uprightmoreQ')
        assert result == ['up', 'right', 'moreQ']

    def test_resolve_well_overlap_no_overlap(self):
        env = self._make_env(n_helper=1)
        env.wellX = 10.0
        env.wellY = 10.0
        env.wellRadius = 10.0
        env.helperWellRadius = 5.0
        env.dCol = 10.0
        env.dRow = 10.0
        env.helperWells['wellX1'] = 80.0
        env.helperWells['wellY1'] = 80.0
        env.extentX = 100.0
        env.extentY = 100.0
        env._resolve_well_overlap()


# =========================================================================
# OBSERVATIONS
# =========================================================================

class TestObservations:

    def _make_env(self):
        env = TestEnv()
        env.ENVTYPE = '1s-c'
        env.minX = 0.0
        env.extentX = 100.0
        env.minY = 0.0
        env.extentY = 100.0
        env.minH = 0.0
        env.maxH = 10.0
        env.minQ = 10.0
        env.maxQ = 200.0
        env.zBot = 0.0
        env.zTop = 10.0
        env.wellQ = 50.0
        env.wellX = 50.0
        env.wellY = 50.0
        env.wellZ = 5.0
        env.particleCoords = np.array([25.0, 25.0])
        env.heads = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        env.minQhelper = 5.0
        env.nHelperWells = 0
        env.helperWells = {}
        env.actionValueNorth = 1.0
        env.actionValueSouth = 1.0
        env.actionValue = 1.0
        env.headSpecSouth = 10.0
        env.headSpecNorth = 20.0
        return env

    def test_normalize_heads(self):
        env = self._make_env()
        heads = np.array([0.0, 5.0, 10.0])
        result = env._normalize_heads(heads, 0.0, 10.0)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_normalize_coords(self):
        env = self._make_env()
        coords = np.array([0.0, 50.0, 100.0])
        result = env._normalize_coords(coords, 0.0, 100.0)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_normalize_wellQ(self):
        env = self._make_env()
        wellQ = np.array([10.0, 20.0, 50.0])
        result = env._normalize_wellQ(wellQ, 10.0)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 5.0])

    def test_normalize_scalar(self):
        env = self._make_env()
        result = env._normalize(5.0, 0.0, 10.0)
        assert result == 0.5

    def test_normalize_array(self):
        env = self._make_env()
        result = env._normalize(np.array([0.0, 5.0, 10.0]), 0.0, 10.0)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_normalize_well(self):
        env = self._make_env()
        result = env._normalize_well(50.0, 50.0, 50.0, 5.0)
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(0.5)
        assert result[3] == pytest.approx(0.5)

    def test_normalize_helper_wells(self):
        env = self._make_env()
        env.nHelperWells = 2
        env.helperWells = {
            'wellQ1': 50.0, 'wellX1': 30.0, 'wellY1': 30.0, 'wellZ1': 5.0,
            'wellQ2': 100.0, 'wellX2': 60.0, 'wellY2': 60.0, 'wellZ2': 5.0,
        }
        result = env._normalize_helper_wells()
        assert len(result) == 8

    def test_get_normalized_stresses_1s(self):
        env = self._make_env()
        env.ENVTYPE = '1s-c'
        result = env.get_normalized_stresses()
        assert len(result) == 6

    def test_get_normalized_stresses_unknown(self):
        env = self._make_env()
        env.ENVTYPE = '99-unknown'
        result = env.get_normalized_stresses()
        assert result == []

    def test_get_normalized_stresses_helper_wells(self):
        env = self._make_env()
        env.ENVTYPE = '6s-c'
        env.nHelperWells = 2
        env.helperWells = {
            'wellQ1': 50.0, 'wellX1': 30.0, 'wellY1': 30.0, 'wellZ1': 5.0,
            'wellQ2': 100.0, 'wellX2': 60.0, 'wellY2': 60.0, 'wellZ2': 5.0,
        }
        result = env.get_normalized_stresses()
        assert len(result) == 10

    def test_get_normalized_stresses_2s(self):
        env = self._make_env()
        env.ENVTYPE = '2s-c'
        env.actionValue = 5.0
        result = env.get_normalized_stresses()
        assert len(result) == 5  # 1 base + 4 well

    def test_get_normalized_stresses_3s(self):
        env = self._make_env()
        env.ENVTYPE = '3s-c'
        result = env.get_normalized_stresses()
        assert len(result) == 6  # 2 base + 4 well

    def test_normalize_observations(self):
        env = self._make_env()
        env.wellCoords = np.array([50.0, 50.0])
        env.observations = {
            'heads': np.array([1.0, 2.0, 3.0]),
            'particleCoords': np.array([25.0, 25.0]),
            'wellQ': 50.0,
            'wellCoords': np.array([50.0, 50.0]),
        }
        env.observationsNormalized = {}
        env.observationsNormalizedHeads = {}
        env._normalize_observations()
        assert 'heads' in env.observationsNormalized
        assert 'particleCoords' in env.observationsNormalized
        assert 'heads' in env.observationsNormalizedHeads
