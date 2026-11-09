#!/usr/bin/env python3
"""Deep coverage tests for ModflowMixin in modflow_runner.py.

Targets 90%+ statement coverage. All flopy classes are mocked.
No MODFLOW executables are invoked.
"""
from unittest.mock import MagicMock, patch, PropertyMock
import gc
import math
import os
import tempfile
from builtins import min as py_min, max as py_max
from copy import copy, deepcopy

import numpy as np
import pytest

from flopyarcade.modflow_runner import ModflowMixin


# ---------------------------------------------------------------------------
# Dummy environment that inherits the mixin and supplies required helpers
# ---------------------------------------------------------------------------
class DummyEnv(ModflowMixin):
    """Minimal harness providing helpers the mixin expects from env_base."""

    def __init__(self):
        self.nlay = 2
        self.nrow = 4
        self.ncol = 5
        self.delc = 10.0
        self.delr = 10.0
        self.top = 10.0
        self.botm = [0.0, -10.0]
        self.strt = np.ones((self.nlay, self.nrow, self.ncol))
        self.ibound = np.ones((self.nlay, self.nrow, self.ncol), dtype=int)
        self.hk = 10.0
        self.name = 'testmodel'
        self.nameUpper = 'TESTMODEL'
        self.ENVTYPE = '1s-d'
        self.MODELNAME = 'MODEL001'
        self.modelpth = tempfile.mkdtemp()
        self.model_ws = tempfile.mkdtemp()
        self.wrkspc = '/tmp'
        self.exdir = 'test_exp'
        self.exe_name = 'mf2005.exe'
        self.mf6dll = 'mf6.dll'

        # well
        self.wellX = 50.0
        self.wellY = 50.0
        self.wellZ = -5.0
        self.wellQ = 100.0
        self.wellCellLayer = 1
        self.wellCellColumn = 3
        self.wellCellRow = 2
        self.nHelperWells = 2
        self.helperWells = {
            'wellX1': 20.0, 'wellY1': 30.0, 'wellZ1': -3.0, 'wellQ1': 50.0,
            'l1': 1, 'c1': 2, 'r1': 1,
            'wellX2': 60.0, 'wellY2': 70.0, 'wellZ2': -4.0, 'wellQ2': 60.0,
            'l2': 2, 'c2': 4, 'r2': 3,
        }
        self.shapeLayer = (self.nlay, self.nrow, self.ncol)
        self.actionsDict = {'well0cellID': 7, 'well1cellID': 12}

        # simulation params
        self.nstp = 10
        self.nper = 3
        self.periodSteps = 5
        self.tdis_rc = [(1.0, 1, 1)]
        self.hclose = 1e-5
        self.nouter = 50
        self.ninner = 100
        self.rclose = 1e-6
        self.relax = 1.0
        self.chd_west = 10.0
        self.chd_east = 5.0
        self.rech0 = np.zeros((self.nlay, self.nrow, self.ncol))
        self.rechTimeSeries = [self.rech0]

        # storms
        self.nStorms = 2
        self.minStormDuration = 1
        self.maxStormDuration = 5
        self.storm_rch_min = 0.1
        self.storm_rch_max = 1.0
        self.extentX = 50.0
        self.extentY = 40.0
        self.minStormRadius = 5.0
        self.maxStormRadius = 20.0
        self.stormStarts = [2, 5]
        self.stormDurations = [3, 2]
        self.stormIntensities = [0.5, 0.8]
        self.stormCentersX = [10.0, 30.0]
        self.stormCentersY = [15.0, 25.0]
        self.stormRadii = [10.0, 15.0]

        # observations
        self.particleCoords = np.array([1.0, 2.0, 3.0])
        self.heads = np.ones((self.nlay, self.nrow, self.ncol))
        self.sampleHeadsEvery = 1
        self.observations = {}
        self.minH = 0.0
        self.maxH = 100.0
        self.timeStep = 0
        self.success = False
        self.teardownSuccess = True

        # coordinate system
        self.zBot = 0.0
        self.dVer = 10.0
        self.nLay = self.nlay
        self.minX = 0.0
        self.dCol = 10.0
        self.nCol = self.ncol
        self.minY = 0.0
        self.dRow = 10.0
        self.nRow = self.nrow

        # mf6 mock
        self.mf6 = MagicMock()
        self.ax = MagicMock()

    # --- helpers the mixin expects ---
    def _calculate_index(self, coord, delta, max_index):
        max_index = int(max_index)
        idx = int(math.ceil(coord / delta))
        return py_max(1, py_min(idx, max_index))

    def _envHasHelperWells(self):
        return self.ENVTYPE in {
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c',
        }

    def _normalize(self, v, min_val, range_val):
        if range_val == 0:
            return 0.5
        return (v - min_val) / range_val

    def normalize(self, v, min_val, range_val):
        return self._normalize(v, min_val, range_val)

    def bmi_return(self, success, model_ws, modelname):
        return 'BMI_RETURN'

    def teardown(self):
        self.teardownSuccess = True

    def run_convergence_loop(self):
        return True

    def handle_convergence_failure(self, modelname):
        return 'CONV_FAIL'

    def get_max_iterations(self):
        self.max_iterations = 1000

    def setup_rch_0sc(self, recharge):
        pass

    def setup_wells_0sc(self):
        pass

    def create_discretization_other(self):
        pass

    def initialize_ibound_other(self):
        pass

    def initialize_start_heads_other(self):
        pass

    def get_cellID(self, x, y):
        # Override to use mixin's own implementation but keep it callable
        return super().get_cellID(x, y)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def env():
    return DummyEnv()


# =========================================================================
# WORKSPACE / SETUP
# =========================================================================
class TestWorkspace:

    def test_setup_workspace(self, env):
        env._setup_workspace()
        assert hasattr(env, 'wrkspc')
        assert hasattr(env, 'tempDir')
        assert hasattr(env, 'modelpth')
        assert hasattr(env, 'MODELNAME')
        env.tempDir.cleanup()

    def test_remove_model_workspace_exists(self, env):
        env.model_ws = tempfile.mkdtemp()
        env.tempDir = MagicMock()
        env._remove_model_workspace()
        assert env.teardownSuccess is True

    def test_remove_model_workspace_not_exists(self, env):
        env.model_ws = '/nonexistent/path/xyz'
        env._remove_model_workspace()
        assert env.teardownSuccess is True

    def test_remove_model_workspace_cleanup_fails(self, env):
        env.model_ws = tempfile.mkdtemp()
        env.tempDir = MagicMock()
        env.tempDir.cleanup.side_effect = Exception("boom")
        env._remove_model_workspace()
        assert env.teardownSuccess is False

    def test_get_executable_path_override(self, env):
        result = env._get_executable_path('/custom/path.exe', 'linux', 'mf', '')
        assert result == '/custom/path.exe'

    def test_get_executable_path_default(self, env):
        result = env._get_executable_path(None, 'linux', 'mf', '.exe')
        assert 'linux' in result
        assert result.endswith('.exe')

    def test_set_executable_permissions(self, env):
        p = os.path.join(env.model_ws, 'test_exe')
        with open(p, 'w') as f:
            f.write('x')
        with patch('flopyarcade.modflow_runner.chmod') as mock_chmod:
            env._set_executable_permissions(p)
            mock_chmod.assert_called_once_with(p, 0o775)


# =========================================================================
# STORMS / RECHARGE
# =========================================================================
class TestStorms:

    def test_init_storms(self, env):
        env._init_storms()
        assert len(env.stormStarts) == env.nStorms
        assert len(env.stormDurations) == env.nStorms
        assert len(env.stormIntensities) == env.nStorms

    def test_init_recharge_time_series(self, env):
        env._init_recharge_time_series()
        assert len(env.rechTimeSeries) == env.nstp + 1

    def test_build_observations_storms(self, env):
        env.timeStep = 3
        obs = env.build_observations_storms()
        # each storm contributes 6 values
        assert len(obs) == env.nStorms * 6


# =========================================================================
# WELL PACKAGE
# =========================================================================
class TestWellPackage:

    def test_set_main_well_cell_indices(self, env):
        with patch.object(env, 'cellInfoFromCoordinates', return_value=(1, 3, 2)):
            env._setMainWellCellIndices()
        assert env.wellCellLayer == 1
        assert env.wellCellColumn == 3
        assert env.wellCellRow == 2

    @patch('flopyarcade.modflow_runner.ModflowWel')
    def test_create_well_package_no_helpers(self, MockWel, env):
        env.ENVTYPE = '1s-d'
        env.mf = MagicMock()
        env._createWellPackage()
        MockWel.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowWel')
    def test_create_well_package_with_helpers(self, MockWel, env):
        env.ENVTYPE = '4s-d'
        env.mf = MagicMock()
        env._createWellPackage()
        MockWel.assert_called_once()
        kw = MockWel.call_args.kwargs
        spd = kw['stress_period_data']
        assert len(spd[0]) == 1 + env.nHelperWells

    def test_get_well_cells(self, env):
        cells = env.get_well_cells()
        assert cells.dtype == np.int32
        assert len(cells) == env.nHelperWells

    def test_get_wel_x(self, env):
        env.mf6 = MagicMock()
        env.mf6.get_var_address.return_value = 'TAG'
        env.mf6.get_value.return_value = np.array([1, 2, 3])
        tag, val = env.get_wel_x()
        assert tag == 'TAG'
        np.testing.assert_array_equal(val, [1, 2, 3])

    def test_update_wel_x(self, env):
        env.mf6 = MagicMock()
        new_vals = np.array([4, 5, 6])
        env.update_wel_x('TAG', new_vals)
        env.mf6.set_value.assert_called_once_with('TAG', new_vals)

    def test_update_well_cells(self, env):
        env.mf6 = MagicMock()
        env.mf6.get_var_address.return_value = 'TAG'
        env.mf6.get_value.return_value = np.array([0.0, 0.0])
        env.update_well_cells()
        env.mf6.set_value.assert_called_once()


# =========================================================================
# MF6 MODEL LOADING
# =========================================================================
class TestMF6Loading:

    @patch('flopyarcade.modflow_runner.XmiWrapper')
    @patch('flopyarcade.modflow_runner.deepcopy')
    def test_load_mf6_model_success(self, mock_dc, MockXmi, env):
        mock_wrapper = MagicMock()
        MockXmi.return_value = mock_wrapper
        mock_dc.return_value = mock_wrapper
        result = env.load_mf6_model('mf6.exe', 'testmodel')
        assert result is mock_wrapper

    @patch('flopyarcade.modflow_runner.XmiWrapper')
    def test_load_mf6_model_failure(self, MockXmi, env):
        MockXmi.side_effect = Exception("bad exe")
        result = env.load_mf6_model('bad.exe', 'testmodel')
        assert result == 'BMI_RETURN'

    def test_initialize_simulation_success(self, env):
        env.mf6 = MagicMock()
        result = env.initialize_simulation('testmodel')
        assert result is None
        env.mf6.initialize.assert_called_once_with('testmodel.nam')

    def test_initialize_simulation_failure(self, env):
        env.mf6 = MagicMock()
        env.mf6.initialize.side_effect = Exception("init fail")
        result = env.initialize_simulation('testmodel')
        assert result == 'BMI_RETURN'

    def test_setup_workspace_and_model(self, env):
        with patch('flopyarcade.modflow_runner.chdir'):
            env.setup_workspace_and_model()
        assert env.success is False
        assert env.nameUpper == 'TESTMODEL'

    def test_load_and_initialize_model(self, env):
        with (patch.object(env, "load_mf6_model") as lm,
             patch.object(env, "initialize_simulation") as iz):
            env.load_and_initialize_model('exe', 'model')
            lm.assert_called_once_with('exe', 'model')
            iz.assert_called_once_with('model')


# =========================================================================
# STEADY STATE / TRANSIENT
# =========================================================================
class TestSteadyTransient:

    def test_run_steady_state_solution(self, env):
        env.sim_steadyState = MagicMock()
        env.mf6 = MagicMock()
        env.mf6.get_time_step.return_value = 1.0
        env.mf6.get_var_address.return_value = 'X_TAG'
        env.mf6.get_value_ptr.return_value = np.ones(40)
        env.head_steadyState = None
        with patch.object(env, 'initialize_simulation'), \
             patch.object(env, 'get_max_iterations'), \
             patch.object(env, 'run_convergence_loop', return_value=True):
            env.run_steady_state_solution('exe', 'model')
        assert env.head_steadyState is not None
        env.mf6.finalize.assert_called()

    def test_run_steady_state_convergence_fail(self, env):
        env.sim_steadyState = MagicMock()
        env.mf6 = MagicMock()
        env.mf6.get_time_step.return_value = 1.0
        with patch.object(env, 'initialize_simulation'), \
             patch.object(env, 'get_max_iterations'), \
             patch.object(env, 'run_convergence_loop', return_value=False):
            result = env.run_steady_state_solution('exe', 'model')
        assert result == 'CONV_FAIL'

    def test_prepare_transient_simulation(self, env):
        env.sim = MagicMock()
        env.head_steadyState = np.ones((2, 4, 5))
        with patch.object(env, 'load_mf6_model'), \
             patch.object(env, 'initialize_simulation'), \
             patch.object(env, 'get_simulated_heads'):
            env.prepare_transient_simulation('exe', 'model')
        env.sim.write_simulation.assert_called_with(silent=True)

    def test_retrieve_boundary_conditions(self, env):
        env.mf6 = MagicMock()
        env.mf6.get_value.return_value = np.ones(5)
        env.mf6.get_var_address.return_value = 'BC_TAG'
        result = env.retrieve_boundary_conditions()
        assert result is not None

    def test_get_simulated_heads(self, env):
        env.mf6 = MagicMock()
        env.mf6.get_var_address.return_value = 'X_TAG'
        env.mf6.get_value_ptr.return_value = np.ones(20)
        env.get_simulated_heads()
        assert env.head is not None

    def test_get_recharge(self, env):
        env.mf6 = MagicMock()
        env.mf6.get_value.return_value = 0.5
        env.mf6.get_var_address.return_value = 'RCH_TAG'
        rch = env.get_recharge()
        assert rch == 0.5

    def test_finalize_simulation(self, env):
        env.mf6 = MagicMock()
        env.finalize_simulation()
        assert env.success is True

    def test_finalize_simulation_error(self, env):
        env.mf6 = MagicMock()
        env.mf6.finalize.side_effect = Exception("err")
        env.finalize_simulation()
        assert env.success is True

    def test_post_simulation_teardown_yes(self, env):
        with patch('flopyarcade.modflow_runner.chdir'):
            env.post_simulation_teardown(True)
        assert env.teardownSuccess is True

    def test_post_simulation_teardown_no(self, env):
        with patch('flopyarcade.modflow_runner.chdir'):
            env.post_simulation_teardown(False)


# =========================================================================
# MODEL UPDATE
# =========================================================================
class TestModelUpdate:

    def test_update_model(self, env):
        with patch.object(env, 'constructModel') as cm:
            env.updateModel()
            cm.assert_called_once()

    def test_update_cell_info_no_helpers(self, env):
        env.ENVTYPE = '1s-d'
        with patch.object(env, 'cellInfoFromCoordinates', return_value=(1, 2, 3)):
            env._update_cell_info()
        assert env.wellCellLayer == 1

    def test_update_cell_info_with_helpers(self, env):
        env.ENVTYPE = '4s-d'
        with patch.object(env, 'cellInfoFromCoordinates', return_value=(2, 3, 4)):
            env._update_cell_info()
        assert env.wellCellLayer == 2
        assert env.helperWells['l1'] == 2

    @patch('flopyarcade.modflow_runner.ModflowWel')
    def test_update_modflow_wel_no_helpers(self, MockWel, env):
        env.ENVTYPE = '1s-d'
        env.mf = MagicMock()
        env._update_modflow_wel()
        MockWel.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowWel')
    def test_update_modflow_wel_with_helpers(self, MockWel, env):
        env.ENVTYPE = '6r-c'
        env.mf = MagicMock()
        env._update_modflow_wel()
        MockWel.assert_called_once()
        kw2 = MockWel.call_args.kwargs
        spd = kw2['stress_period_data']
        assert len(spd[0]) == 1 + env.nHelperWells


# =========================================================================
# CONSTRUCT MODEL
# =========================================================================
class TestConstructModel:

    def test_construct_model_0sc(self, env):
        env.ENVTYPE = '0s-c'
        with patch.object(env, 'build_model_0sc') as bm:
            env.constructModel()
            bm.assert_called_once()

    def test_construct_model_other(self, env):
        env.ENVTYPE = '1s-d'
        with patch.object(env, 'build_model_other') as bm:
            env.constructModel()
            bm.assert_called_once()

    def test_initialize_model(self, env):
        with patch.object(env, 'constructModel') as cm:
            env.initializeModel()
            cm.assert_called_once()


# =========================================================================
# 0s-c MODEL BUILDING (MF6)
# =========================================================================
class TestBuild0SC:

    @patch('flopyarcade.modflow_runner.MFSimulation')
    def test_create_simulation_0sc(self, MockSim, env):
        env.create_simulation_0sc()
        MockSim.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowTdis')
    def test_create_tdis_0sc_steady(self, MockTdis, env):
        env.sim = MagicMock()
        env.create_tdis_0sc(steady_state=True)
        MockTdis.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowTdis')
    def test_create_tdis_0sc_transient(self, MockTdis, env):
        env.sim = MagicMock()
        env.create_tdis_0sc(steady_state=False)
        MockTdis.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowIms')
    def test_create_ims_0sc(self, MockIms, env):
        env.sim = MagicMock()
        env.create_ims_0sc()
        MockIms.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowGwf')
    def test_create_gwf_model_0sc(self, MockGwf, env):
        env.sim = MagicMock()
        env.create_gwf_model_0sc()
        MockGwf.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowGwfdis')
    @patch('flopyarcade.modflow_runner.ModflowGwfic')
    def test_setup_dis_ic_0sc(self, MockIc, MockDis, env):
        env.gwf = MagicMock()
        env.setup_dis_ic_0sc()
        MockDis.assert_called_once()
        MockIc.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowGwfchd')
    def test_setup_chd_0sc(self, MockChd, env):
        env.gwf = MagicMock()
        env.setup_chd_0sc()
        MockChd.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowGwfnpf')
    def test_setup_npf_0sc(self, MockNpf, env):
        env.gwf = MagicMock()
        env.setup_npf_0sc()
        MockNpf.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowGwfoc')
    def test_setup_oc_0sc(self, MockOc, env):
        env.gwf = MagicMock()
        env.setup_oc_0sc()
        MockOc.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowGwfsto')
    def test_setup_sto_0sc(self, MockSto, env):
        env.gwf = MagicMock()
        env.setup_sto_0sc()
        MockSto.assert_called_once()

    def test_build_steady_state_sim_0sc(self, env):
        env.rechTimeSeries = [np.zeros((2, 4, 5))]
        with patch('flopyarcade.modflow_runner.MFSimulation'),              patch('flopyarcade.modflow_runner.ModflowTdis'),              patch('flopyarcade.modflow_runner.ModflowIms'),              patch('flopyarcade.modflow_runner.ModflowGwf'),              patch('flopyarcade.modflow_runner.ModflowGwfdis'),              patch('flopyarcade.modflow_runner.ModflowGwfic'),              patch('flopyarcade.modflow_runner.ModflowGwfchd'),              patch('flopyarcade.modflow_runner.ModflowGwfnpf'),              patch('flopyarcade.modflow_runner.ModflowGwfoc'),              patch('flopyarcade.modflow_runner.deepcopy', return_value=MagicMock()),              patch.object(env, 'setup_rch_0sc'):
            env.build_steady_state_sim_0sc()
        assert env.sim_steadyState is not None

    @patch('flopyarcade.modflow_runner.ModflowGwfsto')
    @patch('flopyarcade.modflow_runner.ModflowTdis')
    def test_build_transient_sim_0sc(self, MockTdis, MockSto, env):
        env.sim = MagicMock()
        env.tdis = MagicMock()
        env.gwf = MagicMock()
        env.rechTimeSeries = [np.zeros((2, 4, 5))]
        with patch.object(env, 'setup_rch_0sc'), \
             patch.object(env, 'setup_wells_0sc'), \
             patch.object(env, 'create_tdis_0sc'):
            env.build_transient_sim_0sc()
        env.sim.remove_package.assert_called_once_with(env.tdis)

    @patch.object(DummyEnv, 'build_transient_sim_0sc')
    @patch.object(DummyEnv, 'build_steady_state_sim_0sc')
    def test_build_model_0sc(self, m_ss, m_tr, env):
        env.build_model_0sc()
        m_ss.assert_called_once()
        m_tr.assert_called_once()


# =========================================================================
# OTHER MODEL BUILDING (MODFLOW 2005)
# =========================================================================
class TestBuildOther:

    @patch('flopyarcade.modflow_runner.Modflow')
    def test_create_model_other(self, MockMF, env):
        MockMF.return_value = MagicMock()
        env.create_model_other()
        MockMF.assert_called_once_with('MODEL001', exe_name='mf2005.exe', verbose=False)
        env.mf.change_model_ws.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowBas')
    def test_add_bas_package_other(self, MockBas, env):
        env.mf = MagicMock()
        env.add_bas_package_other()
        MockBas.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowLpf')
    def test_add_lpf_package_other(self, MockLpf, env):
        env.mf = MagicMock()
        env.add_lpf_package_other()
        MockLpf.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowOc')
    def test_add_oc_package_other(self, MockOc, env):
        env.mf = MagicMock()
        env.add_oc_package_other()
        MockOc.assert_called_once()

    @patch('flopyarcade.modflow_runner.ModflowPcg')
    def test_add_solver_package_other(self, MockPcg, env):
        env.mf = MagicMock()
        env.add_solver_package_other()
        MockPcg.assert_called_once_with(env.mf)

    @patch('flopyarcade.modflow_runner.Modflow')
    def test_build_model_other(self, MockMF, env):
        env.mf = MagicMock()
        with patch.object(env, 'create_discretization_other'), \
             patch.object(env, 'initialize_ibound_other'), \
             patch.object(env, 'initialize_start_heads_other'), \
             patch.object(env, 'add_bas_package_other'), \
             patch.object(env, 'add_lpf_package_other'), \
             patch.object(env, 'add_oc_package_other'), \
             patch.object(env, 'add_solver_package_other'):
            env.build_model_other()


# =========================================================================
# CELL GEOMETRY
# =========================================================================
class TestCellGeometry:

    def test_get_cellid_valid(self, env):
        cid = env.get_cellID(25.0, 35.0)
        assert isinstance(cid, int)
        assert cid >= 1

    def test_get_cellid_x_out_of_bounds(self, env):
        with pytest.raises(ValueError, match='x coordinate'):
            env.get_cellID(999.0, 0.0)

    def test_get_cellid_y_out_of_bounds(self, env):
        with pytest.raises(ValueError, match='y coordinate'):
            env.get_cellID(0.0, 999.0)

    def test_get_cellcolrow_valid(self, env):
        col, row = env.get_cellColRow(25.0, 35.0)
        assert col >= 0
        assert row >= 0

    def test_get_cellcolrow_out_of_bounds(self, env):
        with pytest.raises(ValueError):
            env.get_cellColRow(999.0, 0.0)

    def test_cellinfofromcoordinates(self, env):
        layer, col, row = env.cellInfoFromCoordinates([50.0, 35.0, -5.0])
        assert layer >= 1
        assert col >= 1
        assert row >= 1


# =========================================================================
# RUN MODFLOW
# =========================================================================
class TestRunModflow:

    def test_run_modflow_no_check(self, env):
        env.timeStep = 0
        env.mf = MagicMock()
        env.mf.run_model.return_value = (True, 'buffer')
        with patch.object(env, '_write_input') as wi, \
             patch.object(env, '_time_run_model') as trm, \
             patch.object(env, '_load_simulation_results') as lsr:
            env.runMODFLOW(check=False)
            wi.assert_called_once()
            trm.assert_called_once()
            lsr.assert_called_once()

    def test_run_modflow_with_check(self, env):
        env.timeStep = 0
        env.mf = MagicMock()
        env.mf.check.return_value = None
        with patch.object(env, '_write_input'), \
             patch.object(env, '_check_model') as cm, \
             patch.object(env, '_time_run_model'), \
             patch.object(env, '_load_simulation_results'):
            env.runMODFLOW(check=True)
            cm.assert_called_once()

    def test_write_input_step0(self, env):
        env.timeStep = 0
        env.mf = MagicMock()
        env._write_input()
        assert env.debug_written == 'full'
        env.mf.write_input.assert_called_once()

    def test_write_input_step_gt0(self, env):
        env.timeStep = 5
        env.mf_bas = MagicMock()
        env.wel = MagicMock()
        env._write_input()
        assert env.debug_written == 'partial'
        env.mf_bas.write_file.assert_called_once_with(check=False)
        env.wel.write_file.assert_called_once()

    def test_check_model(self, env):
        env.mf = MagicMock()
        env.mf.check.return_value = 'CHECK_OK'
        env._check_model()
        assert env.check == 'CHECK_OK'
        env.mf.check.assert_called_once_with(verbose=True)

    @patch('flopyarcade.modflow_runner.sleep')
    def test_time_run_model_success(self, mock_sleep, env):
        env.mf = MagicMock()
        env.mf.run_model.return_value = (True, 'buf')
        with patch('flopyarcade.modflow_runner.time', return_value=1.0):
            env._time_run_model()
        assert env.success_modflow is True
        assert env.buff == 'buf'

    @patch('flopyarcade.modflow_runner.sleep')
    def test_time_run_model_retry_success(self, mock_sleep, env):
        env.mf = MagicMock()
        env.mf.run_model.side_effect = [(False, 'b1'), (True, 'b2')]
        with patch('flopyarcade.modflow_runner.time', return_value=1.0):
            env._time_run_model()
        assert env.success_modflow is True

    @patch('flopyarcade.modflow_runner.sleep')
    def test_time_run_model_all_retries_fail(self, mock_sleep, env):
        env.mf = MagicMock()
        env.mf.run_model.return_value = (False, 'fail')
        with patch('flopyarcade.modflow_runner.time', return_value=1.0):
            with pytest.raises(RuntimeError, match='did not terminate'):
                env._time_run_model()

    @patch('flopyarcade.modflow_runner.HeadFile')
    def test_load_simulation_results(self, MockHF, env):
        mock_hf = MagicMock()
        MockHF.return_value.__enter__ = lambda s: mock_hf
        MockHF.return_value.__exit__ = MagicMock(return_value=False)
        mock_hf.get_times.return_value = [0.0, 1.0, 2.0]
        mock_hf.get_data.return_value = np.ones((2, 4, 5))
        env._load_simulation_results()
        assert len(env.times) == 3

    @patch('flopyarcade.modflow_runner.PlotMapView')
    @patch('flopyarcade.modflow_runner.MfGrdFile')
    def test_load_grid(self, MockGrd, MockMap, env):
        mock_grd = MagicMock()
        MockGrd.return_value = mock_grd
        mock_grd.modelgrid = 'MG'
        env._load_grid()
        MockMap.assert_called_once_with(modelgrid='MG', ax=env.ax)


# =========================================================================
# OBSERVATIONS / NORMALIZATION
# =========================================================================
class TestObservations:

    def test_set_basic_observations(self, env):
        env.heads = np.ones((2, 4, 5))
        env.particleCoords = np.array([1.0, 2.0])
        env.sampleHeadsEvery = 1
        env.observations = {}
        env._set_basic_observations()
        assert 'particleCoords' in env.observations
        assert 'heads' in env.observations
        assert 'headsSampledField' in env.observations

    def test_normalize_basic_stresses(self, env):
        env.minH = 0.0
        env.maxH = 100.0
        result = env._normalize_basic_stresses([0, 50, 100])
        assert len(result) == 3


# =========================================================================
# DLL LOADING
# =========================================================================
class TestDllLoading:

    def test_load_temporary_dll(self, env):
        src = os.path.join(env.model_ws, 'test.dll')
        with open(src, 'w') as f:
            f.write('fake dll')
        import flopyarcade.modflow_runner as mr
        mock_tf = MagicMock()
        mock_sh = MagicMock()
        tmp = MagicMock()
        tmp.name = '/tmp/fake.dll'
        mock_tf.NamedTemporaryFile.return_value = tmp
        # temporarily inject module-level names the method expects
        mr.tempfile = mock_tf
        mr.shutil = mock_sh
        try:
            result = env.load_temporary_dll(src)
            assert result == '/tmp/fake.dll'
        finally:
            del mr.tempfile
            del mr.shutil


# =========================================================================
# SIM DEEP COPY PATCH for build_steady_state
# =========================================================================
@pytest.fixture(autouse=True)
def _patch_deepcopy():
    """Ensure deepcopy works on MagicMock objects used as sim."""
    _orig = deepcopy
    def _safe_deepcopy(obj, *a, **kw):
        if isinstance(obj, MagicMock):
            return MagicMock()
        return _orig(obj, *a, **kw)
    import flopyarcade.modflow_runner as mr
    mr.deepcopy = _safe_deepcopy
    yield
    mr.deepcopy = _orig

