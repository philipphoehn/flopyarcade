"""High-coverage tests for flopyarcade.env_base.EnvBaseMixin.

All external dependencies (flopy, matplotlib, tensorflow, etc.) are mocked.
No actual MODFLOW simulation is run.
"""
import sys
import os
import hashlib
from unittest.mock import MagicMock, patch, PropertyMock, call
from copy import deepcopy
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Mock all heavy / external imports BEFORE importing env_base
# ---------------------------------------------------------------------------

mock_flopy = MagicMock()
mock_flopy_modflow = MagicMock()
mock_flopy_mf6 = MagicMock()
mock_flopy_mf6_modflow = MagicMock()
mock_flopy_mf6_modflow_mfgwfwel = MagicMock()
mock_flopy_modpath = MagicMock()
mock_flopy_plot = MagicMock()
mock_flopy_utils = MagicMock()
mock_flopy_mf6_utils = MagicMock()
mock_flopy_mf6_utils_binarygrid_util = MagicMock()
mock_flopy_utils_binaryfile = MagicMock()

for mod_name, mock in [
    ('flopy', mock_flopy),
    ('flopy.modflow', mock_flopy_modflow),
    ('flopy.mf6', mock_flopy_mf6),
    ('flopy.mf6.modflow', mock_flopy_mf6_modflow),
    ('flopy.mf6.modflow.mfgwfwel', mock_flopy_mf6_modflow_mfgwfwel),
    ('flopy.modpath', mock_flopy_modpath),
    ('flopy.plot', mock_flopy_plot),
    ('flopy.utils', mock_flopy_utils),
    ('flopy.utils.binaryfile', mock_flopy_utils_binaryfile),
    ('flopy.mf6.utils', mock_flopy_mf6_utils),
    ('flopy.mf6.utils.binarygrid_util', mock_flopy_mf6_utils_binarygrid_util),
]:
    sys.modules.setdefault(mod_name, mock)

for mod_name in [
    'matplotlib', 'matplotlib.ticker', 'matplotlib.pyplot',
    'matplotlib.colors', 'matplotlib.transforms', 'matplotlib.text',
    'matplotlib.cm', 'imageio', 'joblib', 'xmipy',
    'IPython', 'IPython.display', 'tqdm',
]:
    sys.modules.setdefault(mod_name, MagicMock())


mock_pathos = MagicMock()
mock_pathos.multiprocessing = MagicMock()
mock_pathos.multiprocessing.cpu_count = MagicMock(return_value=4)
mock_pathos.helpers = MagicMock()
mock_pathos.pools = MagicMock()
for mod_name in ['pathos', 'pathos.multiprocessing', 'pathos.pools', 'pathos.helpers']:
    sys.modules.setdefault(mod_name, mock_pathos if mod_name == 'pathos' else getattr(mock_pathos, mod_name.split('.')[-1], MagicMock()))

mock_tf = MagicMock()
mock_tf.keras = MagicMock()
mock_tf.compat = MagicMock()
for mod_name in [
    'tensorflow', 'tensorflow.keras', 'tensorflow.keras.initializers',
    'tensorflow.keras.layers', 'tensorflow.keras.models',
    'tensorflow.keras.optimizers', 'tensorflow.keras.random',
    'tensorflow.keras.backend', 'tensorflow.compat', 'tensorflow.compat.v1',
]:
    sys.modules.setdefault(mod_name, MagicMock())

# ---------------------------------------------------------------------------
# Import the module under test (bypass package __init__ to avoid metaclass conflicts)
# ---------------------------------------------------------------------------
import importlib.util
import types
import os as _os

_pkg_path = _os.path.join(_os.path.dirname(__file__), '..', 'env_base.py')
_pkg_path = _os.path.abspath(_pkg_path)

_pkg = types.ModuleType('flopyarcade')
_pkg.__path__ = [_os.path.dirname(_pkg_path)]
if 'flopyarcade' not in sys.modules:
    sys.modules['flopyarcade'] = _pkg

_spec = importlib.util.spec_from_file_location('flopyarcade.env_base', _pkg_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['flopyarcade.env_base'] = _mod
_spec.loader.exec_module(_mod)
EnvBaseMixin = _mod.EnvBaseMixin

# Clean up fake sys.modules entries to prevent polluting other test files
# KEEP flopyarcade.env_base in sys.modules (needed for unittest.mock.patch targets)
# Only remove the bare flopyarcade ModuleType that lacks the real package attributes
if 'flopyarcade' in sys.modules and isinstance(sys.modules['flopyarcade'], types.ModuleType) and not hasattr(sys.modules['flopyarcade'], 'FloPyEnv'):
    del sys.modules['flopyarcade']



# ---------------------------------------------------------------------------
# Helper: create a MagicMock that masquerades as an EnvBaseMixin instance
# ---------------------------------------------------------------------------

def make_env(**overrides):
    """Create a MagicMock with all commonly-needed attributes set."""
    env = MagicMock(spec=EnvBaseMixin)
    defaults = {
        'ENVTYPE': '1s-d',
        'nLay': 1, 'nRow': 100, 'nCol': 100,
        'dRow': 1.0, 'dCol': 1.0, 'dVer': 50.0,
        'zTop': 50.0, 'zBot': 0.0,
        'botM': np.array([50.0, 0.0]),
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
        'done': False, 'truncated': False, 'closed': False,
        'reward': 0.0, 'rewardCurrent': 0.0, 'success': True,
        'info': {}, 'observations': [1.0, 2.0, 3.0],
        'observationsVectorNormalized': np.array([1.0, 2.0, 3.0]),
        'actionType': 'discrete',
        'actionSpace': ['up1', 'down1', 'left1', 'right1', 'Qup1', 'Qdown1'],
        'actionSpaceSize': 6, 'actionsDict': {},
        'actionValueNorth': 58.0, 'actionValueSouth': 58.0,
        'actionValue': 58.0, 'actionValueX': 50.0, 'actionValueY': 50.0,
        'particleCoords': [50.0, 50.0, 0.0],
        'periodSteadiness': True, 'periods': 1, 'periodLength': 1.0, 'periodSteps': 11,
        'RENDER': False, 'SAVEPLOT': False, 'MANUALCONTROL': False,
        'MANUALCONTROLTIME': 0.1, 'renderFlag': False, 'delFiles': True,
        'envhash': '', 'teardownSuccess': True,
        'maxQChange': 50.0, 'maxHChange': 0.2, 'maxCoordChange': 2.5,
        'wellRadius': 2.83,
        'wellSpawnBufferXWest': 50.0, 'wellSpawnBufferXEast': 20.0,
        'wellSpawnBufferY': 20.0,
        'dt': 1.0, 'current_time': 0.0, 'max_iter': 100,
        'trajectories': {'x': [], 'y': [], 'z': []},
        'keyPressed': None, 'state': {},
        'nStorms': 0, 'nper': 1, 'nstp': 50, 'perlen': 50,
        'nlay': 1, 'nrow': 25, 'ncol': 25,
        'delr': 100.0, 'delc': 100.0, 'top': 50.0, 'botm': 0.0, 'hk': 0.05,
        'strt': 58.0, 'chd_west': 58.5, 'chd_east': 57.5,
        'well_dxMax': 200.0, 'well_dyMax': 200.0, 'well_dQMax': 5.0,
        'diffQ': 30.0, 'avg_rch_min': 0.0, 'avg_rch_max': 0.0,
        'storm_rch_min': 0.1, 'storm_rch_max': 20.0, 'rch_diff': 20.0,
        'minStormDuration': 3, 'maxStormDuration': 20,
        'minStormRadius': 300, 'maxStormRadius': 1000,
        'shapeLayer': (1, 25, 25), 'shapeGrid': (1, 25, 25),
        'rech0': np.zeros((1, 25, 25)),
        'domainArea': 10000.0, 'nouter': 100, 'ninner': 100,
        'hclose': 1e-08, 'rclose': 1e-08, 'relax': 1.0,
        'sampleHeadsEvery': 3, 'initWithSolution': True,
        'versionMODFLOW': 'mf2005', 'versionMODPATH': 'mp6',
        'PATHMF2005': None, 'PATHMP6': None, 'PATHMF6DLL': None,
        'OBSPREP': 'perceptron', 'ANIMATIONFOLDER': 'Test',
        'solver_backend': 'mf2005', 'env_config': {}, '_SEED': 42,
        'exdir': 'TestModel', 'name': 'TestModel', 'simpath': None,
        'mf6dll': None, 'well_tag': 'WEL_TAG', 'rch_tag': 'RCH_TAG',
        'well': [1.0, 2.0], 'mf6': MagicMock(), 'gwf': MagicMock(),
        'mf': MagicMock(), 'dis': MagicMock(),
        'ibound': np.ones((1, 100, 100), dtype=np.int32),
        'headsPrev': np.ones((1, 100, 100), dtype=np.float32),
        'tempDir': MagicMock(),
        '_obs_hasher': hashlib.sha256(),
        'env_hash': '',
        'actionSpaceIndividual': ['up', 'down', 'left', 'right', 'Qup', 'Qdown'],
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        try:
            setattr(env, k, v)
        except (AttributeError, TypeError):
            env.__dict__[k] = v
    # Mock internal helper methods
    for attr in [
        '_feed_obs_hash', 'update_envhash', 'handle_done', 'render',
        'cleanup_temp_dir', 'finalize_simulation', 'post_simulation_teardown',
        'update_reward_on_fail', '_resolve_well_overlap',
        'cellInfoFromCoordinates', '_setMainWellCellIndices', '_createWellPackage',
        '_update_main_well_location', '_update_helper_wells_location',
        '_update_helper_wells_Q', '_update_cell_info', '_update_modflow_wel',
        '_update_random_head_fluctuations', '_assign_boundary_heads',
        'setActionValue', 'initializeState', 'updateModel',
        'runMODFLOW', 'runMODPATH', 'evaluateParticleTracking',
        'calculate_reward', 'update_state', 'get_normalized_observations',
        'get_normalized_stresses', 'check_particle_state',
        'changeActionDict', 'getActionValues', 'update_well_Q',
        'update_well_cells', 'get_simulated_heads', 'get_recharge',
        'calculateGameRewardHeadChange', 'prepare_observations',
        'getActionType', 'getActionSpaceSize', 'get_action_value',
        'initializeParticle', 'initializeModel', 'initializeAction',
        'initialize_helper_wells', '_init_helper_wells', '_init_storms',
        '_init_recharge_time_series', '_setup_workspace', 'constructModel',
        'setup_workspace_and_model', 'load_mf6_model',
        'run_steady_state_solution', 'prepare_transient_simulation',
        'retrieve_boundary_conditions', 'example_feature_extractor',
    ]:
        setattr(env, attr, MagicMock())
    env.changeActionDict.return_value = {}
    env.getActionValues.return_value = {}
    env.get_recharge.return_value = 1.0
    env.calculateGameRewardHeadChange.return_value = 1.0
    env.getActionType.return_value = 'discrete'
    env.getActionSpaceSize.return_value = 6
    env.get_action_value.return_value = 1.0
    env.cellInfoFromCoordinates.return_value = (1, 1, 1)
    env.example_feature_extractor.return_value = np.array([1.0, 2.0, 3.0])
    env.rechTimeSeries = [1.0] * 10
    env.head_steadyState_flat = np.zeros(25)
    env.head = np.zeros(25)
    env._last_action = 'up1'
    return env


# ============================================================================
# Tests: _init_env_config
# ============================================================================

class TestInitEnvConfig:
    def test_basic_config(self):
        env = make_env()
        local_vars = {
            'ENVTYPE': '1s-d', 'PATHMF2005': None, 'PATHMP6': None,
            'MODELNAME': 'Test', 'ANIMATIONFOLDER': 'Anim',
            'flagSavePlot': False, 'flagManualControl': False,
            'manualControlTime': 0.1, 'flagRender': False,
            'NAGENTSTEPS': 100, 'nLay': 1, 'nRow': 50, 'nCol': 50,
            'OBSPREP': 'perceptron', 'initWithSolution': True,
            'PATHMF6DLL': None, '_seed': 42,
        }
        EnvBaseMixin._init_env_config(env, local_vars, None)
        assert env.env_config['ENVTYPE'] == '1s-d'
        assert env.ENVTYPE == '1s-d'
        assert env.NAGENTSTEPS == 100
        assert env.SAVEPLOT is False
        assert env.MANUALCONTROL is False

    def test_config_override(self):
        env = make_env()
        local_vars = {
            'ENVTYPE': '1s-d', 'PATHMF2005': None, 'PATHMP6': None,
            'MODELNAME': 'Test', 'ANIMATIONFOLDER': 'Anim',
            'flagSavePlot': False, 'flagManualControl': False,
            'manualControlTime': 0.1, 'flagRender': False,
            'NAGENTSTEPS': 100, 'nLay': 1, 'nRow': 50, 'nCol': 50,
            'OBSPREP': 'perceptron', 'initWithSolution': True,
            'PATHMF6DLL': None, '_seed': None,
        }
        EnvBaseMixin._init_env_config(env, local_vars, {'NAGENTSTEPS': 500})
        assert env.NAGENTSTEPS == 500


# ============================================================================
# Tests: _init_env_vars
# ============================================================================

class TestInitEnvVars:
    def test_basic(self):
        env = make_env()
        EnvBaseMixin._init_env_vars(env, 42)
        assert env.info == {}
        assert env.done is False
        assert env.closed is False
        assert env._SEED == 42
        assert env.timeStep == 0
        assert env.reward == 0.0


# ============================================================================
# Tests: _init_env_specific
# ============================================================================

class TestInitEnvSpecific:
    def test_dispatch_0sc(self):
        env = make_env(ENVTYPE='0s-c')
        env._init_env_0s_c = MagicMock()
        env._init_env_other = MagicMock()
        EnvBaseMixin._init_env_specific(env)
        env._init_env_0s_c.assert_called_once()
        env._init_env_other.assert_not_called()

    def test_dispatch_other(self):
        env = make_env(ENVTYPE='1s-d')
        env._init_env_0s_c = MagicMock()
        env._init_env_other = MagicMock()
        EnvBaseMixin._init_env_specific(env)
        env._init_env_other.assert_called_once()
        env._init_env_0s_c.assert_not_called()


# ============================================================================
# Tests: _setup_gym_spaces
# ============================================================================

class TestSetupGymSpaces:
    def test_discrete(self):
        env = make_env(actionType='discrete', actionSpace=['a', 'b', 'c'])
        EnvBaseMixin._setup_gym_spaces(env)
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_continuous(self):
        env = make_env(actionType='continuous', actionSpaceSize=4, ENVTYPE='3s-c')
        EnvBaseMixin._setup_gym_spaces(env)
        assert env.action_space is not None


# ============================================================================
# Tests: seed, close, teardown
# ============================================================================

class TestLifecycle:
    def test_seed_with_value(self):
        env = make_env()
        EnvBaseMixin.seed(env, 123)
        assert env._SEED == 123

    def test_seed_none(self):
        env = make_env()
        EnvBaseMixin.seed(env, None)

    def test_close(self):
        env = make_env()
        EnvBaseMixin.close(env)
        assert env.closed is True

    def test_teardown_success(self):
        env = make_env(delFiles=True, teardownSuccess=True)
        env._change_to_workspace = MagicMock()
        env.finalize_simulation = MagicMock()
        env._remove_model_workspace = MagicMock()
        EnvBaseMixin.teardown(env)
        env._remove_model_workspace.assert_called_once()

    def test_teardown_retain_files(self):
        env = make_env(delFiles=False)
        env._change_to_workspace = MagicMock()
        EnvBaseMixin.teardown(env)


# ============================================================================
# Tests: bmi_return, create_single_hash
# ============================================================================

class TestBMIAndHash:
    def test_bmi_return(self, tmp_path):
        env = make_env()
        stdout_file = tmp_path / 'mfsim.stdout'
        stdout_file.write_text('line1\nline2\n')
        result = EnvBaseMixin.bmi_return(env, True, str(tmp_path), 'test')
        assert result[0] is True
        assert len(result[1]) == 2

    def test_bmi_return_missing(self):
        env = make_env()
        with pytest.raises(Exception):
            EnvBaseMixin.bmi_return(env, True, '/nonexistent/path', 'test')

    def test_create_single_hash(self):
        env = make_env()
        result = EnvBaseMixin.create_single_hash(env, [np.array([1, 2, 3]), np.array([4, 5, 6])])
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex


# ============================================================================
# Tests: defineEnvironment, _setup_0sc, _setup_general_env
# ============================================================================

class TestDefineEnvironment:
    def test_0sc(self):
        env = make_env(ENVTYPE='0s-c')
        env._setup_0sc = MagicMock()
        env._setup_general_env = MagicMock()
        env._setup_action_space = MagicMock()
        EnvBaseMixin.defineEnvironment(env, 42)
        env._setup_0sc.assert_called_once()

    def test_other(self):
        env = make_env(ENVTYPE='1s-d')
        env._setup_0sc = MagicMock()
        env._setup_general_env = MagicMock()
        env._setup_action_space = MagicMock()
        EnvBaseMixin.defineEnvironment(env, 42)
        env._setup_general_env.assert_called_once()

    def test_setup_0sc_values(self):
        env = make_env(ENVTYPE='0s-c')
        EnvBaseMixin._setup_0sc(env)
        assert env.perlen == 50
        assert env.nrow == 25
        assert env.ncol == 25
        assert env.nHelperWells == 5

    def test_setup_general_env(self):
        env = make_env(ENVTYPE='1s-d', nCol=100, nRow=100, nLay=1)
        env._setup_envtype_specifics = MagicMock()
        EnvBaseMixin._setup_general_env(env)
        assert env.rewardMax == 1000
        assert env.minQ == -2000.0
        assert env.maxQ == -500.0

    def test_setup_envtype_specifics_type1(self):
        env = make_env(ENVTYPE='1s-d')
        env.wellRadius = 2.83
        EnvBaseMixin._setup_envtype_specifics(env)
        assert env.nHelperWells == 0

    def test_setup_envtype_specifics_type4(self):
        env = make_env(ENVTYPE='4s-d', wellRadius=2.83, maxSteps=200)
        EnvBaseMixin._setup_envtype_specifics(env)
        assert env.nHelperWells == 20
        assert env.minQhelper == -600.0

    def test_setup_envtype_specifics_random(self):
        env = make_env(ENVTYPE='4r-d', wellRadius=2.83, maxSteps=200)
        EnvBaseMixin._setup_envtype_specifics(env)
        assert env.maxHChange == 0.2
        assert env.maxQChange == 50.0


# ============================================================================
# Tests: update_head_specs_if_needed
# ============================================================================

class TestUpdateHeadSpecs:
    def test_needs_update(self):
        env = make_env(ENVTYPE='3s-d', minH=56.0, maxH=60.0)
        EnvBaseMixin.update_head_specs_if_needed(env)
        assert hasattr(env, 'headSpecNorth')

    def test_no_update_needed(self):
        env = make_env(ENVTYPE='1s-d')
        env.headSpecNorth = MagicMock(side_effect=AssertionError)
        EnvBaseMixin.update_head_specs_if_needed(env)


# ============================================================================
# Tests: initializeSimulators, _get_platform_info
# ============================================================================

class TestInitializeSimulators:
    def test_platform_info_linux(self):
        env = make_env()
        with patch('flopyarcade.env_base.platformSystem', return_value='Linux'):
            result = EnvBaseMixin._get_platform_info(env)
            assert result[0] == 'linux'
            assert result[3] is True

    def test_platform_info_windows(self):
        env = make_env()
        with patch('flopyarcade.env_base.platformSystem', return_value='Windows'):
            with patch('flopyarcade.env_base.sizeof', return_value=8):
                with patch('flopyarcade.env_base.c_voidp'):
                    result = EnvBaseMixin._get_platform_info(env)
                    assert result[0] == 'win64'

    def test_platform_info_unknown(self):
        env = make_env()
        with patch('flopyarcade.env_base.platformSystem', return_value='Amiga'):
            result = EnvBaseMixin._get_platform_info(env)
            assert result[0] is None

    def test_simulators_unknown_platform(self):
        env = make_env()
        env._get_platform_info = MagicMock(return_value=(None, None, None, None))
        EnvBaseMixin.initializeSimulators(env)


# ============================================================================
# Tests: _init_type_1 through _init_type_6
# ============================================================================

class TestInitTypes:
    def test_type_1(self):
        env = make_env(minH=56.0, maxH=60.0)
        EnvBaseMixin._init_type_1(env)
        assert 56.0 <= env.actionValueNorth <= 60.0
        assert 56.0 <= env.actionValueSouth <= 60.0

    def test_type_2(self):
        env = make_env(minH=56.0, maxH=60.0)
        EnvBaseMixin._init_type_2(env)
        assert 56.0 <= env.actionValue <= 60.0

    def test_type_3(self):
        env = make_env(wellX=50.0, wellY=40.0)
        EnvBaseMixin._init_type_3(env)
        assert env.actionValueX == 50.0
        assert env.actionValueY == 40.0

    def test_type_4(self):
        env = make_env(nHelperWells=2)
        env.helperWells = {'wellX1': 10.0, 'wellY1': 20.0, 'wellX2': 30.0, 'wellY2': 40.0}
        EnvBaseMixin._init_type_4(env)
        assert env.helperWells['actionValueX1'] == 10.0
        assert env.helperWells['actionValueY2'] == 40.0

    def test_type_5(self):
        env = make_env(nHelperWells=2)
        env.helperWells = {'wellQ1': -100.0, 'wellQ2': -200.0}
        EnvBaseMixin._init_type_5(env)
        assert env.helperWells['actionValueQ1'] == -100.0

    def test_type_6(self):
        env = make_env(nHelperWells=2)
        env.helperWells = {
            'wellX1': 10.0, 'wellY1': 20.0, 'wellQ1': -100.0,
            'wellX2': 30.0, 'wellY2': 40.0, 'wellQ2': -200.0,
        }
        EnvBaseMixin._init_type_6(env)
        assert env.helperWells['actionValueX1'] == 10.0
        assert env.helperWells['actionValueY1'] == 20.0
        assert env.helperWells['actionValueQ1'] == -100.0


# ============================================================================
# Tests: initializeWellRate, initializeWell
# ============================================================================

class TestInitializeWell:
    def test_initializeWellRate(self):
        env = make_env(
            extentX=100.0, extentY=100.0, zTop=50.0,
            wellSpawnBufferXWest=10.0, wellSpawnBufferXEast=10.0,
            wellSpawnBufferY=10.0,
        )
        result = EnvBaseMixin.initializeWellRate(env, -2000, -500)
        wx, wy, wz, wcoords, wq = result
        assert 10.0 <= wx <= 90.0
        assert 10.0 <= wy <= 90.0
        assert wz == 50.0
        assert -2000 <= wq <= -500

    def test_initializeWellRate_invalid(self):
        env = make_env()
        with pytest.raises(AssertionError):
            EnvBaseMixin.initializeWellRate(env, -500, -2000)

    def test_initializeWell(self):
        env = make_env(wellX=50, wellY=50, wellZ=50, wellQ=-1000)
        EnvBaseMixin.initializeWell(env)
        env._validateMainWellAttributes.assert_called()

    def test_validateMainWellAttributes_ok(self):
        env = make_env(wellX=50, wellY=50, wellZ=50, wellQ=-1000)
        EnvBaseMixin._validateMainWellAttributes(env)

    def test_validateMainWellAttributes_missing(self):
        env = make_env()
        del env.wellX
        del env.wellY
        del env.wellZ
        del env.wellQ
        with pytest.raises(AssertionError):
            EnvBaseMixin._validateMainWellAttributes(env)

    def test_envHasHelperWells_true(self):
        env = make_env(ENVTYPE='4s-d')
        assert EnvBaseMixin._envHasHelperWells(env) is True

    def test_envHasHelperWells_false(self):
        env = make_env(ENVTYPE='1s-d')
        assert EnvBaseMixin._envHasHelperWells(env) is False

    def test_initializeHelperWells(self):
        env = make_env(nHelperWells=2)
        env.helperWells = {'wellX1': 10, 'wellY1': 20, 'wellZ1': 50,
                           'wellX2': 30, 'wellY2': 40, 'wellZ2': 50}
        EnvBaseMixin._initializeHelperWells(env)
        assert env.helperWells['l1'] == 1
        assert env.helperWells['c1'] == 1

    def test_initializeHelperWells_missing(self):
        env = make_env(nHelperWells=1)
        env.helperWells = {'wellX1': 10, 'wellY1': 20}  # missing wellZ1
        with pytest.raises(AssertionError):
            EnvBaseMixin._initializeHelperWells(env)


# ============================================================================
# Tests: step, _first_step
# ============================================================================

class TestStep:
    def test_step_first_step(self):
        env = make_env(timeStep=0, initWithSolution=True, actionType='discrete')
        env.actionSpace = ['up1', 'down1']
        env._first_step = MagicMock(return_value=(np.zeros(3), 0.0, False, False, {}))
        result = EnvBaseMixin.step(env, 'up1')
        env._first_step.assert_called_once()

    def test_step_0sc(self):
        env = make_env(ENVTYPE='0s-c', timeStep=5, initWithSolution=False)
        env._handle_0sc_environment = MagicMock(return_value=(np.zeros(3), 1.0, False, False, {}))
        result = EnvBaseMixin.step(env, 'up1')
        env._handle_0sc_environment.assert_called_once()

    def test_step_other(self):
        env = make_env(ENVTYPE='1s-d', timeStep=5, initWithSolution=False)
        env._handle_other_environments = MagicMock(return_value=(np.zeros(3), 1.0, False, False, {}))
        result = EnvBaseMixin.step(env, 'up1')
        env._handle_other_environments.assert_called_once()

    def test_first_step_0sc(self):
        env = make_env(ENVTYPE='0s-c', renderFlag=False, SAVEPLOT=False)
        env.setup_workspace_and_model = MagicMock()
        env.load_mf6_model = MagicMock()
        env.run_steady_state_solution = MagicMock()
        env.prepare_transient_simulation = MagicMock()
        env.retrieve_boundary_conditions = MagicMock(return_value=1.0)
        env.prepare_observations = MagicMock()
        env._feed_obs_hash = MagicMock()
        result = EnvBaseMixin._first_step(env, 0.0, {})
        assert env.timeStep == 1

    def test_first_step_other(self):
        env = make_env(ENVTYPE='1s-d', renderFlag=False, SAVEPLOT=False)
        env.runMODFLOW = MagicMock()
        env.update_state = MagicMock()
        env.get_normalized_observations = MagicMock()
        env.get_normalized_stresses = MagicMock()
        env._feed_obs_hash = MagicMock()
        result = EnvBaseMixin._first_step(env, 0.0, {})
        assert env.timeStep == 1


# ============================================================================
# Tests: _handle_0sc_environment, _handle_other_environments
# ============================================================================

class TestHandleEnvironments:
    def test_0sc(self):
        env = make_env(ENVTYPE='0s-c', timeStep=1, nstp=50, renderFlag=False, SAVEPLOT=False)
        env._prepare_time_step = MagicMock()
        env.get_simulated_heads = MagicMock()
        env.get_recharge = MagicMock(return_value=1.0)
        env.update_well_cells = MagicMock()
        env.changeActionDict = MagicMock(return_value={})
        env.getActionValues = MagicMock(return_value={})
        env.update_well_Q = MagicMock()
        env.run_convergence_loop = MagicMock(return_value=True)
        env.calculateGameRewardHeadChange = MagicMock(return_value=1.0)
        env.prepare_observations = MagicMock()
        env._feed_obs_hash = MagicMock()
        env.update_envhash = MagicMock()
        env.handle_done = MagicMock()
        env.rechTimeSeries = [0.5] * 60
        env.head_steadyState_flat = np.zeros(25)
        env.head = np.ones(25)
        result = EnvBaseMixin._handle_0sc_environment(env, 'up1', 0.0, {}, True)
        assert env.timeStep == 2

    def test_other(self):
        env = make_env(ENVTYPE='1s-d', timeStep=1, RENDER=False, MANUALCONTROL=False, SAVEPLOT=False)
        env._step_apply_action = MagicMock()
        env._step_run_simulation = MagicMock()
        env._step_evaluate_reward = MagicMock()
        env._step_update_state = MagicMock()
        env._step_finalize = MagicMock()
        env.done = False
        env.observations = np.array([1.0, 2.0, 3.0])
        env.reward = 0.5
        env.truncated = False
        env.info = {}
        result = EnvBaseMixin._handle_other_environments(env, 'up1', 0.0, {}, True)
        env._step_apply_action.assert_called_once()
        env._step_run_simulation.assert_called_once()


# ============================================================================
# Tests: _step_apply_action through _step_finalize
# ============================================================================

class TestStepPipeline:
    def test_step_apply_action(self):
        env = make_env()
        EnvBaseMixin._step_apply_action(env, 'up1')
        env.setActionValue.assert_called_with('up1')

    def test_step_run_simulation(self):
        env = make_env(solver_backend='mf2005')
        env.runMODFLOW = MagicMock()
        env.runMODPATH = MagicMock()
        env.evaluateParticleTracking = MagicMock()
        EnvBaseMixin._step_run_simulation(env)
        env.runMODFLOW.assert_called_once()
        env.runMODPATH.assert_called_once()

    def test_step_run_simulation_mf6prt(self):
        env = make_env(solver_backend='mf6_prt', env_config={'solver_backend': 'mf6_prt'})
        env._step_run_simulation_mf6prt = MagicMock()
        EnvBaseMixin._step_run_simulation(env)
        env._step_run_simulation_mf6prt.assert_called_once()

    def test_step_evaluate_reward(self):
        env = make_env()
        env.calculate_reward = MagicMock()
        EnvBaseMixin._step_evaluate_reward(env)
        env.calculate_reward.assert_called_once()

    def test_step_update_state(self):
        env = make_env()
        env.update_state = MagicMock()
        env.get_normalized_observations = MagicMock()
        env.get_normalized_stresses = MagicMock()
        env.check_particle_state = MagicMock()
        EnvBaseMixin._step_update_state(env)
        env.update_state.assert_called_once()

    def test_step_finalize(self):
        env = make_env(RENDER=False, MANUALCONTROL=False, SAVEPLOT=False)
        env._feed_obs_hash = MagicMock()
        env.update_envhash = MagicMock()
        EnvBaseMixin._step_finalize(env)
        assert env.timeStep == 1


# ============================================================================
# Tests: _update_1, _update_2, _update_3
# ============================================================================

class TestUpdates:
    def test_update_1(self):
        env = make_env(actionValueNorth=58.5, actionValueSouth=57.5, state={})
        EnvBaseMixin._update_1(env)
        assert env.state['actionValueNorth'] == 58.5

    def test_update_2(self):
        env = make_env(actionValue=59.0, state={})
        EnvBaseMixin._update_2(env)
        assert env.state['actionValue'] == 59.0

    def test_update_3(self):
        env = make_env(actionValueX=50.0, actionValueY=40.0, state={})
        EnvBaseMixin._update_3(env)
        assert env.state['actionValueX'] == 50.0
        assert env.state['actionValueY'] == 40.0


# ============================================================================
# Tests: distance_to_well, check_boundaries
# ============================================================================

class TestDistanceAndBoundaries:
    def test_distance_to_well(self):
        env = make_env(extentY=100.0)
        d = EnvBaseMixin.distance_to_well(env, (50, 50), (50, 50))
        # dx=0, dy=100-50-50=0, so d=0
        assert abs(d - 0.0) < 0.01

    def test_distance_to_well_offset(self):
        env = make_env(extentY=100.0)
        d = EnvBaseMixin.distance_to_well(env, (50, 50), (30, 50))
        assert abs(d - 20.0) < 0.01

    def test_check_eastern_boundary_hit(self):
        env = make_env(dCol=1.0)
        assert EnvBaseMixin._check_eastern_boundary(env, 99.5, 100.0) is True
        assert env.done is True

    def test_check_eastern_boundary_miss(self):
        env = make_env(dCol=1.0)
        assert EnvBaseMixin._check_eastern_boundary(env, 50.0, 100.0) is False

    def test_check_western_boundary_hit(self):
        env = make_env(dCol=1.0)
        assert EnvBaseMixin._check_western_boundary(env, 0.5, 0.0) is True
        assert env.done is True

    def test_check_western_boundary_miss(self):
        env = make_env(dCol=1.0)
        assert EnvBaseMixin._check_western_boundary(env, 50.0, 0.0) is False

    def test_check_northern_boundary_hit(self):
        env = make_env(ENVTYPE='1s-d', dRow=1.0)
        assert EnvBaseMixin._check_northern_boundary(env, 99.5, 100.0) is True

    def test_check_northern_boundary_miss(self):
        env = make_env(ENVTYPE='1s-d', dRow=1.0)
        assert EnvBaseMixin._check_northern_boundary(env, 50.0, 100.0) is False

    def test_check_northern_boundary_invalid_env(self):
        env = make_env(ENVTYPE='0s-c', dRow=1.0)
        assert EnvBaseMixin._check_northern_boundary(env, 99.5, 100.0) is False

    def test_check_southern_boundary_hit(self):
        env = make_env(dRow=1.0)
        EnvBaseMixin._check_southern_boundary(env, 0.5, 0.0)
        assert env.done is True

    def test_check_boundaries_all(self):
        env = make_env(dCol=1.0, dRow=1.0, particleCoords=[50, 50, 0], minX=0, minY=0, extentX=100, extentY=100)
        EnvBaseMixin.check_boundaries(env)
        assert env.done is False


# ============================================================================
# Tests: check_max_steps, _prepare_time_step
# ============================================================================

class TestMaxStepsAndPrepare:
    def test_check_max_steps_hit(self):
        env = make_env(timeStep=200, maxSteps=200, done=False)
        EnvBaseMixin.check_max_steps(env)
        assert env.done is True

    def test_check_max_steps_not_hit(self):
        env = make_env(timeStep=100, maxSteps=200, done=False)
        EnvBaseMixin.check_max_steps(env)
        assert env.done is False

    def test_prepare_time_step(self):
        env = make_env(dt=1.0)
        env.mf6 = MagicMock()
        env.mf6.prepare_time_step = MagicMock()
        env.mf6.get_current_time = MagicMock(return_value=5.0)
        EnvBaseMixin._prepare_time_step(env)
        env.mf6.prepare_time_step.assert_called_with(1.0)
        assert env.current_time == 5.0


# ============================================================================
# Tests: collect_Qs, get_max_iterations
# ============================================================================

class TestCollectAndIter:
    def test_collect_Qs(self):
        env = make_env(nHelperWells=3)
        env.get_action_value = MagicMock(side_effect=lambda i, k: float(i + 1))
        Qs = EnvBaseMixin.collect_Qs(env)
        assert len(Qs) == 3

    def test_get_max_iterations(self):
        env = make_env()
        env.mf6 = MagicMock()
        env.mf6.get_var_address = MagicMock(return_value='addr')
        env.mf6.get_value = MagicMock(return_value=MagicMock(item=MagicMock(return_value=100)))
        EnvBaseMixin.get_max_iterations(env)
        assert env.max_iter == 100


# ============================================================================
# Tests: run_convergence_loop, handle_convergence_failure
# ============================================================================

class TestConvergence:
    def test_run_convergence_loop_success(self):
        env = make_env(max_iter=5)
        env.mf6 = MagicMock()
        env.mf6.prepare_solve = MagicMock()
        env.mf6.solve = MagicMock(return_value=True)
        assert EnvBaseMixin.run_convergence_loop(env) is True

    def test_run_convergence_loop_failure(self):
        env = make_env(max_iter=3)
        env.mf6 = MagicMock()
        env.mf6.prepare_solve = MagicMock()
        env.mf6.solve = MagicMock(return_value=False)
        assert EnvBaseMixin.run_convergence_loop(env) is False

    def test_handle_convergence_failure(self):
        env = make_env(success=True, model_ws='/tmp/ws')
        env.bmi_return = MagicMock(return_value=(True, []))
        result = EnvBaseMixin.handle_convergence_failure(env, 'testmodel')
        env.bmi_return.assert_called_with(True, '/tmp/ws', 'testmodel')


# ============================================================================
# Tests: update_envhash, handle_done
# ============================================================================

class TestEnvhashAndDone:
    def test_update_envhash(self):
        env = make_env()
        env._obs_hasher = hashlib.sha256(b'test')
        EnvBaseMixin.update_envhash(env)
        assert env.envhash == hashlib.sha256(b'test').hexdigest()

    def test_handle_done_not_done(self):
        env = make_env(done=False)
        EnvBaseMixin.handle_done(env, True)

    def test_handle_done_done(self):
        env = make_env(done=True)
        env.cleanup_temp_dir = MagicMock()
        env.finalize_simulation = MagicMock()
        env.post_simulation_teardown = MagicMock()
        EnvBaseMixin.handle_done(env, True)
        env.cleanup_temp_dir.assert_called_once()
        env.finalize_simulation.assert_called_once()
        env.post_simulation_teardown.assert_called_once_with(True)


# ============================================================================
# Tests: updateWellRate, updateWell
# ============================================================================

class TestUpdateWell:
    def test_updateWellRate_random(self):
        env = make_env(ENVTYPE='1r-d', wellQ=-1000, minQ=-2000, maxQ=-500, maxQChange=50.0)
        EnvBaseMixin.updateWellRate(env)
        assert -2000 <= env.wellQ <= -500

    def test_updateWellRate_steady(self):
        env = make_env(ENVTYPE='1s-d', wellQ=-1000)
        old_q = env.wellQ
        EnvBaseMixin.updateWellRate(env)
        assert env.wellQ == old_q

    def test_updateWell(self):
        env = make_env()
        EnvBaseMixin.updateWell(env)
        env._update_main_well_location.assert_called()
        env._update_helper_wells_location.assert_called()
        env._update_helper_wells_Q.assert_called()
        env._update_cell_info.assert_called()
        env._update_modflow_wel.assert_called()


# ============================================================================
# Tests: setup_rch_0sc
# ============================================================================

class TestSetupRch:
    def test_setup_rch_0sc(self):
        env = make_env(gwf=MagicMock())
        with patch('flopyarcade.env_base.ModflowGwfrcha') as mock_rch:
            EnvBaseMixin.setup_rch_0sc(env, 0.05)
            mock_rch.assert_called_once()


# ============================================================================
# Tests: _clamp
# ============================================================================

class TestClamp:
    def test_clamp_within(self):
        env = make_env()
        assert EnvBaseMixin._clamp(env, 5.0, 0.0, 10.0) == 5.0

    def test_clamp_below(self):
        env = make_env()
        assert EnvBaseMixin._clamp(env, -1.0, 0.0, 10.0) == 0.0

    def test_clamp_above(self):
        env = make_env()
        assert EnvBaseMixin._clamp(env, 15.0, 0.0, 10.0) == 10.0


# ============================================================================
# Tests: _update_random_head_fluctuations, _assign_boundary_heads
# ============================================================================

class TestHeadFluctuations:
    def test_update_random_head_fluctuations(self):
        env = make_env(maxHChange=0.2, minH=56.0, maxH=60.0)
        env.headSpecSouth = 58.0
        env.headSpecNorth = 58.0
        env._clamp = lambda val, vmin, vmax: max(vmin, min(val, vmax))
        EnvBaseMixin._update_random_head_fluctuations(env)
        assert 56.0 <= env.headSpecSouth <= 60.0
        assert 56.0 <= env.headSpecNorth <= 60.0

    def test_assign_boundary_heads_type1(self):
        env = make_env(ENVTYPE='1s-d', headSpecWest=60.0, headSpecEast=56.0,
                       actionValueSouth=57.0, actionValueNorth=59.0)
        env.strt = np.zeros((1, 10, 10), dtype=np.float32)
        EnvBaseMixin._assign_boundary_heads(env)
        assert env.strt[0, 5, 0] == 60.0
        assert env.strt[0, 5, -1] == 56.0
        assert (env.strt[0, 0, :] == 57.0).all()
        assert (env.strt[0, -1, :] == 59.0).all()

    def test_assign_boundary_heads_type2(self):
        env = make_env(ENVTYPE='2s-d', headSpecWest=60.0, headSpecEast=56.0, actionValue=58.0)
        env.strt = np.zeros((1, 10, 10), dtype=np.float32)
        EnvBaseMixin._assign_boundary_heads(env)
        assert (env.strt[0, -1, :] == 58.0).all()

    def test_assign_boundary_heads_type3(self):
        env = make_env(ENVTYPE='3s-d', headSpecWest=60.0, headSpecEast=56.0,
                       headSpecSouth=57.0, headSpecNorth=59.0)
        env.strt = np.zeros((1, 10, 10), dtype=np.float32)
        EnvBaseMixin._assign_boundary_heads(env)
        assert (env.strt[0, 0, :] == 57.0).all()
        assert (env.strt[0, -1, :] == 59.0).all()


# ============================================================================
# Tests: _sum_flows
# ============================================================================

class TestSumFlows:
    def test_sum_flows(self):
        env = make_env(nHelperWells=3)
        env.helperWells = {'wellQ1': -100.0, 'wellQ2': 50.0, 'wellQ3': -200.0}
        pump, recharge = EnvBaseMixin._sum_flows(env)
        assert pump == 300.0
        assert recharge == 50.0


# ============================================================================
# Tests: _get_init_solution
# ============================================================================

class TestGetInitSolution:
    def test_none(self):
        env = make_env(initWithSolution=True)
        assert EnvBaseMixin._get_init_solution(env, None) is True

    def test_override(self):
        env = make_env(initWithSolution=True)
        assert EnvBaseMixin._get_init_solution(env, False) is False


# ============================================================================
# Tests: add_lib_dependencies
# ============================================================================

class TestAddLibDependencies:
    def test_empty_list(self):
        env = make_env()
        EnvBaseMixin.add_lib_dependencies(env, [])

    def test_add_path(self):
        env = make_env()
        original_path = os.environ.get('PATH', '')
        try:
            EnvBaseMixin.add_lib_dependencies(env, ['/nonexistent/lib/path/test123'])
            assert '/nonexistent/lib/path/test123' in os.environ['PATH']
        finally:
            os.environ['PATH'] = original_path


# ============================================================================
# Tests: _update_mplst_line, _filterByTime
# ============================================================================

class TestMpsimAndFilter:
    def test_update_mplst_line(self):
        env = make_env()
        lines = ['header\n', 'mplst\n', 'old_value\n', 'footer\n']
        result = EnvBaseMixin._update_mplst_line(env, lines)
        assert '2   1   2' in result[2]

    def test_filterByTime_valid(self):
        env = make_env(periodLength=1.0)
        import pandas as pd
        data = pd.DataFrame({'time': [0.5, 1.0, 1.5]})
        result = EnvBaseMixin._filterByTime(env, data)
        assert result[0] == True
        assert result[1] == True
        assert result[2] == False

    def test_filterByTime_invalid(self):
        env = make_env(periodLength=1.0)
        import pandas as pd
        data = pd.DataFrame({'time': [2.0, 3.0]})
        with pytest.raises(ValueError):
            EnvBaseMixin._filterByTime(env, data)


# ============================================================================
# Tests: _appendTrajectories
# ============================================================================

class TestAppendTrajectories:
    def test_append(self):
        env = make_env(trajectories={'x': [], 'y': [], 'z': []})
        import pandas as pd
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
        valid = pd.Series([True, True, False])
        EnvBaseMixin._appendTrajectories(env, data, valid)
        assert len(env.trajectories['x']) == 1


# ============================================================================
# Tests: _circle_coords
# ============================================================================

class TestCircleCoords:
    def test_transform(self):
        env = make_env(extentY=100.0)
        result = EnvBaseMixin._circle_coords(env, (50, 30))
        assert result == (50, 70)


# ============================================================================
# Tests: _calculate_index
# ============================================================================

class TestCalculateIndex:
    def test_calculate_index(self):
        env = make_env()
        assert EnvBaseMixin._calculate_index(env, 50.0, 1.0, 100) == 50


# ============================================================================
# Tests: _detect_ipython
# ============================================================================

class TestDetectIPython:
    def test_detect(self):
        env = make_env()
        EnvBaseMixin._detect_ipython(env)
        assert hasattr(env, 'flagFromIPythonNotebook')


# ============================================================================
# Tests: _setup_axes, _create_outline_circle, _create_filled_circle
# ============================================================================

class TestPlotting:
    def test_setup_axes(self):
        env = make_env()
        env.fig = MagicMock()
        EnvBaseMixin._setup_axes(env)
        env.fig.add_subplot.assert_called_once()

    def test_create_outline_circle(self):
        env = make_env()
        circle = EnvBaseMixin._create_outline_circle(env, (5, 5), 10.0, 'red', 1, lw=2.0)
        assert circle is not None

    def test_create_filled_circle(self):
        env = make_env()
        circle = EnvBaseMixin._create_filled_circle(env, (5, 5), 10.0, 'blue', 0.5, 1)
        assert circle is not None


# ============================================================================
# Tests: _handle_image_array, _handle_display_or_return
# ============================================================================

class TestHandleImage:
    def test_handle_image_array(self):
        env = make_env(SAVEPLOT=False)
        env.fig = MagicMock()
        env.fig_to_rgb_array = MagicMock(return_value=np.zeros((10, 10, 3), dtype=np.uint8))
        env._ensure_plot_dirs = MagicMock()
        env._get_plot_filepath = MagicMock(return_value='/tmp/plot.png')
        env.plotArrays = []
        env._save_gif_if_needed = MagicMock()
        cb = MagicMock()
        result = EnvBaseMixin._handle_image_array(env, cb)
        assert result is not None
        cb.remove.assert_called_once()

    def test_handle_display_or_return_display(self):
        env = make_env()
        env._display_plot = MagicMock()
        cb = MagicMock()
        result = EnvBaseMixin._handle_display_or_return(env, cb, False, 120)
        assert result is None
        env._display_plot.assert_called_once()

    def test_handle_display_or_return_figure(self):
        env = make_env(SAVEPLOT=False)
        env.fig_to_rgb_array = MagicMock(return_value=np.zeros((10, 10, 3), dtype=np.uint8))
        env._ensure_plot_dirs = MagicMock()
        env._get_plot_filepath = MagicMock(return_value='/tmp/plot.png')
        env.plotArrays = []
        env._save_gif_if_needed = MagicMock()
        cb = MagicMock()
        result = EnvBaseMixin._handle_display_or_return(env, cb, True, 120)
        assert result is not None


# ============================================================================
# Tests: _handle_display_mode
# ============================================================================

class TestHandleDisplayMode:
    def test_manual_control(self):
        env = make_env(MANUALCONTROL=True, RENDER=False)
        env.renderUserInterAction = MagicMock()
        env.renderClearAxes = MagicMock()
        EnvBaseMixin._handle_display_mode(env)
        env.renderUserInterAction.assert_called_once()

    def test_render(self):
        env = make_env(MANUALCONTROL=False, RENDER=True, _blit_enabled=False)
        env.fig = MagicMock()
        env.renderFPS = MagicMock()
        env.renderClearAxes = MagicMock()
        EnvBaseMixin._handle_display_mode(env)
        env.renderFPS.assert_called_once()

    def test_render_blit(self):
        env = make_env(MANUALCONTROL=False, RENDER=True, _blit_enabled=True)
        env.render_blit_frame = MagicMock()
        env.renderFPS = MagicMock()
        env.renderClearAxes = MagicMock()
        EnvBaseMixin._handle_display_mode(env)
        env.render_blit_frame.assert_called_once()


# ============================================================================
# Tests: initialize_ibound_other
# ============================================================================

class TestInitializeIbound:
    def test_type1(self):
        env = make_env(ENVTYPE='1s-d', nLay=1, nRow=10, nCol=10)
        EnvBaseMixin.initialize_ibound_other(env)
        assert env.ibound[0, 5, 0] == -1
        assert env.ibound[0, 5, -1] == -1
        assert env.ibound[0, 0, 5] == -1
        assert env.ibound[0, -1, 5] == -1

    def test_type2(self):
        env = make_env(ENVTYPE='2s-d', nLay=1, nRow=10, nCol=10)
        EnvBaseMixin.initialize_ibound_other(env)
        assert env.ibound[0, 5, 0] == -1
        assert env.ibound[0, 5, -1] == -1
        assert env.ibound[0, -1, 5] == -1
        # row 0 should NOT be -1 for type2
        assert env.ibound[0, 0, 5] == 1


# ============================================================================
# Tests: initialize_start_heads_other
# ============================================================================

class TestInitializeStartHeads:
    def test_steady(self):
        env = make_env(periodSteadiness=True, nLay=1, nRow=10, nCol=10, ENVTYPE='1s-d')
        env._assign_boundary_heads = MagicMock()
        EnvBaseMixin.initialize_start_heads_other(env)
        assert env.strt.shape == (1, 10, 10)

    def test_transient(self):
        env = make_env(periodSteadiness=False, nLay=1, nRow=10, nCol=10, ENVTYPE='1s-d', timeStep=0)
        env.headsPrev = np.ones((1, 10, 10), dtype=np.float32) * 55.0
        env._assign_boundary_heads = MagicMock()
        EnvBaseMixin.initialize_start_heads_other(env)
        assert env.strt.shape == (1, 10, 10)


# ============================================================================
# Tests: _load_head_file
# ============================================================================

class TestLoadHeadFile:
    def test_load_head_file_missing(self):
        env = make_env(model_ws='/nonexistent', MODELNAME='Test')
        with patch('flopyarcade.env_base.bf.HeadFile', side_effect=Exception('not found')):
            result = EnvBaseMixin._load_head_file(env)
        assert result is None


# ============================================================================
# Tests: _change_to_workspace
# ============================================================================

class TestChangeToWorkspace:
    def test_change_to_workspace(self, tmp_path):
        env = make_env(wrkspc=str(tmp_path))
        old_cwd = os.getcwd()
        try:
            EnvBaseMixin._change_to_workspace(env)
            assert os.getcwd() == str(tmp_path)
        finally:
            os.chdir(old_cwd)


# ============================================================================
# Tests: reset
# ============================================================================

class TestReset:
    def test_reset_0sc(self):
        env = make_env(ENVTYPE='0s-c', _SEED=42)
        env._reset_0s_c = MagicMock()
        env._reset_other = MagicMock()
        EnvBaseMixin.reset(env, seed=42)
        env._reset_0s_c.assert_called_once()

    def test_reset_other(self):
        env = make_env(ENVTYPE='1s-d', _SEED=42)
        env._reset_0s_c = MagicMock()
        env._reset_other = MagicMock()
        EnvBaseMixin.reset(env, seed=42)
        env._reset_other.assert_called_once()


# ============================================================================
# Tests: _reset_0s_c, _reset_other
# ============================================================================

class TestResetMethods:
    def test_reset_0s_c(self):
        env = make_env(ENVTYPE='0s-c')
        env._setup_workspace = MagicMock()
        env._init_env_vars = MagicMock()
        env._init_env_specific = MagicMock()
        env.defineEnvironment = MagicMock()
        env.constructModel = MagicMock()
        EnvBaseMixin._reset_0s_c(env, 42)
        env._setup_workspace.assert_called_once()
        env.constructModel.assert_called_once()

    def test_reset_other(self):
        env = make_env(ENVTYPE='1s-d', SAVEPLOT=False, MANUALCONTROL=False, RENDER=False)
        env.PATHMF2005 = '/opt/mf'
        env.PATHMP6 = '/opt/mp'
        env.OBSPREP = True
        with patch.object(EnvBaseMixin, '__init__', create=True) as mock_init:
            EnvBaseMixin._reset_other(env, 42, None, True)
        # _reset_other calls self.__init__(env_config=...) then close()


# ============================================================================
# Tests: _update_title
# ============================================================================

class TestUpdateTitle:
    def test_update_title(self):
        env = make_env(timeStep=5, rewardCurrent=12.34)
        env.ax = MagicMock()
        EnvBaseMixin._update_title(env)
        env.ax.cla.assert_called_once()
        env.ax.set_title.assert_called_once()


# ============================================================================
# Tests: setup_wells_0sc
# ============================================================================

class TestSetupWells0sc:
    def test_setup_wells_0sc(self):
        env = make_env(nHelperWells=2, gwf=MagicMock())
        env.actionsDict = {
            'well0iCol': 1, 'well0iRow': 2, 'well0Q': -100.0,
            'well1iCol': 3, 'well1iRow': 4, 'well1Q': -200.0,
        }
        with patch('flopyarcade.env_base.ModflowGwfwel') as mock_wel:
            EnvBaseMixin.setup_wells_0sc(env)
            mock_wel.assert_called_once()


# ============================================================================
# Tests: create_discretization_other
# ============================================================================

class TestCreateDiscretization:
    def test_steady(self):
        env = make_env(nLay=1, nRow=10, nCol=10, dRow=1.0, dCol=1.0,
                       zTop=50.0, botM=np.array([50.0, 0.0]), periodSteadiness=True)
        env.mf = MagicMock()
        with patch('flopyarcade.env_base.ModflowDis') as mock_dis:
            mock_dis.return_value = MagicMock()
            EnvBaseMixin.create_discretization_other(env)
            mock_dis.assert_called_once()

    def test_transient(self):
        env = make_env(nLay=1, nRow=10, nCol=10, dRow=1.0, dCol=1.0,
                       zTop=50.0, botM=np.array([50.0, 0.0]), periodSteadiness=False,
                       periods=1, periodSteps=11, periodLength=1.0)
        env.mf = MagicMock()
        with patch('flopyarcade.env_base.ModflowDis') as mock_dis:
            mock_dis.return_value = MagicMock()
            EnvBaseMixin.create_discretization_other(env)
            mock_dis.assert_called_once()


# ============================================================================
# Tests: _init_env_0s_c, _init_env_other
# ============================================================================

class TestInitEnvMethods:
    def test_init_env_0s_c(self):
        env = make_env(ENVTYPE='0s-c', MODELNAME='TestModel')
        env.initializeSimulators = MagicMock()
        env.defineEnvironment = MagicMock()
        env.constructModel = MagicMock()
        env.add_lib_dependencies = MagicMock()
        env.getActionSpaceSize = MagicMock(return_value=30)
        env.actionsDict = {}
        with patch('flopyarcade.env_base.makedirs', create=True):
            with patch('flopyarcade.env_base.platformSystem', return_value='Linux'):
                EnvBaseMixin._init_env_0s_c(env)
        env.initializeSimulators.assert_called_once()
        env.constructModel.assert_called_once()

    def test_init_env_other(self):
        env = make_env(ENVTYPE='1s-d', minQ=-2000, maxQ=-500)
        env.initializeSimulators = MagicMock()
        env.defineEnvironment = MagicMock()
        env.initializeAction = MagicMock()
        env.initializeParticle = MagicMock()
        env.update_head_specs_if_needed = MagicMock()
        env.initializeModel = MagicMock()
        env.initializeWellRate = MagicMock(return_value=(50, 50, 50, [50, 50, 50], -1000))
        env.initialize_helper_wells = MagicMock()
        env.initializeWell = MagicMock()
        env._resolve_well_overlap = MagicMock()
        EnvBaseMixin._init_env_other(env)
        env.initializeSimulators.assert_called_once()
        env.initializeWell.assert_called_once()


# ============================================================================
# Tests: _modify_mpsim_file, _read_mpsim_lines, _write_mpsim_lines
# ============================================================================

class TestMpsimFile:
    def test_modify_mpsim_file(self, tmp_path):
        env = make_env(modelpth=str(tmp_path), MODELNAME='Test')
        mpsim = tmp_path / 'Test.mpsim'
        mpsim.write_text('line1\nmplst\nold_value\nline4\n')
        env._read_mpsim_lines = lambda: mpsim.read_text().splitlines(True)
        env._insert_custom_particle_settings = lambda lines: lines
        env._update_mplst_line = lambda lines: lines
        env._write_mpsim_lines = lambda lines: mpsim.write_text(''.join(lines))
        EnvBaseMixin._modify_mpsim_file(env)
        content = mpsim.read_text()
        assert 'line1' in content

    def test_read_mpsim_lines(self, tmp_path):
        env = make_env(modelpth=str(tmp_path), MODELNAME='Test')
        mpsim = tmp_path / 'Test.mpsim'
        mpsim.write_text('line1\nline2\n')
        lines = EnvBaseMixin._read_mpsim_lines(env)
        assert len(lines) == 2

    def test_write_mpsim_lines(self, tmp_path):
        env = make_env(modelpth=str(tmp_path), MODELNAME='Test')
        EnvBaseMixin._write_mpsim_lines(env, ['a\n', 'b\n'])
        mpsim = tmp_path / 'Test.mpsim'
        assert mpsim.read_text() == 'a\nb\n'


# ============================================================================
# Tests: _generate_circle_coords
# ============================================================================

class TestGenerateCircleCoords:
    def test_generate(self):
        env = make_env()
        x, y = EnvBaseMixin._generate_circle_coords(env, 10.0, 100)
        assert len(x) == 100
        assert len(y) == 100


# ============================================================================
# Tests: _get_game_result
# ============================================================================

class TestGetGameResult:
    def test_success(self):
        env = make_env(success=True)
        result = EnvBaseMixin._get_game_result(env)
        assert result == 'Success'

    def test_failure(self):
        env = make_env(success=False)
        result = EnvBaseMixin._get_game_result(env)
        assert result == 'Failure'


# ============================================================================
# Tests: format_time
# ============================================================================

class TestFormatTime:
    def test_format(self):
        env = make_env(current_time=123.456)
        result = EnvBaseMixin.format_time(env)
        assert isinstance(result, str)


# ============================================================================
# Tests: _count_trajectory_points
# ============================================================================

class TestCountTrajectoryPoints:
    def test_count(self):
        env = make_env(trajectories={'x': [np.array([1, 2])], 'y': [np.array([3, 4])], 'z': []})
        total, lengths = EnvBaseMixin._count_trajectory_points(env)
        assert total == 2
        assert lengths == [2]


# ============================================================================
# Tests: _generate_fade_colors
# ============================================================================

class TestGenerateFadeColors:
    def test_generate(self):
        env = make_env()
        colors = EnvBaseMixin._generate_fade_colors(env, 5)
        assert len(colors) == 5


# ============================================================================
# Tests: figToArray
# ============================================================================

class TestFigToArray:
    def test_fig_to_array(self):
        env = make_env()
        fig = MagicMock()
        fig.canvas.get_width_height.return_value = (10, 20)
        fig.canvas.tostring_argb.return_value = b'\x00' * 800
        with patch('flopyarcade.env_base.frombuffer', side_effect=lambda buf, dtype: np.zeros(800, dtype=np.uint8)):
            result = EnvBaseMixin.figToArray(env, fig)


# ============================================================================
# Tests: clear_axis
# ============================================================================

class TestClearAxis:
    def test_clear_axis(self):
        env = make_env()
        axis = MagicMock()
        EnvBaseMixin.clear_axis(env, axis)
        axis.clear.assert_called_once()


# ============================================================================
# Tests: textSpanAcrossAxis
# ============================================================================

class TestTextSpanAcrossAxis:
    def test_text_span(self):
        env = make_env()
        fig = MagicMock()
        ax = MagicMock()
        result = EnvBaseMixin.textSpanAcrossAxis(env, 'Hello', 100, 50, fig, ax)


# ============================================================================
# Tests: _fits_within_bounds
# ============================================================================

class TestFitsWithinBounds:
    def test_fits(self):
        env = make_env()
        bbox = MagicMock()
        bbox.width = 10.0
        bbox.height = 5.0
        assert EnvBaseMixin._fits_within_bounds(env, bbox, 100, 50, 1.0) is True

    def test_does_not_fit(self):
        env = make_env()
        bbox = MagicMock()
        bbox.width = 200.0
        bbox.height = 100.0
        assert EnvBaseMixin._fits_within_bounds(env, bbox, 100, 50, 1.0) is False


# ============================================================================
# Tests: _decrease_font_until_fit, _increase_font_until_fit
# ============================================================================

class TestFontFit:
    def test_decrease_font(self):
        env = make_env()
        fig = MagicMock()
        ax = MagicMock()
        text = MagicMock()
        text.get_fontsize.return_value = 12.0
        bbox = MagicMock()
        bbox.width = 50.0
        bbox.height = 10.0
        text.get_window_extent.return_value.transformed.return_value = bbox
        env._fits_within_bounds = lambda b, w, h, r: True
        result = EnvBaseMixin._decrease_font_until_fit(env, text, 100, 50, fig, ax, 1.0)

    def test_increase_font(self):
        env = make_env()
        fig = MagicMock()
        ax = MagicMock()
        text = MagicMock()
        text.get_fontsize.return_value = 8.0
        text.set_fontsize = MagicMock()
        bbox = MagicMock()
        bbox.width = 200.0
        bbox.height = 100.0
        text.get_window_extent.return_value.transformed.return_value = bbox
        result = EnvBaseMixin._increase_font_until_fit(env, text, 100, 50, fig, ax, 1.0)


# ============================================================================
# Tests: _add_curved_text
# ============================================================================

class TestAddCurvedText:
    def test_add_curved_text(self):
        env = make_env(extentY=100.0)
        ax = MagicMock()
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        with patch('flopyarcade.env_base.mtext') as mock_mtext:
            result = EnvBaseMixin._add_curved_text(env, x, y, 'test', 'center', ax, '\n', 12, 'red')


# ============================================================================
# Tests: _sum_flows edge case
# ============================================================================

class TestSumFlowsEdge:
    def test_no_helper_wells(self):
        env = make_env(nHelperWells=0)
        env.helperWells = {}
        pump, recharge = EnvBaseMixin._sum_flows(env)
        assert pump == 0.0
        assert recharge == 0.0


# ============================================================================
# Tests: _init_env_config solver_backend default
# ============================================================================

class TestSolverBackend:
    def test_default_mf2005(self):
        env = make_env()
        local_vars = {
            'ENVTYPE': '1s-d', 'PATHMF2005': None, 'PATHMP6': None,
            'MODELNAME': 'Test', 'ANIMATIONFOLDER': 'Anim',
            'flagSavePlot': False, 'flagManualControl': False,
            'manualControlTime': 0.1, 'flagRender': False,
            'NAGENTSTEPS': 100, 'nLay': 1, 'nRow': 50, 'nCol': 50,
            'OBSPREP': 'perceptron', 'initWithSolution': True,
            'PATHMF6DLL': None, '_seed': None, 'solver_backend': 'mf2005',
        }
        EnvBaseMixin._init_env_config(env, local_vars, None)
        assert env.solver_backend == 'mf2005'


# ============================================================================
# Tests: _update_envhash integration
# ============================================================================

class TestUpdateEnvhash:
    def test_hash_updates(self):
        env = make_env()
        env._obs_hasher = hashlib.sha256(b'initial')
        EnvBaseMixin.update_envhash(env)
        first_hash = env.envhash
        env._obs_hasher = hashlib.sha256(b'different')
        EnvBaseMixin.update_envhash(env)
        assert env.envhash != first_hash


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
