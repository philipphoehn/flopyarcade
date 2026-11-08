"""Conftest: mock heavy deps to prevent thread hangs + isolate sys.modules pollution."""
import sys
import os
import time
from unittest.mock import MagicMock
import pytest

# Prevent module-level parser.parse_args() in example modules from seeing pytest CLI args
sys.argv = ["pytest"]

# Ensure real gymnasium is in sys.modules before any test mocks it
import gymnasium



# Break infinite retry loops (sleep in while loop) with a call counter
_original_sleep = time.sleep
_sleep_call_count = [0]
def _mock_sleep(*a, **k):
    _sleep_call_count[0] += 1
    if _sleep_call_count[0] > 200:
        raise RuntimeError("time.sleep called >200 times - breaking infinite retry loop")
time.sleep = _mock_sleep

# --- Permanent mocks: prevent real thread-spawning imports ---

# ipykernel is not installed
sys.modules['ipykernel'] = MagicMock()

# tqdm spawns a monitor thread that prevents clean shutdown
_tqdm_mock = MagicMock()
_tqdm_mock.__call__ = lambda *a, **k: (iter(a[0]) if a and hasattr(a[0], '__iter__') else iter([0]))
_tqdm_mock.auto = MagicMock()
_tqdm_mock.notebook = MagicMock()
sys.modules['tqdm'] = _tqdm_mock
sys.modules['tqdm.auto'] = _tqdm_mock.auto
sys.modules['tqdm.notebook'] = _tqdm_mock.notebook

# pathos can spawn real Pool threads
_pm = MagicMock()
_pm.multiprocessing = MagicMock()
_pm.multiprocessing.cpu_count = MagicMock(return_value=4)
_pm.pools = MagicMock()
_pm.helpers = MagicMock()
sys.modules['pathos'] = _pm
sys.modules['pathos.multiprocessing'] = _pm.multiprocessing
sys.modules['pathos.pools'] = _pm.pools
sys.modules['pathos.helpers'] = _pm.helpers

# ray: mock entirely to prevent TF thread spawning and cluster connection
_ray = MagicMock()
_ray.__version__ = '2.0.0'
_ray.tune = MagicMock()
_ray.tune.registry = MagicMock()
_ray.tune.registry.register_env = MagicMock()
_ray.init = MagicMock()
_ray.shutdown = MagicMock()
_ray.train = MagicMock()
_ray.rllib = MagicMock()
_ray.rllib.agents = MagicMock()
_ray.rllib.agents.dqn = MagicMock()
_ray.rllib.agents.dqn.DQNSeparateWorkerTrainer = MagicMock
_ray.rllib.agents.dqn.ApexTrainer = MagicMock
_ray.rllib.agents.ppo = MagicMock()
_ray.rllib.agents.ppo.PPOTrainer = MagicMock
_ray.rllib.agents.appo = MagicMock()
_ray.rllib.agents.appo.APPOTrainer = MagicMock
_ray.rllib.agents.impala = MagicMock()
_ray.rllib.agents.impala.IMPATrainer = MagicMock
_ray.rllib.algorithms = MagicMock()
_ray.rllib.algorithms.dqn = MagicMock()
_ray.rllib.algorithms.dqn.DQNSeparateWorkerTrainer = MagicMock
_ray.rllib.algorithms.dqn.ApexTrainer = MagicMock
_ray.rllib.algorithms.ppo = MagicMock()
_ray.rllib.algorithms.ppo.PPOTrainer = MagicMock
_ray.rllib.algorithms.appo = MagicMock()
_ray.rllib.algorithms.appo.APPOTrainer = MagicMock
_ray.rllib.algorithms.impala = MagicMock()
_ray.rllib.algorithms.impala.IMPATrainer = MagicMock
_ray.rllib.utils = MagicMock()
_ray.rllib.utils.check_env = MagicMock()
sys.modules['ray'] = _ray
sys.modules['ray.tune'] = _ray.tune
sys.modules['ray.tune.registry'] = _ray.tune.registry
sys.modules['ray.train'] = _ray.train
sys.modules['ray.rllib'] = _ray.rllib
sys.modules['ray.rllib.agents'] = _ray.rllib.agents
sys.modules['ray.rllib.agents.dqn'] = _ray.rllib.agents.dqn
sys.modules['ray.rllib.agents.ppo'] = _ray.rllib.agents.ppo
sys.modules['ray.rllib.agents.appo'] = _ray.rllib.agents.appo
sys.modules['ray.rllib.agents.impala'] = _ray.rllib.agents.impala
sys.modules['ray.rllib.algorithms'] = _ray.rllib.algorithms
sys.modules['ray.rllib.algorithms.dqn'] = _ray.rllib.algorithms.dqn
sys.modules['ray.rllib.algorithms.ppo'] = _ray.rllib.algorithms.ppo
sys.modules['ray.rllib.algorithms.appo'] = _ray.rllib.algorithms.appo
sys.modules['ray.rllib.algorithms.impala'] = _ray.rllib.algorithms.impala
sys.modules['ray.rllib.utils'] = _ray.rllib.utils

# --- sys.modules pollution cleanup ---

_EXTERNAL_KEYS = [
    'flopy', 'flopy.modflow', 'flopy.mf6', 'flopy.mf6.modflow',
    'flopy.mf6.modflow.mfgwfwel', 'flopy.mf6.utils',
    'flopy.mf6.utils.binarygrid_util', 'flopy.modpath', 'flopy.plot',
    'flopy.utils', 'flopy.utils.binaryfile',
    'gymnasium', 'gymnasium.spaces',
    'imageio', 'joblib', 'xmipy',
    'tensorflow', 'tensorflow.keras', 'tensorflow.keras.initializers',
    'tensorflow.keras.layers', 'tensorflow.keras.models',
    'tensorflow.keras.optimizers', 'tensorflow.keras.random',
    'tensorflow.keras.backend', 'tensorflow.compat', 'tensorflow.compat.v1',
    'IPython', 'IPython.display',
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.ticker',
    'matplotlib.colors', 'matplotlib.transforms', 'matplotlib.text', 'matplotlib.cm',
]


def _is_fake(mod):
    """True if mod is a test-injected fake (MagicMock or spec-less ModuleType)."""
    if mod is None:
        return False
    if isinstance(mod, MagicMock):
        return True
    spec = getattr(mod, '__spec__', 'sentinel')
    return spec is None


def _clean_pollution():
    """Remove fake sys.modules entries introduced by tests."""
    cleaned = False
    for key in _EXTERNAL_KEYS:
        if key in sys.modules and _is_fake(sys.modules[key]):
            del sys.modules[key]
            cleaned = True
    for key in list(sys.modules.keys()):
        if key.startswith('flopy.') and _is_fake(sys.modules[key]):
            del sys.modules[key]
            cleaned = True
    # ALWAYS clean flopyarcade entries (unconditional)
    for key in list(sys.modules.keys()):
        if key == 'flopyarcade' or key.startswith('flopyarcade.'):
            if _is_fake(sys.modules[key]):
                del sys.modules[key]
                cleaned = True
    # Reset sleep counter so next module gets a fresh budget
    _sleep_call_count[0] = 0


@pytest.fixture(scope="module", autouse=True)
def _teardown_sys_modules():
    """Clean sys.modules pollution before AND after each test module."""
    _clean_pollution()
    yield
    _clean_pollution()
