"""Conftest: ray compat, TF eager, sys.modules isolation, sys.argv, import repair."""
import sys
import os
import pytest
from unittest.mock import MagicMock

# ray >= 2.43 removed ray.rllib.agents from public API
try:
    import ray.rllib
    if not hasattr(ray.rllib, 'agents'):
        _agents_mod = MagicMock()
        _agents_mod.dqn = MagicMock()
        _agents_mod.dqn.ApexTrainer = type("ApexTrainer", (), {})
        ray.rllib.agents = _agents_mod
        sys.modules['ray.rllib.agents'] = _agents_mod
        sys.modules['ray.rllib.agents.dqn'] = _agents_mod.dqn
except ImportError:
    pass

# ray.rllib disables TF eager execution; re-enable it for Keras/TF tests
try:
    import tensorflow as tf
    if not tf.executing_eagerly():
        tf.compat.v1.enable_eager_execution()
except ImportError:
    pass

# Force non-interactive matplotlib backend for headless/CI environments
import matplotlib
matplotlib.use('Agg')

# Guard matplotlib's units.Registry.get_converter against infinite recursion.
# When a type's __iter__ returns instances of the same type (e.g. MagicMock
# or certain numpy wrappers), get_converter recurses forever. We wrap it
# with a depth counter that returns None after 16 levels.
import matplotlib.units as _mu
import numpy as _np
from matplotlib import cbook as _cbook

_orig_gc = _mu.Registry.get_converter

def _safe_get_converter(self, x, _depth=0):
    """get_converter with recursion depth guard."""
    if _depth > 16:
        return None
    x = _cbook._unpack_to_numpy(x)
    if isinstance(x, _np.ndarray):
        x = _np.ma.getdata(x).ravel()
        if not x.size:
            return self.get_converter(_np.array([0], dtype=x.dtype))
    for cls in type(x).__mro__:
        try:
            return self[cls]
        except KeyError:
            pass
    try:
        first = _cbook._safe_first_finite(x)
    except (TypeError, StopIteration):
        pass
    else:
        if type(first) is not type(x):
            return _safe_get_converter(self, first, _depth + 1)
    return None

_mu.Registry.get_converter = _safe_get_converter

# Set sys.argv before play_againstai module-level parse_args()
sys.argv = ["play_againstai.py", "--envtype", "1s-d", "--cpus", "4", "--gpus", "0",
            "--playbenchmark", "False", "--external", "True", "--suffix", "_test"]


# ──────────────────────────────────────────────────────────────────────────────
# Robust import repair: some test files (test_cov_play_rl, test_mock_rllib,
# test_cov_final_push, test_cov_big) delete or replace sys.modules entries
# for flopy submodules and don't always restore cleanly. This hook validates
# that the critical import chain still works before each test, and purges+
# re-imports if it's broken.
# ──────────────────────────────────────────────────────────────────────────────

import importlib
import importlib.util
import collections.abc as _cabc

# Save real ABC references IMMEDIATELY at conftest load time.
# Some test files corrupt collections.abc by replacing classes with
# MagicMock. We save the real ones so we can restore them later.
_REAL_ABC = {
    'Iterator': _cabc.Iterator,
    'Iterable': _cabc.Iterable,
    'Mapping': _cabc.Mapping,
    'Sized': _cabc.Sized,
    'Container': _cabc.Container,
}


def _restore_abc():
    """Restore collections.abc entries if they've been corrupted."""
    import collections.abc as cur
    for name, real in _REAL_ABC.items():
        try:
            if getattr(cur, name) is not real:
                setattr(cur, name, real)
        except (AttributeError, TypeError):
            pass


def _flopy_is_broken():
    """Check if the flopy import chain is functional."""
    # Check 1: top-level flopy is not a mock
    flopy_mod = sys.modules.get('flopy')
    if flopy_mod is not None and isinstance(flopy_mod, MagicMock):
        return True
    # Check 2: flopy.mf6.modflow should be importable
    try:
        spec = importlib.util.find_spec('flopy.mf6.modflow')
        if spec is None:
            return True
    except (ImportError, AttributeError, TypeError):
        return True
    # Check 3: if flopy.mf6 is in sys.modules, it should have __path__
    mf6 = sys.modules.get('flopy.mf6')
    if mf6 is not None:
        if isinstance(mf6, MagicMock):
            return True
        if not hasattr(mf6, '__path__') and not hasattr(mf6, '__file__'):
            return True
    # Check 4: flopy must have __version__ (pakbase.py does `from . import __version__`)
    if flopy_mod is not None:
        try:
            _ = flopy_mod.__version__
        except (AttributeError, TypeError):
            return True
    return False


def _purge_and_reimport():
    """Remove all corrupted flopy/flopyarcade modules and re-import."""
    to_purge = []
    for name in list(sys.modules.keys()):
        if name == 'flopy' or name.startswith('flopy.') or \
           name == 'flopyarcade' or name.startswith('flopyarcade.'):
            mod = sys.modules[name]
            if isinstance(mod, MagicMock):
                to_purge.append(name)
            elif name.startswith('flopy.'):
                # Purge file-loaded modules that don't have proper spec
                spec = getattr(mod, '__spec__', None)
                if spec is None and not hasattr(mod, '__path__') and not hasattr(mod, '__file__'):
                    to_purge.append(name)
                # Also purge if __file__ points to a non-existent path
                elif hasattr(mod, '__file__') and mod.__file__ and not os.path.exists(mod.__file__):
                    to_purge.append(name)

    for name in to_purge:
        del sys.modules[name]

    # Purge 'flopy' top-level if broken or missing __version__
    if 'flopy' in sys.modules:
        mod = sys.modules['flopy']
        if isinstance(mod, MagicMock) or not hasattr(mod, '__path__') or not hasattr(mod, '__version__'):
            del sys.modules['flopy']

    # Force clean re-import
    try:
        import flopy  # noqa: F401
    except ImportError:
        pass
    try:
        import flopy.mf6  # noqa: F401
    except ImportError:
        pass
    try:
        import flopy.mf6.modflow  # noqa: F401
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _ensure_imports():
    """Validate and repair module state before each test."""
    # Save CWD (play_againstai does os.chdir at module level)
    saved_cwd = os.getcwd()

    # Repair if broken
    if _flopy_is_broken():
        _purge_and_reimport()

    # Restore collections.abc if corrupted by a previous test
    _restore_abc()

    # Validate matplotlib can create a figure. If it can't (ABC corruption,
    # registry corruption, etc.), do a nuclear reimport of matplotlib.
    _mat_ok = False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _fig = plt.figure()
        plt.close(_fig)
        _mat_ok = True
    except Exception:
        pass

    if not _mat_ok:
        # Nuclear reset: purge all matplotlib modules and reimport
        _mat_modules = [n for n in sys.modules if n.startswith('matplotlib')]
        for n in _mat_modules:
            del sys.modules[n]
        # Also purge the ABC caches that matplotlib's imports populated
        import abc as _abc_mod
        if hasattr(_abc_mod, '_abc_subclasscheck'):
            pass  # can't easily reset internal state
        import matplotlib
        matplotlib.use('Agg')

    yield

    # After test: close all figures
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass

    # Restore CWD after test
    try:
        os.chdir(saved_cwd)
    except OSError:
        pass
    except NameError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Note: test_cov_envbase.py and test_cov_render.py replace sys.modules with
# mock packages at module level. They MUST run in their own pytest invocation.
# When collected alongside other tests they corrupt imports for all subsequent
# tests. Use --ignore flags or run them in a separate process.
# ──────────────────────────────────────────────────────────────────────────────
