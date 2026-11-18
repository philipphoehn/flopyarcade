"""C8: Test that the environment works with different mesh resolutions.

Verifies that FloPyEnv can be constructed and stepped with various
nRow/nCol values (beyond the default 100×100 and the test 10×10),
and that observations are valid.
"""

import pytest
import numpy as np


ENV_TYPES_TO_TEST = [
    "1s-d",   # 1 helper well, discrete
    "2s-d",   # 2 helper wells, discrete
    "3s-d",   # 3 helper wells, discrete
    "1s-c",   # 1 helper well, continuous
]


GRID_SIZES = [
    (10, 10),   # small (already used in regression tests)
    (15, 15),   # non-standard square
    (20, 20),   # medium
    (25, 15),   # non-square (more rows than cols)
    (15, 25),   # non-square (more cols than rows)
]


def _make_env(nrow, ncol, envtype="1s-d", modelname_suffix=""):
    """Create a FloPyEnv with the given grid size."""
    from flopyarcade import FloPyEnv

    return FloPyEnv(
        ENVTYPE=envtype,
        MODELNAME=f"c8_{envtype}_{nrow}x{ncol}{modelname_suffix}",
        _seed=42,
        flagSavePlot=False,
        nRow=nrow,
        nCol=ncol,
    )


@pytest.mark.parametrize("nrow,ncol", GRID_SIZES)
def test_env_creates_and_resets(nrow, ncol):
    """Environment can be created and reset with various grid sizes."""
    env = _make_env(nrow, ncol)
    obs, info = env.reset()
    assert obs is not None
    env.close()


@pytest.mark.parametrize("nrow,ncol", [(10, 10), (20, 20), (25, 15)])
def test_env_step_produces_valid_obs(nrow, ncol):
    """A single step returns a valid observation."""
    env = _make_env(nrow, ncol)
    obs, _ = env.reset()
    result = env.step(0)
    # Gym-style: (obs, reward, terminated, truncated, info)
    obs2, reward, terminated, truncated, info = result
    assert obs2 is not None
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


@pytest.mark.parametrize("nrow,ncol", [(10, 10), (20, 20)])
def test_observation_is_finite(nrow, ncol):
    """Observation values are finite and non-empty."""
    env = _make_env(nrow, ncol)
    obs, _ = env.reset()
    assert obs.size > 0
    assert np.all(np.isfinite(obs)), "Observation contains non-finite values"
    env.close()


@pytest.mark.parametrize("envtype", ENV_TYPES_TO_TEST)
def test_multiple_env_types_same_grid(envtype):
    """Multiple env types work with the same non-default grid."""
    env = _make_env(15, 15, envtype=envtype)
    obs, _ = env.reset()
    assert obs is not None
    env.close()


def test_grid_size_stored_correctly():
    """The environment stores the requested grid dimensions."""
    env = _make_env(30, 20)
    assert env.nRow == 30, f"Expected nRow=30, got {env.nRow}"
    assert env.nCol == 20, f"Expected nCol=20, got {env.nCol}"
    env.close()


def test_small_grid_extreme():
    """Very small grid (5×5) still works without crash."""
    env = _make_env(5, 5)
    obs, _ = env.reset()
    assert obs is not None
    env.close()


def test_non_square_grid_step():
    """Non-square grid (25 rows, 15 cols) can complete a step."""
    env = _make_env(25, 15, modelname_suffix="_step")
    obs, _ = env.reset()
    result = env.step(0)
    obs2, reward, terminated, truncated, info = result
    assert np.isfinite(reward)
    env.close()
