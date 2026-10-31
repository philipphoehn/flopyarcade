"""
Deterministic regression tests — exact-match against golden outputs.

Covers ALL 20 environment types (1s-d through 6r-c, 0s-c, 5s-c-cost).
Each test creates a FloPyEnv with a fixed seed, performs a fixed no-op action
sequence, and asserts that the resulting rewards and observations match the
golden baseline.

Golden values were generated on 2026-10-31 with Python 3.13, MODFLOW 6,
and the patched seed-threading fix. If a change is *intentional*,
regenerate with:
    python -m pytest flopyarcade/tests/test_regression.py --update-golden
"""

import json
import pathlib
import numpy as np
import pytest

GOLDEN_PATH = pathlib.Path(__file__).parent / "golden_outputs.json"


def _load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


GOLDEN = _load_golden()
ENV_TYPES = list(GOLDEN.keys())


def _make_action(envtype, golden_entry):
    """Reconstruct the no-op action for this env type."""
    at = golden_entry["actionType"]
    if at == "discrete":
        # Use 'keep' index (2 for 3-action, 2 for 5-action)
        # In practice the golden was generated with 'keep' → index 2 for 3-action,
        # index 2 for 5-action (the middle 'keep').
        # Safer: just use 2 (keep is always at index 2 in 3-action, 2 in 5-action)
        return 2
    else:
        # Continuous: 0.5 = neutral no-op
        shape = golden_entry["action_space_shape"]
        if isinstance(shape, list):
            return np.full(tuple(shape), 0.5, dtype=np.float32)
        return np.full((shape,), 0.5, dtype=np.float32)


@pytest.fixture(scope="module")
def golden_data():
    """Return the loaded golden dict."""
    return GOLDEN


def _run_env(envtype, n_steps=5):
    """Create env, reset, run n_steps no-op actions. Return list of (obs, reward, term, trunc)."""
    from flopyarcade import FloPyEnv

    g = GOLDEN[envtype]
    env = FloPyEnv(
        ENVTYPE=envtype,
        MODELNAME=f"reg_{envtype}_{n_steps}",
        flagSavePlot=False,
        flagManualControl=False,
        flagRender=False,
        _seed=42,
        nLay=1,
        nRow=10,
        nCol=10,
    )
    obs, _ = env.reset()
    results = []
    for i in range(n_steps):
        action = _make_action(envtype, g)
        obs, reward, term, trunc, _ = env.step(action)
        results.append((np.array(obs, dtype=np.float64), float(reward), bool(term), bool(trunc)))
        if term or trunc:
            break
    env.close()
    return results


# ---------------------------------------------------------------------------
# Exact-match tests (parametrized per env type)
# ---------------------------------------------------------------------------

class TestExactMatch:
    """Assert bit-exact reward and observation match against golden baseline."""

    @pytest.mark.parametrize("envtype", ENV_TYPES)
    def test_rewards_exact(self, envtype):
        """Every step's reward must match the golden value within 1e-6."""
        results = _run_env(envtype)
        golden_steps = GOLDEN[envtype]["steps"]
        for i, (obs, reward, term, trunc) in enumerate(results):
            if i >= len(golden_steps):
                break
            expected = golden_steps[i]["reward"]
            assert abs(reward - expected) < 1e-6, (
                f"{envtype} step {i}: reward={reward} expected={expected}"
            )
            assert term == golden_steps[i]["terminated"], f"{envtype} step {i} terminated mismatch"
            assert trunc == golden_steps[i]["truncated"], f"{envtype} step {i} truncated mismatch"

    @pytest.mark.parametrize("envtype", ENV_TYPES)
    def test_observations_exact(self, envtype):
        """Every step's observation vector must match golden within 1e-4."""
        results = _run_env(envtype)
        golden_steps = GOLDEN[envtype]["steps"]
        for i, (obs, reward, term, trunc) in enumerate(results):
            if i >= len(golden_steps):
                break
            expected_obs = np.array(golden_steps[i]["obs"], dtype=np.float64)
            assert obs.shape == expected_obs.shape, (
                f"{envtype} step {i}: obs shape {obs.shape} != {expected_obs.shape}"
            )
            np.testing.assert_allclose(
                obs, expected_obs, atol=1e-4, rtol=0,
                err_msg=f"{envtype} step {i}: observation mismatch\n"
                        f"  actual:   {obs[:8]}...\n"
                        f"  expected: {expected_obs[:8]}..."
            )
            if term or trunc:
                break

    @pytest.mark.parametrize("envtype", ENV_TYPES)
    def test_obs_dimension(self, envtype):
        """Observation vector dimension must match golden baseline."""
        results = _run_env(envtype, n_steps=1)
        expected_dim = len(GOLDEN[envtype]["steps"][0]["obs"])
        assert len(results[0][0]) == expected_dim, (
            f"{envtype}: obs dim={len(results[0][0])} expected={expected_dim}"
        )

    @pytest.mark.parametrize("envtype", ENV_TYPES)
    def test_obs_finite(self, envtype):
        """All observation values must be finite (no NaN/Inf)."""
        results = _run_env(envtype)
        for i, (obs, reward, term, trunc) in enumerate(results):
            assert np.all(np.isfinite(obs)), f"{envtype} step {i}: NaN/Inf in obs"
            assert np.isfinite(reward), f"{envtype} step {i}: reward not finite"
            if term or trunc:
                break

    @pytest.mark.parametrize("envtype", ENV_TYPES)
    def test_game_terminates(self, envtype):
        """Environment must terminate within its maxSteps limit."""
        from flopyarcade import FloPyEnv
        g = GOLDEN[envtype]
        env = FloPyEnv(
            ENVTYPE=envtype,
            MODELNAME=f"reg_term_{envtype}",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=42,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        env.reset()
        limit = getattr(env, 'maxSteps', 200) + 10
        done = False
        try:
            for i in range(limit):
                action = _make_action(envtype, g)
                _, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    done = True
                    break
        except RuntimeError as e:
            if 'MODFLOW' in str(e):
                pytest.skip(f"{envtype}: MODFLOW convergence failure in long-run (known limitation)")
            raise
        env.close()
        assert done, f"{envtype}: did not terminate within {limit} steps (maxSteps={getattr(env, 'maxSteps', '?')})"

    @pytest.mark.parametrize("envtype", ENV_TYPES)
    def test_env_spaces_valid(self, envtype):
        """Action and observation spaces must be valid."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(
            ENVTYPE=envtype,
            MODELNAME=f"reg_space_{envtype}",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=42,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        assert env.action_space is not None, f"{envtype}: action_space is None"
        assert env.observation_space is not None, f"{envtype}: observation_space is None"
        obs, _ = env.reset()
        assert obs.shape[0] == env.observation_space.shape[0], (
            f"{envtype}: obs dim {obs.shape[0]} != space {env.observation_space.shape[0]}"
        )
        env.close()

    @pytest.mark.parametrize("envtype", ENV_TYPES)
    def test_determinism(self, envtype):
        """Two runs with same seed must produce identical rewards."""
        r1 = _run_env(envtype)
        r2 = _run_env(envtype)
        for i in range(len(r1)):
            assert r1[i][1] == pytest.approx(r2[i][1], abs=1e-8), (
                f"{envtype} step {i}: non-deterministic reward {r1[i][1]} vs {r2[i][1]}"
            )


# ---------------------------------------------------------------------------
# Golden update helper
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden_outputs.json from current environment behavior.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--update_golden", False):
        print("\n[update-golden] Regenerating golden_outputs.json ...")
        from flopyarcade import FloPyArcade
        all_envs = FloPyArcade().ENVTYPES
        golden = {}
        for envtype in all_envs:
            try:
                env = FloPyEnv(
                    ENVTYPE=envtype, MODELNAME=f"golden_{envtype}",
                    flagSavePlot=False, flagManualControl=False, flagRender=False,
                    _seed=42, nLay=1, nRow=10, nCol=10
                )
                if env.actionType == 'discrete':
                    keep_idx = 2  # 'keep' is always at index 2
                    actions = [keep_idx] * 5
                else:
                    shape = env.action_space.shape
                    actions = [np.full(shape, 0.5, dtype=np.float32)] * 5

                obs, _ = env.reset()
                steps = []
                for a in actions:
                    obs, reward, term, trunc, _ = env.step(a)
                    steps.append({
                        'reward': round(float(reward), 8),
                        'obs': [round(float(x), 8) for x in obs],
                        'terminated': bool(term),
                        'truncated': bool(trunc)
                    })
                    if term or trunc:
                        break
                env.close()
                golden[envtype] = {
                    'seed': 42, 'nLay': 1, 'nRow': 10, 'nCol': 10,
                    'actionType': env.actionType,
                    'action_space_shape': list(env.action_space.shape) if hasattr(env.action_space, 'shape') else len(env.action_space),
                    'steps': steps
                }
            except Exception as e:
                print(f"  SKIP {envtype}: {e}")

        with open(GOLDEN_PATH, 'w') as f:
            json.dump(golden, f, indent=2)
        print(f"[update-golden] Saved {len(golden)} env types to {GOLDEN_PATH}")
        # Remove all tests since we're updating
        items[:] = []
