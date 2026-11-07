"""
CI-safe golden value regression tests.

Validates exact-match against golden outputs for 4 representative environment types:
- 1s-d: discrete, single well
- 1s-c: continuous, single well
- 3s-d: discrete, multi-well
- 6r-c: continuous, multi-layer, multi-well

These catch structural regressions (dimension changes, normalization breaks,
reward formula changes) on every push without running all 20 env types.

The full suite (all 20 env types) lives in test_regression.py (marked slow).
"""

import json
import pathlib
import numpy as np
import pytest

GOLDEN_PATH = pathlib.Path(__file__).parent / "golden_outputs.json"

CI_ENV_TYPES = ["1s-d", "1s-c", "3s-d", "6r-c"]


def _load_golden():
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k in CI_ENV_TYPES}


GOLDEN = _load_golden()


def _make_action(envtype, golden_entry):
    at = golden_entry["actionType"]
    if at == "discrete":
        return 2
    else:
        shape = golden_entry["action_space_shape"]
        if isinstance(shape, list) and len(shape) > 0:
            return np.full(tuple(shape), 0.5, dtype=np.float32)
        return np.array([0.5], dtype=np.float32)


def _run_env(envtype, n_steps=5):
    from flopyarcade import FloPyEnv
    entry = GOLDEN[envtype]
    env = FloPyEnv(
        ENVTYPE=envtype,
        MODELNAME=f"ci_reg_{envtype}",
        flagSavePlot=False,
        flagManualControl=False,
        flagRender=False,
        _seed=entry["seed"],
        nLay=entry["nLay"],
        nRow=entry["nRow"],
        nCol=entry["nCol"],
    )
    obs, _ = env.reset()
    action = _make_action(envtype, entry)
    results = []
    for _ in range(n_steps):
        obs, reward, terminated, truncated, _ = env.step(action)
        results.append({
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "obs": np.array(obs, dtype=np.float64).flatten().tolist(),
        })
        if terminated or truncated:
            break
    env.close()
    return results


class TestGoldenRegressionCI:
    """Exact-match golden value regression for CI (4 env types)."""

    @pytest.mark.parametrize("envtype", CI_ENV_TYPES)
    def test_rewards_exact(self, envtype):
        entry = GOLDEN[envtype]
        steps = entry["steps"]
        results = _run_env(envtype, len(steps))
        for i, (got, expected) in enumerate(zip(results, steps)):
            assert abs(got["reward"] - expected["reward"]) < 1e-6, (
                f"{envtype}: step {i} reward {got[chr(39)+chr(114)+chr(101)+chr(119)+chr(97)+chr(114)+chr(100)+chr(39)]} != expected {expected[chr(39)+chr(114)+chr(101)+chr(119)+chr(97)+chr(114)+chr(100)+chr(39)]}"
            )

    @pytest.mark.parametrize("envtype", CI_ENV_TYPES)
    def test_observations_match(self, envtype):
        entry = GOLDEN[envtype]
        steps = entry["steps"]
        results = _run_env(envtype, len(steps))
        for i, (got, expected) in enumerate(zip(results, steps)):
            got_obs = np.array(got["obs"])
            exp_obs = np.array(expected["obs"])
            assert got_obs.shape == exp_obs.shape, (
                f"{envtype}: step {i} obs shape mismatch"
            )
            assert np.allclose(got_obs, exp_obs, atol=1e-5), (
                f"{envtype}: step {i} obs mismatch, max diff={np.max(np.abs(got_obs - exp_obs))}"
            )

    @pytest.mark.parametrize("envtype", CI_ENV_TYPES)
    def test_termination_matches(self, envtype):
        entry = GOLDEN[envtype]
        steps = entry["steps"]
        results = _run_env(envtype, len(steps))
        for i, (got, expected) in enumerate(zip(results, steps)):
            assert got["terminated"] == expected["terminated"], (
                f"{envtype}: step {i} terminated mismatch"
            )
            assert got["truncated"] == expected["truncated"], (
                f"{envtype}: step {i} truncated mismatch"
            )

    @pytest.mark.parametrize("envtype", CI_ENV_TYPES)
    def test_obs_finite(self, envtype):
        results = _run_env(envtype, 3)
        for i, r in enumerate(results):
            arr = np.array(r["obs"])
            assert np.all(np.isfinite(arr)), (
                f"{envtype}: step {i} has non-finite observations"
            )

    @pytest.mark.parametrize("envtype", CI_ENV_TYPES)
    def test_determinism(self, envtype):
        """Same seed + same actions = same rewards."""
        entry = GOLDEN[envtype]
        n = len(entry["steps"])
        r1 = _run_env(envtype, n)
        r2 = _run_env(envtype, n)
        for i in range(min(len(r1), len(r2))):
            assert r1[i]["reward"] == r2[i]["reward"], (
                f"{envtype}: step {i} not deterministic"
            )

    @pytest.mark.parametrize("envtype", CI_ENV_TYPES)
    def test_env_spaces_valid(self, envtype):
        from flopyarcade import FloPyEnv
        entry = GOLDEN[envtype]
        env = FloPyEnv(
            ENVTYPE=envtype,
            MODELNAME=f"ci_sp_{envtype}",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=entry["seed"],
            nLay=entry["nLay"],
            nRow=entry["nRow"],
            nCol=entry["nCol"],
        )
        obs, _ = env.reset()
        assert env.action_space is not None
        assert env.observation_space is not None
        env.close()
