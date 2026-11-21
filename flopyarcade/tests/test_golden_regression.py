"""Tests for flopyarcade.golden_regression."""

import json
import os
import tempfile

import pytest

from flopyarcade.golden_regression import (
    GoldenRecord,
    GoldenReference,
    RegressionResult,
    all_passed,
    ci_check,
    compare_to_golden,
    failure_summary,
    format_regression_report,
    generate_golden_from_run,
)


def _make_record(envtype="env_a", seed=42, episode=0, reward=100.0, steps=10):
    return GoldenRecord(
        envtype=envtype,
        seed=seed,
        episode=episode,
        final_reward=reward,
        n_steps=steps,
    )


def _make_reference():
    recs = [
        _make_record("env_a", 42, 0, 100.0, 10),
        _make_record("env_a", 42, 1, 200.0, 20),
        _make_record("env_a", 42, 2, 300.0, 30),
    ]
    return GoldenReference(recs)


# ---------------------------------------------------------------------------
# GoldenRecord
# ---------------------------------------------------------------------------


def test_goldenrecord_creation():
    r = GoldenRecord(
        envtype="env_x",
        seed=7,
        episode=3,
        final_reward=42.5,
        n_steps=100,
        obs_final=[1.0, 2.0, 3.0],
        action_sequence=[0, 1, 2],
    )
    assert r.envtype == "env_x"
    assert r.seed == 7
    assert r.episode == 3
    assert r.final_reward == 42.5
    assert r.n_steps == 100
    assert r.obs_final == [1.0, 2.0, 3.0]
    assert r.action_sequence == [0, 1, 2]


# ---------------------------------------------------------------------------
# GoldenReference
# ---------------------------------------------------------------------------


def test_goldenreference_init():
    ref = _make_reference()
    assert ref.n_records() == 3


def test_goldenreference_get_correct():
    ref = _make_reference()
    rec = ref.get("env_a", 42, 1)
    assert rec.final_reward == 200.0
    assert rec.n_steps == 20


def test_goldenreference_get_missing_raises():
    ref = _make_reference()
    with pytest.raises(KeyError):
        ref.get("env_zzz", 99, 0)


def test_goldenreference_n_records():
    ref = _make_reference()
    assert ref.n_records() == 3
    ref2 = GoldenReference([])
    assert ref2.n_records() == 0


def test_save_json_creates_file():
    ref = _make_reference()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        ref.save_json(path)
        assert os.path.isfile(path)
        with open(path) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["envtype"] == "env_a"
    finally:
        os.unlink(path)


def test_load_json_roundtrip():
    ref = _make_reference()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        ref.save_json(path)
        loaded = GoldenReference.load_json(path)
        assert loaded.n_records() == 3
        for i in range(3):
            orig = ref.all()[i]
            new = loaded.all()[i]
            assert orig.envtype == new.envtype
            assert orig.seed == new.seed
            assert orig.episode == new.episode
            assert orig.final_reward == new.final_reward
            assert orig.n_steps == new.n_steps
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# compare_to_golden
# ---------------------------------------------------------------------------


def test_compare_exact_match():
    ref = _make_reference()
    actual = [_make_record("env_a", 42, 0, 100.0, 10)]
    results = compare_to_golden(ref, actual, tolerance_pct=0.1)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].diff_pct == 0.0


def test_compare_within_tolerance():
    ref = _make_reference()
    # 100.05 is 0.05% above 100.0 — within 0.1%
    actual = [_make_record("env_a", 42, 0, 100.05, 10)]
    results = compare_to_golden(ref, actual, tolerance_pct=0.1)
    assert results[0].passed is True
    assert results[0].diff_pct < 0.1


def test_compare_exceeds_tolerance():
    ref = _make_reference()
    # 105.0 is 5% above 100.0 — exceeds 0.1%
    actual = [_make_record("env_a", 42, 0, 105.0, 10)]
    results = compare_to_golden(ref, actual, tolerance_pct=0.1)
    assert results[0].passed is False
    assert results[0].diff_pct > 0.1


def test_compare_no_golden_match():
    ref = _make_reference()
    actual = [_make_record("env_unknown", 99, 0, 50.0, 5)]
    results = compare_to_golden(ref, actual, tolerance_pct=0.1)
    assert results[0].passed is False
    assert results[0].expected_reward == 0.0


def test_tolerance_boundary_exact():
    """diff_pct exactly at tolerance should pass (<=)."""
    ref = _make_reference()
    # expected = 100.0; we need diff_pct == 0.1 exactly
    # diff_pct = |x - 100| / (100 + 1e-8) * 100
    # For diff_pct = 0.1: |x-100| = 0.1 * (100+1e-8)/100 ≈ 0.1
    # Use tolerance 1.0 and value 101.0 → diff ≈ 1.0%
    ref2 = GoldenReference([_make_record("env_b", 1, 0, 100.0, 10)])
    actual = [_make_record("env_b", 1, 0, 101.0, 10)]
    results = compare_to_golden(ref2, actual, tolerance_pct=1.0)
    # diff_pct = 1.0/100*100 = 1.0 (approx) → should pass at tolerance 1.0
    assert results[0].passed is True


def test_multiple_envtypes_and_episodes():
    recs = [
        _make_record("env_a", 1, 0, 100.0, 10),
        _make_record("env_a", 1, 1, 200.0, 20),
        _make_record("env_b", 2, 0, 50.0, 5),
        _make_record("env_b", 2, 1, 75.0, 15),
    ]
    ref = GoldenReference(recs)
    actual = [
        _make_record("env_a", 1, 0, 100.1, 10),
        _make_record("env_a", 1, 1, 199.9, 20),
        _make_record("env_b", 2, 0, 50.0, 5),
        _make_record("env_b", 2, 1, 76.0, 15),  # 1.33% diff → fail
    ]
    results = compare_to_golden(ref, actual, tolerance_pct=0.1)
    assert len(results) == 4
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[2].passed is True
    assert results[3].passed is False


# ---------------------------------------------------------------------------
# all_passed
# ---------------------------------------------------------------------------


def test_all_passed_true():
    results = [
        RegressionResult("e", 1, 0, 100.0, 100.0, 0.0, True, 10, 10),
        RegressionResult("e", 1, 1, 200.0, 200.0, 0.0, True, 20, 20),
    ]
    assert all_passed(results) is True


def test_all_passed_false():
    results = [
        RegressionResult("e", 1, 0, 100.0, 100.0, 0.0, True, 10, 10),
        RegressionResult("e", 1, 1, 200.0, 210.0, 5.0, False, 20, 20),
    ]
    assert all_passed(results) is False


# ---------------------------------------------------------------------------
# failure_summary
# ---------------------------------------------------------------------------


def test_failure_summary_structure():
    results = [
        RegressionResult("env_a", 1, 0, 100.0, 100.0, 0.0, True, 10, 10),
        RegressionResult("env_a", 1, 1, 200.0, 210.0, 5.0, False, 20, 20),
    ]
    s = failure_summary(results)
    assert s["n_total"] == 2
    assert s["n_passed"] == 1
    assert s["n_failed"] == 1
    assert len(s["failures"]) == 1
    assert s["failures"][0]["envtype"] == "env_a"
    assert s["failures"][0]["episode"] == 1
    assert s["failures"][0]["diff_pct"] == 5.0


# ---------------------------------------------------------------------------
# generate_golden_from_run
# ---------------------------------------------------------------------------


def test_generate_golden_from_run_count():
    episodes = [
        {"final_reward": 10.0, "n_steps": 5},
        {"final_reward": 20.0, "n_steps": 10},
        {"final_reward": 30.0, "n_steps": 15},
        {"final_reward": 40.0, "n_steps": 20},
    ]
    records = generate_golden_from_run("env_c", 99, episodes, n_episodes=3)
    assert len(records) == 3
    assert records[0].episode == 0
    assert records[0].final_reward == 10.0
    assert records[2].final_reward == 30.0
    assert records[2].n_steps == 15


# ---------------------------------------------------------------------------
# ci_check
# ---------------------------------------------------------------------------


def test_ci_check_full_pipeline():
    ref = _make_reference()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        ref.save_json(path)
        actual = [_make_record("env_a", 42, 0, 100.0, 10)]
        result = ci_check(path, actual, tolerance_pct=0.1)
        assert result["passed"] is True
        assert len(result["results"]) == 1
        assert result["summary"]["n_passed"] == 1
        assert result["summary"]["n_failed"] == 0
    finally:
        os.unlink(path)


def test_ci_check_failure_case():
    ref = _make_reference()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        ref.save_json(path)
        actual = [_make_record("env_a", 42, 0, 150.0, 10)]
        result = ci_check(path, actual, tolerance_pct=0.1)
        assert result["passed"] is False
        assert result["summary"]["n_failed"] == 1
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# format_regression_report
# ---------------------------------------------------------------------------


def test_format_regression_report():
    results = [
        RegressionResult("env_a", 42, 0, 100.0, 100.0, 0.0, True, 10, 10),
        RegressionResult("env_a", 42, 1, 200.0, 210.0, 5.0, False, 20, 20),
    ]
    report = format_regression_report(results)
    assert "PASS" in report
    assert "FAIL" in report
    assert "env_a" in report
    assert "42" in report
