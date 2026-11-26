"""Tests for flopyarcade.perf_gate – Q3 performance regression gate."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from flopyarcade.perf_gate import (
    PerfBaseline,
    PerfGate,
    PerfMeasurement,
    auto_update_baseline,
    ci_gate_check,
    format_gate_report,
    regression_history,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def baseline_a() -> PerfBaseline:
    return PerfBaseline(
        envtype="aquifer_2d",
        baseline_ms_per_step=12.0,
        recorded_at="2025-01-01T00:00:00",
        git_sha="abc123",
    )


@pytest.fixture
def baseline_b() -> PerfBaseline:
    return PerfBaseline(
        envtype="aquifer_3d",
        baseline_ms_per_step=20.0,
        recorded_at="2025-01-01T00:00:00",
        git_sha=None,
    )


@pytest.fixture
def gate(baseline_a, baseline_b) -> PerfGate:
    return PerfGate(
        baselines={"aquifer_2d": baseline_a, "aquifer_3d": baseline_b},
        max_ratio=2.0,
        fail_threshold_pct=50.0,
    )


@pytest.fixture
def meas_pass() -> PerfMeasurement:
    return PerfMeasurement(
        envtype="aquifer_2d",
        ms_per_step=10.0,  # faster than baseline
        n_episodes=50,
        timestamp="2025-06-01T12:00:00",
    )


@pytest.fixture
def meas_equal() -> PerfMeasurement:
    return PerfMeasurement(
        envtype="aquifer_2d",
        ms_per_step=12.0,  # exactly baseline
        n_episodes=50,
        timestamp="2025-06-01T12:00:00",
    )


@pytest.fixture
def meas_warn() -> PerfMeasurement:
    return PerfMeasurement(
        envtype="aquifer_2d",
        ms_per_step=19.2,  # ratio = 1.6
        n_episodes=50,
        timestamp="2025-06-01T12:00:00",
    )


@pytest.fixture
def meas_fail() -> PerfMeasurement:
    return PerfMeasurement(
        envtype="aquifer_2d",
        ms_per_step=30.0,  # ratio = 2.5
        n_episodes=50,
        timestamp="2025-06-01T12:00:00",
    )


# ─── Tests ─────────────────────────────────────────────────────────────────────

class TestDataclasses:
    def test_perf_baseline_creation(self, baseline_a):
        assert baseline_a.envtype == "aquifer_2d"
        assert baseline_a.baseline_ms_per_step == 12.0
        assert baseline_a.recorded_at == "2025-01-01T00:00:00"
        assert baseline_a.git_sha == "abc123"

    def test_perf_baseline_optional_sha(self, baseline_b):
        assert baseline_b.git_sha is None

    def test_perf_measurement_creation(self, meas_pass):
        assert meas_pass.envtype == "aquifer_2d"
        assert meas_pass.ms_per_step == 10.0
        assert meas_pass.n_episodes == 50
        assert meas_pass.timestamp == "2025-06-01T12:00:00"


class TestPerfGateInit:
    def test_init_with_baselines(self, gate, baseline_a, baseline_b):
        bl = gate.baselines()
        assert "aquifer_2d" in bl
        assert "aquifer_3d" in bl
        assert bl["aquifer_2d"].baseline_ms_per_step == 12.0
        assert bl["aquifer_3d"].baseline_ms_per_step == 20.0
        assert gate.max_ratio == 2.0
        assert gate.fail_threshold_pct == 50.0


class TestCheck:
    def test_check_passed_faster(self, gate, meas_pass):
        r = gate.check(meas_pass)
        assert r["action"] == "pass"
        assert r["passed"] is True
        assert r["ratio"] < 1.0
        assert r["slowdown_pct"] < 0

    def test_check_passed_equal(self, gate, meas_equal):
        r = gate.check(meas_equal)
        assert r["action"] == "pass"
        assert r["passed"] is True
        assert r["ratio"] == 1.0
        assert r["slowdown_pct"] == 0.0

    def test_check_warn_ratio_1_6(self, gate, meas_warn):
        r = gate.check(meas_warn)
        assert r["action"] == "warn"
        assert r["passed"] is True
        assert r["ratio"] == pytest.approx(1.6)
        assert r["slowdown_pct"] == pytest.approx(60.0)

    def test_check_fail_ratio_2_5(self, gate, meas_fail):
        r = gate.check(meas_fail)
        assert r["action"] == "fail"
        assert r["passed"] is False
        assert r["ratio"] == pytest.approx(2.5)
        assert r["slowdown_pct"] == pytest.approx(150.0)

    def test_check_unknown_envtype(self, gate):
        m = PerfMeasurement(
            envtype="unknown_env",
            ms_per_step=5.0,
            n_episodes=10,
            timestamp="2025-06-01T12:00:00",
        )
        r = gate.check(m)
        assert r["action"] == "pass"
        assert r["passed"] is True
        assert r["ratio"] is None


class TestCheckAll:
    def test_check_all_multiple(self, gate, meas_pass, meas_warn, meas_fail):
        results = gate.check_all([meas_pass, meas_warn, meas_fail])
        assert len(results) == 3
        assert results[0]["action"] == "pass"
        assert results[1]["action"] == "warn"
        assert results[2]["action"] == "fail"


class TestGatePassed:
    def test_gate_passed_all_pass(self, gate, meas_pass, meas_equal):
        results = gate.check_all([meas_pass, meas_equal])
        assert gate.gate_passed(results) is True

    def test_gate_passed_one_fail(self, gate, meas_pass, meas_fail):
        results = gate.check_all([meas_pass, meas_fail])
        assert gate.gate_passed(results) is False

    def test_gate_passed_empty(self, gate):
        assert gate.gate_passed([]) is True


class TestWorstRegression:
    def test_worst_regression_correct(self, gate, meas_pass, meas_warn, meas_fail):
        results = gate.check_all([meas_pass, meas_warn, meas_fail])
        worst = gate.worst_regression(results)
        assert worst["envtype"] == "aquifer_2d"
        assert worst["slowdown_pct"] == pytest.approx(150.0)

    def test_worst_regression_empty(self, gate):
        assert gate.worst_regression([]) == {}


class TestSetBaseline:
    def test_set_baseline_updates(self, gate):
        gate.set_baseline("aquifer_2d", 8.0, git_sha="newsha")
        bl = gate.baselines()["aquifer_2d"]
        assert bl.baseline_ms_per_step == 8.0
        assert bl.git_sha == "newsha"

    def test_set_baseline_new_envtype(self, gate):
        gate.set_baseline("new_env", 5.0)
        assert "new_env" in gate.baselines()


class TestPersistence:
    def test_save_load_roundtrip(self, gate, tmp_path):
        path = str(tmp_path / "baselines.json")
        gate.save_baselines(path)
        loaded = PerfGate.load_baselines(path)

        assert set(loaded.baselines().keys()) == set(gate.baselines().keys())
        assert loaded.baselines()["aquifer_2d"].baseline_ms_per_step == 12.0
        assert loaded.baselines()["aquifer_3d"].baseline_ms_per_step == 20.0
        assert loaded.max_ratio == 2.0
        assert loaded.fail_threshold_pct == 50.0


class TestCiGateCheck:
    def test_ci_gate_check_full_pipeline(self, gate, meas_pass, meas_warn):
        result = ci_gate_check(gate, [meas_pass, meas_warn])
        assert result["passed"] is True
        assert len(result["results"]) == 2
        assert "worst" in result
        assert "summary" in result
        assert "PASS" in result["summary"] or "pass" in result["summary"]

    def test_ci_gate_check_failure_case(self, gate, meas_fail):
        result = ci_gate_check(gate, [meas_fail])
        assert result["passed"] is False
        assert result["worst"]["action"] == "fail"
        assert "FAILED" in result["summary"]


class TestFormatReport:
    def test_format_gate_report_contains_status(self, gate, meas_pass, meas_warn, meas_fail):
        results = gate.check_all([meas_pass, meas_warn, meas_fail])
        report = format_gate_report(results)
        assert "PASS" in report
        assert "WARN" in report
        assert "FAIL" in report
        assert "aquifer_2d" in report

    def test_format_gate_report_empty(self):
        report = format_gate_report([])
        assert "no measurements" in report


class TestAutoUpdateBaseline:
    def test_only_if_better_updates_faster(self, gate):
        # baseline is 12.0, new measurement is 8.0 (faster)
        m = PerfMeasurement("aquifer_2d", 8.0, 50, "2025-06-01T12:00:00")
        updated = auto_update_baseline(gate, [m], only_if_better=True)
        assert "aquifer_2d" in updated
        assert gate.baselines()["aquifer_2d"].baseline_ms_per_step == 8.0

    def test_only_if_better_skips_slower(self, gate):
        # baseline is 12.0, new measurement is 15.0 (slower)
        m = PerfMeasurement("aquifer_2d", 15.0, 50, "2025-06-01T12:00:00")
        updated = auto_update_baseline(gate, [m], only_if_better=True)
        assert "aquifer_2d" not in updated
        assert gate.baselines()["aquifer_2d"].baseline_ms_per_step == 12.0

    def test_replaces_slower_when_flag_false(self, gate):
        # baseline is 12.0, new measurement is 15.0 (slower) but only_if_better=False
        m = PerfMeasurement("aquifer_2d", 15.0, 50, "2025-06-01T12:00:00")
        updated = auto_update_baseline(gate, [m], only_if_better=False)
        assert "aquifer_2d" in updated
        assert gate.baselines()["aquifer_2d"].baseline_ms_per_step == 15.0

    def test_new_envtype_added(self, gate):
        m = PerfMeasurement("new_env", 5.0, 10, "2025-06-01T12:00:00")
        updated = auto_update_baseline(gate, [m], only_if_better=True)
        assert "new_env" in updated
        assert gate.baselines()["new_env"].baseline_ms_per_step == 5.0


class TestRegressionHistory:
    def test_regression_history_structure(self, tmp_path):
        path = str(tmp_path / "history.jsonl")
        entries = [
            {"timestamp": "2025-01-01T00:00:00", "passed": True, "n": 3},
            {"timestamp": "2025-02-01T00:00:00", "passed": False, "n": 4},
            {"timestamp": "2025-03-01T00:00:00", "passed": True, "n": 2},
        ]
        with open(path, "w") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")

        result = regression_history(path)
        assert len(result) == 3
        assert result[0]["passed"] is True
        assert result[1]["passed"] is False
        assert result[2]["n"] == 2

    def test_regression_history_missing_file(self):
        result = regression_history("/nonexistent/path/history.jsonl")
        assert result == []


class TestMultiEnvtype:
    def test_three_envtypes_mixed_results(self):
        baselines = {
            "env_a": PerfBaseline("env_a", 10.0, "2025-01-01T00:00:00"),
            "env_b": PerfBaseline("env_b", 20.0, "2025-01-01T00:00:00"),
            "env_c": PerfBaseline("env_c", 30.0, "2025-01-01T00:00:00"),
        }
        g = PerfGate(baselines=baselines)

        measurements = [
            PerfMeasurement("env_a", 8.0, 50, "2025-06-01T12:00:00"),   # pass (faster)
            PerfMeasurement("env_b", 34.0, 50, "2025-06-01T12:00:00"),  # warn (ratio 1.7)
            PerfMeasurement("env_c", 75.0, 50, "2025-06-01T12:00:00"),  # fail (ratio 2.5)
        ]

        results = g.check_all(measurements)
        assert results[0]["action"] == "pass"
        assert results[1]["action"] == "warn"
        assert results[2]["action"] == "fail"

        assert g.gate_passed(results) is False

        worst = g.worst_regression(results)
        assert worst["envtype"] == "env_c"
        assert worst["slowdown_pct"] == pytest.approx(150.0)

        report = format_gate_report(results)
        assert "env_a" in report
        assert "env_b" in report
        assert "env_c" in report
        assert "FAIL" in report
