"""Tests for flopyarcade.ci_gate (ROADMAP B24)."""

import json
import os

import pytest

from flopyarcade.ci_gate import (
    BaselineEntry,
    check_gate,
    format_gate_report,
    get_latest_baseline,
    load_baseline,
    run_gate,
    save_baseline,
    update_history,
)


# ─── BaselineEntry ────────────────────────────────────────────────────────────

class TestBaselineEntry:
    def test_creation(self):
        e = BaselineEntry(
            envtype="flopy_mf6",
            score=42.5,
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert e.envtype == "flopy_mf6"
        assert e.score == 42.5
        assert e.timestamp == "2025-01-01T00:00:00+00:00"
        assert e.source == "ci"  # default

    def test_custom_source(self):
        e = BaselineEntry(
            envtype="test",
            score=1.0,
            timestamp="2025-01-01",
            source="manual",
        )
        assert e.source == "manual"


# ─── save / load baseline ─────────────────────────────────────────────────────

class TestBaselineIO:
    def test_roundtrip(self, tmp_path):
        scores = {"env_a": 10.0, "env_b": 20.5}
        p = str(tmp_path / "baseline.json")
        save_baseline(scores, p)
        loaded = load_baseline(p)
        assert loaded == scores

    def test_load_missing_file_returns_empty(self, tmp_path):
        p = str(tmp_path / "nonexistent.json")
        assert load_baseline(p) == {}


# ─── check_gate ───────────────────────────────────────────────────────────────

class TestCheckGate:
    def test_all_pass_identical_scores(self):
        scores = {"env_a": 100.0, "env_b": 200.0}
        result = check_gate(scores, scores)
        assert result["pass"] is True
        assert result["violations"] == []
        assert result["improvements"] == []

    def test_one_violation(self):
        baseline = {"env_a": 100.0, "env_b": 200.0}
        current = {"env_a": 90.0, "env_b": 200.0}  # env_a dropped 10 %
        result = check_gate(current, baseline, threshold_pct=5.0)
        assert result["pass"] is False
        assert len(result["violations"]) == 1
        v = result["violations"][0]
        assert v["envtype"] == "env_a"
        assert v["drop_pct"] == pytest.approx(10.0)

    def test_improvement(self):
        baseline = {"env_a": 100.0}
        current = {"env_a": 110.0}
        result = check_gate(current, baseline)
        assert result["pass"] is True
        assert len(result["improvements"]) == 1
        imp = result["improvements"][0]
        assert imp["envtype"] == "env_a"
        assert imp["gain_pct"] == pytest.approx(10.0)

    def test_new_envtype(self):
        baseline = {"env_a": 100.0}
        current = {"env_a": 100.0, "env_new": 50.0}
        result = check_gate(current, baseline)
        assert "env_new" in result["new_envtypes"]
        assert result["pass"] is True  # new envtypes are not violations

    def test_missing_envtype(self):
        baseline = {"env_a": 100.0, "env_gone": 50.0}
        current = {"env_a": 100.0}
        result = check_gate(current, baseline)
        assert "env_gone" in result["missing_envtypes"]

    def test_threshold_boundary_exactly_at_threshold_passes(self):
        # 5 % drop on 100 → 95; exactly at threshold should PASS
        baseline = {"env_a": 100.0}
        current = {"env_a": 95.0}
        result = check_gate(current, baseline, threshold_pct=5.0)
        assert result["pass"] is True
        assert result["violations"] == []

    def test_threshold_boundary_just_over_threshold_fails(self):
        # 5.01 % drop on 100 → 94.99; just over threshold should FAIL
        baseline = {"env_a": 100.0}
        current = {"env_a": 94.99}
        result = check_gate(current, baseline, threshold_pct=5.0)
        assert result["pass"] is False
        assert len(result["violations"]) == 1


# ─── format_gate_report ───────────────────────────────────────────────────────

class TestFormatGateReport:
    def test_contains_pass(self):
        result = {
            "pass": True,
            "violations": [],
            "improvements": [],
            "new_envtypes": [],
            "missing_envtypes": [],
        }
        report = format_gate_report(result)
        assert "PASS" in report
        assert "FAIL" not in report

    def test_contains_fail(self):
        result = {
            "pass": False,
            "violations": [
                {"envtype": "env_a", "baseline": 100.0, "current": 80.0, "drop_pct": 20.0}
            ],
            "improvements": [],
            "new_envtypes": [],
            "missing_envtypes": [],
        }
        report = format_gate_report(result)
        assert "FAIL" in report
        assert "env_a" in report


# ─── JSONL history ────────────────────────────────────────────────────────────

class TestHistory:
    def test_update_history_creates_file(self, tmp_path):
        p = str(tmp_path / "history.jsonl")
        update_history(p, {"env_a": 10.0})
        assert os.path.isfile(p)
        with open(p) as fh:
            lines = fh.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "timestamp" in data
        assert data["scores"]["env_a"] == 10.0

    def test_update_history_appends(self, tmp_path):
        p = str(tmp_path / "history.jsonl")
        update_history(p, {"env_a": 10.0})
        update_history(p, {"env_a": 20.0, "env_b": 30.0})
        with open(p) as fh:
            lines = fh.readlines()
        assert len(lines) == 2
        second = json.loads(lines[1])
        assert second["scores"]["env_b"] == 30.0

    def test_get_latest_baseline_reads_last_line(self, tmp_path):
        p = str(tmp_path / "history.jsonl")
        update_history(p, {"env_a": 10.0})
        update_history(p, {"env_a": 99.0, "env_b": 5.0})
        latest = get_latest_baseline(p)
        assert latest["env_a"] == 99.0
        assert latest["env_b"] == 5.0

    def test_get_latest_baseline_missing_file(self, tmp_path):
        p = str(tmp_path / "nope.jsonl")
        assert get_latest_baseline(p) == {}


# ─── run_gate end-to-end ─────────────────────────────────────────────────────

class TestRunGate:
    def test_end_to_end_update_modify_gate_fails(self, tmp_path):
        p = str(tmp_path / "history.jsonl")
        # Establish baseline
        update_history(p, {"env_a": 100.0, "env_b": 200.0})
        # Simulate a regression on env_a (20 % drop)
        current = {"env_a": 80.0, "env_b": 200.0}
        result = run_gate(current, p, threshold_pct=5.0)
        assert result["pass"] is False
        assert any(v["envtype"] == "env_a" for v in result["violations"])
        # env_b unchanged → no violation, no improvement
        assert not any(v["envtype"] == "env_b" for v in result["violations"])
