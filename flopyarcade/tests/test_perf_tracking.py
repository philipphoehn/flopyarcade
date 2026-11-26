"""Tests for B23 wall-clock performance tracking module."""

import json
import os
import tempfile

import pytest

from flopyarcade.perf_tracking import (
    PerfSample,
    PerfTracker,
    compare_perf,
    format_perf_report,
)


def make_sample(envtype="1s-d", ms=10.0, n_ep=50, sha=None):
    """Helper to create a PerfSample directly."""
    return PerfSample(
        timestamp="2025-01-01T00:00:00",
        envtype=envtype,
        ms_per_step=ms,
        steps_per_sec=1000.0 / ms,
        n_episodes=n_ep,
        git_sha=sha,
    )


class TestPerfSample:
    def test_creation(self):
        s = PerfSample(
            timestamp="2025-06-01T12:00:00",
            envtype="2s-c",
            ms_per_step=15.5,
            steps_per_sec=64.516,
            n_episodes=100,
            git_sha="abc123",
        )
        assert s.timestamp == "2025-06-01T12:00:00"
        assert s.envtype == "2s-c"
        assert s.ms_per_step == 15.5
        assert s.steps_per_sec == pytest.approx(64.516)
        assert s.n_episodes == 100
        assert s.git_sha == "abc123"

    def test_optional_git_sha(self):
        s = PerfSample(
            timestamp="2025-06-01",
            envtype="1s-d",
            ms_per_step=5.0,
            steps_per_sec=200.0,
            n_episodes=10,
        )
        assert s.git_sha is None


class TestRecord:
    def test_record_appends(self):
        t = PerfTracker()
        s1 = t.record("1s-d", 10.0, 50)
        s2 = t.record("1s-d", 12.0, 50, git_sha="def456")
        assert len(t._samples) == 2
        assert t._samples[0] is s1
        assert t._samples[1] is s2
        assert s2.git_sha == "def456"

    def test_steps_per_sec_calculation(self):
        t = PerfTracker()
        s = t.record("2s-c", 25.0, 20)
        assert s.steps_per_sec == pytest.approx(40.0)


class TestGetBaseline:
    def test_returns_first_sample(self):
        t = PerfTracker()
        s1 = make_sample(envtype="1s-d", ms=10.0)
        s2 = make_sample(envtype="1s-d", ms=15.0)
        t._samples = [s2, s1]  # reversed order on purpose
        baseline = t.get_baseline("1s-d")
        # get_baseline iterates in list order, so first in list is s2
        # Actually it returns the first one it finds in self._samples order
        # Let's just verify it returns one of them with ms=15.0 (first in list)
        assert baseline.ms_per_step == 15.0

    def test_get_baseline_raises_for_unknown(self):
        t = PerfTracker()
        t.record("1s-d", 10.0, 50)
        with pytest.raises(KeyError):
            t.get_baseline("9s-unknown")


class TestCheckDrift:
    def test_no_drift(self):
        t = PerfTracker()
        t.record("1s-d", 10.0, 50)
        t.record("1s-d", 10.0, 50)
        d = t.check_drift("1s-d")
        assert d["drift_pct"] == pytest.approx(0.0)
        assert d["flag"] is False

    def test_drift_15_percent_flagged(self):
        t = PerfTracker()
        t.record("1s-d", 10.0, 50)
        t.record("1s-d", 11.5, 50)  # 15% increase
        d = t.check_drift("1s-d")
        assert d["drift_pct"] == pytest.approx(15.0)
        assert d["flag"] is True

    def test_drift_5_percent_not_flagged(self):
        t = PerfTracker()
        t.record("1s-d", 10.0, 50)
        t.record("1s-d", 10.5, 50)  # 5% increase
        d = t.check_drift("1s-d")
        assert d["drift_pct"] == pytest.approx(5.0)
        assert d["flag"] is False

    def test_single_sample_no_drift(self):
        t = PerfTracker()
        t.record("1s-d", 10.0, 50)
        d = t.check_drift("1s-d")
        assert d["flag"] is False
        assert d["drift_pct"] == 0.0


class TestAllDrifts:
    def test_multiple_envtypes(self):
        t = PerfTracker()
        t.record("1s-d", 10.0, 50)
        t.record("2s-c", 20.0, 30)
        t.record("1s-d", 12.0, 50)  # 20% drift
        t.record("2s-c", 20.0, 30)  # 0% drift
        drifts = t.all_drifts()
        assert set(drifts.keys()) == {"1s-d", "2s-c"}
        assert drifts["1s-d"]["flag"] is True
        assert drifts["2s-c"]["flag"] is False


class TestSummary:
    def test_structure(self):
        t = PerfTracker()
        t.record("1s-d", 10.0, 50)
        t.record("1s-d", 12.0, 50)
        t.record("2s-c", 20.0, 30)
        s = t.summary()
        assert s["n_samples"] == 3
        assert s["envtypes_tracked"] == 2
        assert s["max_drift_pct"] == pytest.approx(20.0)
        assert s["flagged_count"] == 1

    def test_empty_tracker(self):
        t = PerfTracker()
        s = t.summary()
        assert s["n_samples"] == 0
        assert s["envtypes_tracked"] == 0
        assert s["max_drift_pct"] == 0.0
        assert s["flagged_count"] == 0


class TestSaveLoad:
    def test_roundtrip(self):
        t = PerfTracker()
        t.record("1s-d", 10.0, 50, git_sha="aaa")
        t.record("2s-c", 20.0, 30, git_sha="bbb")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        t.save(path)
        t2 = PerfTracker.load(path)

        assert len(t2._samples) == 2
        assert t2._samples[0].envtype == "1s-d"
        assert t2._samples[0].ms_per_step == 10.0
        assert t2._samples[0].git_sha == "aaa"
        assert t2._samples[1].envtype == "2s-c"
        assert t2._samples[1].n_episodes == 30

        # verify it's valid JSON list
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2

        os.unlink(path)


class TestFormatReport:
    def test_contains_envtypes(self):
        t = PerfTracker()
        t.record("1s-d", 10.0, 50)
        t.record("1s-d", 11.0, 50)
        t.record("2s-c", 20.0, 30)
        report = format_perf_report(t)
        assert "1s-d" in report
        assert "2s-c" in report
        assert "drift_pct" in report
        assert "Samples:" in report


class TestComparePerf:
    def test_calculation(self):
        a = [make_sample(envtype="1s-d", ms=10.0), make_sample(envtype="2s-c", ms=20.0)]
        b = [make_sample(envtype="1s-d", ms=12.0), make_sample(envtype="2s-c", ms=15.0)]
        result = compare_perf(a, b)
        assert result["1s-d"] == pytest.approx(20.0)   # (12-10)/10*100
        assert result["2s-c"] == pytest.approx(-25.0)   # (15-20)/20*100

    def test_no_common_envtype(self):
        a = [make_sample(envtype="1s-d", ms=10.0)]
        b = [make_sample(envtype="9s-x", ms=99.0)]
        result = compare_perf(a, b)
        assert result == {}
