"""Tests for flopyarcade.bench_drift (ROADMAP V8)."""

import json
import os
import tempfile

import pytest

from flopyarcade.bench_drift import (
    DriftPoint,
    DriftTracker,
    alert_summary,
    compare_drift,
    format_drift_report,
)


# ─── DriftPoint ──────────────────────────────────────────────────────────────

class TestDriftPoint:
    def test_creation(self):
        p = DriftPoint(date="2025-01-01", score=100.0, ci_low=95.0, ci_high=105.0)
        assert p.date == "2025-01-01"
        assert p.score == 100.0
        assert p.ci_low == 95.0
        assert p.ci_high == 105.0
        assert p.git_sha is None
        assert p.alert is False

    def test_creation_with_sha_and_alert(self):
        p = DriftPoint(date="2025-01-02", score=90.0, git_sha="abc123", alert=True)
        assert p.git_sha == "abc123"
        assert p.alert is True


# ─── DriftTracker init ───────────────────────────────────────────────────────

class TestDriftTrackerInit:
    def test_defaults(self):
        t = DriftTracker(envtype="1s-d")
        assert t.envtype == "1s-d"
        assert t.alert_threshold_pct == 5.0
        assert t.n_points() == 0

    def test_custom_threshold(self):
        t = DriftTracker(envtype="2s-c", alert_threshold_pct=10.0)
        assert t.alert_threshold_pct == 10.0


# ─── add_point ───────────────────────────────────────────────────────────────

class TestAddPoint:
    def test_appends(self):
        t = DriftTracker("1s-d")
        t.add_point("2025-01-01", 100.0)
        t.add_point("2025-01-02", 102.0)
        assert t.n_points() == 2
        assert t.points()[0].score == 100.0
        assert t.points()[1].score == 102.0

    def test_auto_alert_on_drop(self):
        t = DriftTracker("1s-d", alert_threshold_pct=5.0)
        t.add_point("2025-01-01", 100.0)
        t.add_point("2025-01-02", 94.0)  # 6% drop
        assert t.points()[1].alert is True

    def test_no_alert_on_improvement(self):
        t = DriftTracker("1s-d", alert_threshold_pct=5.0)
        t.add_point("2025-01-01", 100.0)
        t.add_point("2025-01-02", 105.0)  # improvement
        assert t.points()[1].alert is False

    def test_no_alert_below_threshold(self):
        t = DriftTracker("1s-d", alert_threshold_pct=5.0)
        t.add_point("2025-01-01", 100.0)
        t.add_point("2025-01-02", 96.0)  # 4% drop, below threshold
        assert t.points()[1].alert is False


# ─── Accessors ───────────────────────────────────────────────────────────────

class TestAccessors:
    def test_latest_best_worst(self):
        t = DriftTracker("1s-d")
        t.add_point("d1", 80.0)
        t.add_point("d2", 100.0)
        t.add_point("d3", 90.0)
        assert t.latest_score() == 90.0
        assert t.best_score() == 100.0
        assert t.worst_score() == 80.0

    def test_total_decline_pct(self):
        t = DriftTracker("1s-d")
        t.add_point("d1", 100.0)
        t.add_point("d2", 80.0)
        assert t.total_decline_pct() == pytest.approx(20.0)

    def test_single_point_no_decline(self):
        t = DriftTracker("1s-d")
        t.add_point("d1", 100.0)
        assert t.total_decline_pct() == 0.0
        assert t.latest_score() == 100.0
        assert t.best_score() == 100.0
        assert t.worst_score() == 100.0


# ─── detect_alerts ───────────────────────────────────────────────────────────

class TestDetectAlerts:
    def test_returns_flagged_points(self):
        t = DriftTracker("1s-d", alert_threshold_pct=5.0)
        t.add_point("d1", 100.0)
        t.add_point("d2", 90.0)   # 10% drop -> alert
        t.add_point("d3", 91.0)   # improvement -> no alert
        t.add_point("d4", 85.0)   # ~6.6% drop -> alert
        alerts = t.detect_alerts()
        assert len(alerts) == 2
        assert alerts[0].date == "d2"
        assert alerts[1].date == "d4"


# ─── trend ───────────────────────────────────────────────────────────────────

class TestTrend:
    def test_declining(self):
        t = DriftTracker("1s-d")
        for i, score in enumerate([100, 95, 90, 85, 80]):
            t.add_point(f"d{i}", float(score))
        tr = t.trend()
        assert tr["declining"] is True
        assert tr["improving"] is False
        assert tr["slope"] < 0
        assert 0 < tr["r_squared"] <= 1.0

    def test_improving(self):
        t = DriftTracker("1s-d")
        for i, score in enumerate([80, 85, 90, 95, 100]):
            t.add_point(f"d{i}", float(score))
        tr = t.trend()
        assert tr["improving"] is True
        assert tr["declining"] is False
        assert tr["slope"] > 0

    def test_single_point_trend(self):
        t = DriftTracker("1s-d")
        t.add_point("d0", 50.0)
        tr = t.trend()
        assert tr["slope"] == 0.0
        assert tr["declining"] is False
        assert tr["improving"] is False


# ─── ci_band_data ────────────────────────────────────────────────────────────

class TestCiBandData:
    def test_structure(self):
        t = DriftTracker("1s-d")
        t.add_point("d1", 100.0, ci_low=95.0, ci_high=105.0)
        t.add_point("d2", 90.0, ci_low=85.0, ci_high=95.0)
        data = t.ci_band_data()
        assert set(data.keys()) == {"dates", "scores", "ci_low", "ci_high", "alerts"}
        assert data["dates"] == ["d1", "d2"]
        assert data["scores"] == [100.0, 90.0]
        assert data["ci_low"] == [95.0, 85.0]
        assert data["ci_high"] == [105.0, 95.0]
        assert isinstance(data["alerts"], list)


# ─── rolling_mean ────────────────────────────────────────────────────────────

class TestRollingMean:
    def test_basic(self):
        t = DriftTracker("1s-d")
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        for i, s in enumerate(scores):
            t.add_point(f"d{i}", s)
        rm = t.rolling_mean(window=3)
        assert len(rm) == 5
        # first element: mean of [10]
        assert rm[0] == pytest.approx(10.0)
        # second: mean of [10, 20]
        assert rm[1] == pytest.approx(15.0)
        # third: mean of [10, 20, 30]
        assert rm[2] == pytest.approx(20.0)
        # fifth: mean of [30, 40, 50]
        assert rm[4] == pytest.approx(40.0)

    def test_empty(self):
        t = DriftTracker("1s-d")
        assert t.rolling_mean() == []


# ─── save / load roundtrip ───────────────────────────────────────────────────

class TestSerialization:
    def test_save_load_roundtrip(self):
        t = DriftTracker("1s-d", alert_threshold_pct=3.0)
        t.add_point("2025-01-01", 100.0, ci_low=95.0, ci_high=105.0, git_sha="abc")
        t.add_point("2025-01-02", 90.0)  # should trigger alert at 3% threshold

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        t.save_json(path)
        t2 = DriftTracker.load_json(path)

        assert t2.envtype == "1s-d"
        assert t2.alert_threshold_pct == 3.0
        assert t2.n_points() == 2
        assert t2.points()[0].date == "2025-01-01"
        assert t2.points()[0].score == 100.0
        assert t2.points()[0].git_sha == "abc"
        assert t2.points()[1].alert is True
        assert t2.latest_score() == 90.0

        os.unlink(path)


# ─── compare_drift ───────────────────────────────────────────────────────────

class TestCompareDrift:
    def test_multiple_envtypes(self):
        t1 = DriftTracker("1s-d")
        t1.add_point("d1", 100.0)
        t1.add_point("d2", 90.0)

        t2 = DriftTracker("2s-c")
        t2.add_point("d1", 200.0)
        t2.add_point("d2", 210.0)

        result = compare_drift({"1s-d": t1, "2s-c": t2})
        assert set(result.keys()) == {"1s-d", "2s-c"}
        assert result["1s-d"]["envtype"] == "1s-d"
        assert result["1s-d"]["latest"] == 90.0
        assert result["1s-d"]["decline_pct"] == pytest.approx(10.0)
        assert result["2s-c"]["latest"] == 210.0
        assert result["2s-c"]["decline_pct"] == pytest.approx(-5.0)


# ─── format_drift_report ─────────────────────────────────────────────────────

class TestFormatReport:
    def test_contains_trend_text(self):
        t = DriftTracker("1s-d")
        t.add_point("d1", 100.0)
        t.add_point("d2", 90.0)
        t.add_point("d3", 80.0)
        report = format_drift_report(t)
        assert "Benchmark Drift Report" in report
        assert "1s-d" in report
        assert "declining" in report
        assert "Latest score" in report


# ─── alert_summary ───────────────────────────────────────────────────────────

class TestAlertSummary:
    def test_structure_with_alerts(self):
        t = DriftTracker("1s-d", alert_threshold_pct=5.0)
        t.add_point("d1", 100.0)
        t.add_point("d2", 90.0)   # alert
        t.add_point("d3", 92.0)   # no alert
        t.add_point("d4", 80.0)   # alert

        summary = alert_summary(t.points())
        assert summary["n_alerts"] == 2
        assert summary["first_alert_date"] == "d2"
        assert summary["last_alert_date"] == "d4"
        assert summary["max_drop_pct"] > 0

    def test_no_alerts(self):
        points = [DriftPoint(date="d1", score=100.0), DriftPoint(date="d2", score=105.0)]
        summary = alert_summary(points)
        assert summary["n_alerts"] == 0
        assert summary["first_alert_date"] is None
        assert summary["last_alert_date"] is None
        assert summary["max_drop_pct"] == 0.0


# ─── 10-point gradual decline ───────────────────────────────────────────────

class TestGradualDecline:
    def test_10_points_gradual(self):
        t = DriftTracker("1s-d", alert_threshold_pct=5.0)
        scores = [100, 97, 94, 91, 88, 85, 82, 79, 76, 73]
        for i, s in enumerate(scores):
            t.add_point(f"d{i:02d}", float(s))

        assert t.n_points() == 10
        assert t.best_score() == 100.0
        assert t.worst_score() == 73.0
        assert t.latest_score() == 73.0
        assert t.total_decline_pct() == pytest.approx(27.0)
        tr = t.trend()
        assert tr["declining"] is True
        # With 3% drops per step and 5% threshold, no individual step triggers alert
        assert len(t.detect_alerts()) == 0
        rm = t.rolling_mean(window=5)
        assert len(rm) == 10
        # rolling mean should be monotonically decreasing for this data
        for i in range(1, len(rm)):
            assert rm[i] <= rm[i - 1] + 1e-9
