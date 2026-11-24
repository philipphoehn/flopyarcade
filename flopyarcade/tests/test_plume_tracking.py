"""V1: Plume front tracking tests."""

import json
import math
import os
import tempfile

import pytest

from flopyarcade.plume_tracking import (
    PlumePoint,
    PlumeTracker,
    compare_plumes,
    format_plume_report,
)


# ------------------------------------------------------------------ helpers
def _make_tracker(n_points=5, episode=1, start_t=0, dist_step=2.0):
    """Create a tracker with known data for deterministic tests."""
    tracker = PlumeTracker(source_pos=(0.0, 0.0))
    for i in range(n_points):
        tracker.record(t=start_t + i, positions=[(dist_step * (i + 1), 0.0)], episode=episode)
    return tracker


# ------------------------------------------------------------------ tests
def test_plume_point_creation():
    """PlumePoint dataclass stores t, distance, episode correctly."""
    p = PlumePoint(t=10, distance=3.5, episode=2)
    assert p.t == 10
    assert p.distance == 3.5
    assert p.episode == 2


def test_plume_tracker_init_defaults():
    """Default source_pos is (0.0, 0.0) and no points recorded."""
    tracker = PlumeTracker()
    assert tracker.source_pos == (0.0, 0.0)
    assert tracker.front_positions() == []
    assert tracker.max_distance() == 0.0
    assert tracker.min_distance() == 0.0
    assert tracker.mean_distance() == 0.0


def test_record_computes_correct_distance():
    """record() with known position computes correct Euclidean distance."""
    tracker = PlumeTracker(source_pos=(0.0, 0.0))
    tracker.record(t=0, positions=[(3.0, 4.0)], episode=1)
    # distance = hypot(3, 4) = 5.0
    assert tracker.front_positions()[0].distance == pytest.approx(5.0)


def test_record_multiple_particles_takes_max():
    """record() with multiple particles stores only the furthest."""
    tracker = PlumeTracker(source_pos=(0.0, 0.0))
    # particles at (1,0), (3,4), (0,2) -> distances 1, 5, 2
    tracker.record(t=0, positions=[(1.0, 0.0), (3.0, 4.0), (0.0, 2.0)], episode=1)
    assert tracker.front_positions()[0].distance == pytest.approx(5.0)


def test_front_positions_length():
    """front_positions() returns all recorded points."""
    tracker = _make_tracker(n_points=7)
    assert len(tracker.front_positions()) == 7


def test_max_min_mean_distance():
    """max/min/mean distance computed correctly."""
    tracker = PlumeTracker(source_pos=(0.0, 0.0))
    tracker.record(t=0, positions=[(1.0, 0.0)], episode=1)  # dist=1
    tracker.record(t=1, positions=[(3.0, 0.0)], episode=1)  # dist=3
    tracker.record(t=2, positions=[(2.0, 0.0)], episode=1)  # dist=2
    assert tracker.max_distance() == pytest.approx(3.0)
    assert tracker.min_distance() == pytest.approx(1.0)
    assert tracker.mean_distance() == pytest.approx(2.0)


def test_front_speed_calculation():
    """front_speed returns correct mean/max/min speeds."""
    tracker = PlumeTracker(source_pos=(0.0, 0.0))
    # t=0 dist=0, t=1 dist=2, t=3 dist=8
    tracker.record(t=0, positions=[(0.0, 0.0)], episode=1)
    tracker.record(t=1, positions=[(2.0, 0.0)], episode=1)
    tracker.record(t=3, positions=[(8.0, 0.0)], episode=1)
    speed = tracker.front_speed()
    # speeds: 2/1=2, 6/2=3
    assert speed["mean_speed"] == pytest.approx(2.5)
    assert speed["max_speed"] == pytest.approx(3.0)
    assert speed["min_speed"] == pytest.approx(2.0)


def test_per_episode_stats_structure():
    """per_episode_stats returns correct keys and values."""
    tracker = PlumeTracker(source_pos=(0.0, 0.0))
    tracker.record(t=0, positions=[(1.0, 0.0)], episode=1)
    tracker.record(t=1, positions=[(3.0, 0.0)], episode=1)
    tracker.record(t=2, positions=[(2.0, 0.0)], episode=2)
    stats = tracker.per_episode_stats()
    assert 1 in stats and 2 in stats
    assert stats[1]["min_dist"] == pytest.approx(1.0)
    assert stats[1]["max_dist"] == pytest.approx(3.0)
    assert stats[1]["mean_dist"] == pytest.approx(2.0)
    assert stats[1]["n_points"] == 2
    assert stats[2]["n_points"] == 1


def test_boxplot_data_structure():
    """boxplot_data has all required keys."""
    tracker = _make_tracker(n_points=6, episode=1)
    bp = tracker.boxplot_data()
    for key in ("episodes", "min", "q1", "median", "q3", "max", "outliers"):
        assert key in bp, f"missing key {key}"
    assert bp["episodes"] == [1]
    assert len(bp["min"]) == 1
    assert len(bp["q1"]) == 1
    assert len(bp["median"]) == 1
    assert len(bp["q3"]) == 1
    assert len(bp["max"]) == 1


def test_boxplot_data_correct_quartiles():
    """boxplot_data computes correct quartiles for known data [1,2,3,4,5,6,7,8,9,10]."""
    tracker = PlumeTracker(source_pos=(0.0, 0.0))
    for i in range(10):
        tracker.record(t=i, positions=[(float(i + 1), 0.0)], episode=1)
    bp = tracker.boxplot_data()
    # values: 1..10, n=10
    # q1 = pct(0.25): idx=2.25 -> 3*0.75 + 4*0.25 = 3.25
    # median = pct(0.50): idx=4.5 -> 5*0.5 + 6*0.5 = 5.5
    # q3 = pct(0.75): idx=6.75 -> 7*0.25 + 8*0.75 = 7.75
    assert bp["q1"][0] == pytest.approx(3.25)
    assert bp["median"][0] == pytest.approx(5.5)
    assert bp["q3"][0] == pytest.approx(7.75)


def test_trend_slope_positive_increasing():
    """trend() slope is positive when distance increases over time."""
    tracker = PlumeTracker(source_pos=(0.0, 0.0))
    for i in range(5):
        tracker.record(t=i, positions=[(float(i + 1), 0.0)], episode=1)
    tr = tracker.trend()
    assert tr["slope"] > 0
    assert tr["n_points"] == 5


def test_trend_r_squared_in_range():
    """r_squared is between 0 and 1."""
    tracker = _make_tracker(n_points=5)
    tr = tracker.trend()
    assert 0.0 <= tr["r_squared"] <= 1.0


def test_save_load_roundtrip():
    """save_json / load_json roundtrip preserves data."""
    tracker = PlumeTracker(source_pos=(1.0, 2.0))
    tracker.record(t=0, positions=[(4.0, 6.0)], episode=1)  # dist=5
    tracker.record(t=1, positions=[(7.0, 6.0)], episode=1)  # dist=sqrt(36+16)=sqrt(52)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        tracker.save_json(path)
        loaded = PlumeTracker.load_json(path)
        assert loaded.source_pos == (1.0, 2.0)
        assert len(loaded.front_positions()) == 2
        assert loaded.front_positions()[0].distance == pytest.approx(5.0)
        assert loaded.max_distance() == pytest.approx(tracker.max_distance())
    finally:
        os.unlink(path)


def test_compare_plumes_structure():
    """compare_plumes returns all required keys."""
    a = PlumeTracker(source_pos=(0.0, 0.0))
    a.record(t=0, positions=[(5.0, 0.0)], episode=1)
    b = PlumeTracker(source_pos=(0.0, 0.0))
    b.record(t=0, positions=[(3.0, 0.0)], episode=1)
    result = compare_plumes(a, b)
    for key in ("a_max", "b_max", "a_mean", "b_mean", "faster", "speed_ratio"):
        assert key in result, f"missing key {key}"
    assert result["a_max"] == pytest.approx(5.0)
    assert result["b_max"] == pytest.approx(3.0)
    assert result["faster"] == "A"


def test_format_plume_report_contains_text():
    """format_plume_report returns non-empty string with key info."""
    tracker = _make_tracker(n_points=3)
    report = format_plume_report(tracker)
    assert isinstance(report, str)
    assert len(report) > 0
    assert "Plume Front Tracking Report" in report
    assert "Max distance" in report
    assert "Episodes" in report


def test_single_episode_single_point():
    """Tracker with exactly one episode and one point works correctly."""
    tracker = PlumeTracker(source_pos=(0.0, 0.0))
    tracker.record(t=0, positions=[(7.0, 0.0)], episode=42)
    assert len(tracker.front_positions()) == 1
    assert tracker.max_distance() == pytest.approx(7.0)
    stats = tracker.per_episode_stats()
    assert 42 in stats
    assert stats[42]["n_points"] == 1
    assert stats[42]["min_dist"] == pytest.approx(7.0)
    speed = tracker.front_speed()
    assert speed["mean_speed"] == 0.0  # < 2 points -> zero speed
    tr = tracker.trend()
    assert tr["n_points"] == 1
    assert tr["slope"] == 0.0
