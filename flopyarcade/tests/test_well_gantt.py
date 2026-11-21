"""Tests for flopyarcade.well_gantt (V7)."""
import json
import os
import tempfile

import pytest

from flopyarcade.well_gantt import (
    WellStep,
    WellTimeline,
    compare_timelines,
    format_gantt_report,
)


# ── WellStep ──────────────────────────────────────────────────────────────

def test_wellstep_creation():
    ws = WellStep(step=3, well_id=1, pumping_rate=5.5, active=True)
    assert ws.step == 3
    assert ws.well_id == 1
    assert ws.pumping_rate == 5.5
    assert ws.active is True


# ── WellTimeline init ─────────────────────────────────────────────────────

def test_welltline_init():
    tl = WellTimeline(n_wells=4, n_steps=10)
    assert tl.n_wells == 4
    assert tl.n_steps == 10
    # all rates should be zero initially
    for w in range(4):
        for s in range(10):
            assert tl.get_rate(s, w) == 0.0


# ── set_rate / get_rate ───────────────────────────────────────────────────

def test_set_rate_and_get_rate():
    tl = WellTimeline(n_wells=2, n_steps=5)
    tl.set_rate(step=0, well_id=0, rate=10.0)
    tl.set_rate(step=2, well_id=1, rate=-3.0)
    assert tl.get_rate(0, 0) == 10.0
    assert tl.get_rate(2, 1) == -3.0
    # untouched stays zero
    assert tl.get_rate(0, 1) == 0.0


# ── set_rate auto-sets active ─────────────────────────────────────────────

def test_set_rate_auto_sets_active():
    tl = WellTimeline(n_wells=1, n_steps=4)
    tl.set_rate(0, 0, 5.0)
    tl.set_rate(1, 0, 0.0)
    tl.set_rate(2, 0, -2.0)
    tl.set_rate(3, 0, 0.0)
    steps = tl.well_timeline(0)
    assert steps[0].active is True
    assert steps[1].active is False
    assert steps[2].active is True
    assert steps[3].active is False


# ── well_timeline length ──────────────────────────────────────────────────

def test_well_timeline_length():
    tl = WellTimeline(n_wells=3, n_steps=7)
    tl.set_rate(1, 0, 4.2)
    tl.set_rate(3, 2, -1.5)
    for w in range(3):
        tl_list = tl.well_timeline(w)
        assert len(tl_list) == 7
        assert all(isinstance(s, WellStep) for s in tl_list)


# ── total_pumpage ─────────────────────────────────────────────────────────

def test_total_pumpage():
    tl = WellTimeline(n_wells=2, n_steps=5)
    tl.set_rate(0, 0, 10.0)
    tl.set_rate(1, 0, -5.0)
    tl.set_rate(2, 0, 3.0)
    # well 0: |10| + |-5| + |3| = 18
    assert tl.total_pumpage(0) == pytest.approx(18.0)
    # well 1: all zero
    assert tl.total_pumpage(1) == 0.0


# ── active_fraction ───────────────────────────────────────────────────────

def test_active_fraction():
    tl = WellTimeline(n_wells=1, n_steps=4)
    tl.set_rate(0, 0, 1.0)
    tl.set_rate(2, 0, 2.0)
    # 2 active out of 4
    assert tl.active_fraction(0) == pytest.approx(0.5)


# ── gantt_data structure ──────────────────────────────────────────────────

def test_gantt_data_structure():
    tl = WellTimeline(n_wells=3, n_steps=6)
    tl.set_rate(0, 0, 1.0)
    tl.set_rate(1, 1, 2.0)
    tl.set_rate(2, 2, 3.0)
    data = tl.gantt_data()
    assert set(data.keys()) == {"wells", "steps", "rates", "active"}
    assert data["wells"] == [0, 1, 2]
    assert data["steps"] == [0, 1, 2, 3, 4, 5]
    assert len(data["rates"]) == 3
    assert len(data["rates"][0]) == 6
    assert len(data["active"]) == 3
    assert len(data["active"][0]) == 6
    assert data["active"][0][0] is True
    assert data["active"][0][1] is False
    assert data["rates"][1][1] == 2.0


# ── stacked_bar_data structure ────────────────────────────────────────────

def test_stacked_bar_data_structure():
    tl = WellTimeline(n_wells=2, n_steps=4)
    tl.set_rate(0, 0, 5.0)
    tl.set_rate(0, 1, 3.0)
    tl.set_rate(2, 1, -1.0)
    data = tl.stacked_bar_data()
    assert set(data.keys()) == {"steps", "per_well", "total"}
    assert data["steps"] == [0, 1, 2, 3]
    assert 0 in data["per_well"] and 1 in data["per_well"]
    assert len(data["per_well"][0]) == 4
    assert data["total"][0] == pytest.approx(8.0)
    assert data["total"][2] == pytest.approx(-1.0)


# ── peak_step ─────────────────────────────────────────────────────────────

def test_peak_step():
    tl = WellTimeline(n_wells=1, n_steps=5)
    tl.set_rate(0, 0, 1.0)
    tl.set_rate(1, 0, -8.0)   # max abs
    tl.set_rate(2, 0, 3.0)
    tl.set_rate(3, 0, 2.0)
    tl.set_rate(4, 0, 5.0)
    peak = tl.peak_step(0)
    assert peak["step"] == 1
    assert peak["rate"] == -8.0


# ── duty_cycle ────────────────────────────────────────────────────────────

def test_duty_cycle():
    tl = WellTimeline(n_wells=1, n_steps=5)
    tl.set_rate(0, 0, 1.0)
    tl.set_rate(1, 0, 2.0)
    tl.set_rate(2, 0, 3.0)
    # 3 active out of 5
    assert tl.duty_cycle(0) == pytest.approx(0.6)
    assert tl.duty_cycle(0) == tl.active_fraction(0)


# ── save / load roundtrip ─────────────────────────────────────────────────

def test_save_load_roundtrip():
    tl = WellTimeline(n_wells=3, n_steps=4)
    tl.set_rate(0, 0, 10.0)
    tl.set_rate(1, 1, -4.5)
    tl.set_rate(3, 2, 7.0)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        tl.save_json(path)
        loaded = WellTimeline.load_json(path)
        assert loaded.n_wells == 3
        assert loaded.n_steps == 4
        assert loaded.get_rate(0, 0) == 10.0
        assert loaded.get_rate(1, 1) == -4.5
        assert loaded.get_rate(3, 2) == 7.0
        assert loaded.total_pumpage(0) == pytest.approx(10.0)
    finally:
        os.unlink(path)


# ── compare_timelines ─────────────────────────────────────────────────────

def test_compare_timelines():
    tl_a = WellTimeline(n_wells=2, n_steps=4)
    tl_a.set_rate(0, 0, 10.0)
    tl_a.set_rate(1, 1, 5.0)

    tl_b = WellTimeline(n_wells=2, n_steps=4)
    tl_b.set_rate(0, 0, 20.0)
    tl_b.set_rate(1, 1, 5.0)

    result = compare_timelines(tl_a, tl_b)
    assert "wells" in result
    assert len(result["wells"]) == 2

    w0 = result["wells"][0]
    assert w0["well_id"] == 0
    assert w0["pumpage_a"] == pytest.approx(10.0)
    assert w0["pumpage_b"] == pytest.approx(20.0)
    assert w0["diff"] == pytest.approx(-10.0)
    assert w0["faster"] == "b"

    w1 = result["wells"][1]
    assert w1["faster"] == "equal"


# ── format_gantt_report ───────────────────────────────────────────────────

def test_format_gantt_report():
    tl = WellTimeline(n_wells=3, n_steps=5)
    tl.set_rate(0, 0, 100.0)
    tl.set_rate(1, 1, 50.0)
    tl.set_rate(2, 2, 25.0)
    report = format_gantt_report(tl)
    # should contain well ids as strings
    assert "0" in report
    assert "1" in report
    assert "2" in report
    # header keywords
    assert "pumpage" in report.lower()
    assert "peak" in report.lower()
    # 3 data rows + header + separator = 5 lines
    lines = report.strip().split("\n")
    assert len(lines) == 5


# ── 3 wells 10 steps integration ──────────────────────────────────────────

def test_integration_3wells_10steps():
    tl = WellTimeline(n_wells=3, n_steps=10)
    # well 0: active at steps 0, 2, 4, 6, 8
    for s in [0, 2, 4, 6, 8]:
        tl.set_rate(s, 0, 10.0)
    # well 1: active at steps 1, 3, 5, 7, 9
    for s in [1, 3, 5, 7, 9]:
        tl.set_rate(s, 1, 20.0)
    # well 2: active at steps 0, 5
    tl.set_rate(0, 2, 5.0)
    tl.set_rate(5, 2, 15.0)

    # total pumpage
    assert tl.total_pumpage(0) == pytest.approx(50.0)
    assert tl.total_pumpage(1) == pytest.approx(100.0)
    assert tl.total_pumpage(2) == pytest.approx(20.0)

    # active fraction
    assert tl.active_fraction(0) == pytest.approx(0.5)
    assert tl.active_fraction(1) == pytest.approx(0.5)
    assert tl.active_fraction(2) == pytest.approx(0.2)

    # gantt data
    g = tl.gantt_data()
    assert len(g["wells"]) == 3
    assert len(g["steps"]) == 10
    assert g["rates"][0][0] == 10.0
    assert g["rates"][1][1] == 20.0

    # stacked bar
    sb = tl.stacked_bar_data()
    assert sb["total"][0] == pytest.approx(15.0)  # well0:10 + well2:5
    assert sb["total"][1] == pytest.approx(20.0)  # well1:20

    # peak step
    assert tl.peak_step(0)["rate"] == 10.0
    assert tl.peak_step(1)["rate"] == 20.0
    assert tl.peak_step(2)["step"] == 5
    assert tl.peak_step(2)["rate"] == 15.0

    # report
    report = format_gantt_report(tl)
    assert len(report.strip().split("\n")) == 5


# ── all-zero rates ────────────────────────────────────────────────────────

def test_all_zero_rates():
    tl = WellTimeline(n_wells=2, n_steps=6)
    # no set_rate calls — everything stays zero
    for w in range(2):
        assert tl.total_pumpage(w) == 0.0
        assert tl.active_fraction(w) == 0.0
        assert tl.duty_cycle(w) == 0.0

    # peak should be step 0, rate 0
    for w in range(2):
        peak = tl.peak_step(w)
        assert peak["step"] == 0
        assert peak["rate"] == 0.0

    # gantt active all False
    g = tl.gantt_data()
    for w in range(2):
        for s in range(6):
            assert g["active"][w][s] is False

    # stacked bar totals all zero
    sb = tl.stacked_bar_data()
    assert all(t == 0.0 for t in sb["total"])
