"""Tests for flopyarcade.field_snapshots (ROADMAP P5)."""

import os
import tempfile

import numpy as np
import pytest

from flopyarcade.field_snapshots import (
    FieldRecorder,
    FieldSnapshot,
    compare_fields,
    compute_field_stats,
    format_field_report,
    temporal_change,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_grid(h=3, w=4, fill=1.0):
    return [[fill] * w for _ in range(h)]


def make_grid_ramp(h=3, w=4, offset=0.0):
    return [[(r * w + c + offset) for c in range(w)] for r in range(h)]


# ---------------------------------------------------------------------------
# FieldSnapshot creation
# ---------------------------------------------------------------------------

def test_field_snapshot_creation_without_velocity():
    head = make_grid(3, 4, 2.5)
    snap = FieldSnapshot(step=10, head=head)
    assert snap.step == 10
    assert snap.head == head
    assert snap.velocity_x is None
    assert snap.velocity_y is None


def test_field_snapshot_creation_with_velocity():
    head = make_grid(2, 2, 1.0)
    vx = make_grid(2, 2, 0.5)
    vy = make_grid(2, 2, -0.3)
    snap = FieldSnapshot(step=5, head=head, velocity_x=vx, velocity_y=vy)
    assert snap.step == 5
    assert snap.head == head
    assert snap.velocity_x == vx
    assert snap.velocity_y == vy


# ---------------------------------------------------------------------------
# FieldRecorder init
# ---------------------------------------------------------------------------

def test_field_recorder_init_defaults():
    rec = FieldRecorder(envtype="5s-c-cost", grid_shape=(4, 4))
    assert rec.envtype == "5s-c-cost"
    assert rec.grid_shape == (4, 4)
    assert rec.record_every == 1
    assert rec.n_snapshots() == 0
    assert rec.snapshots() == []


# ---------------------------------------------------------------------------
# record respects record_every
# ---------------------------------------------------------------------------

def test_record_respects_record_every():
    rec = FieldRecorder(envtype="test", grid_shape=(3, 3), record_every=3)
    head = make_grid(3, 3, 1.0)
    for step in range(1, 11):
        rec.record(step, head)
    # Should record at steps 3, 6, 9
    assert rec.n_snapshots() == 3
    recorded_steps = [s.step for s in rec.snapshots()]
    assert recorded_steps == [3, 6, 9]


def test_record_appends_when_step_matches():
    rec = FieldRecorder(envtype="test", grid_shape=(2, 2), record_every=2)
    h1 = make_grid(2, 2, 1.0)
    h2 = make_grid(2, 2, 2.0)
    rec.record(2, h1)
    rec.record(3, h2)  # should NOT record
    rec.record(4, h2)
    assert rec.n_snapshots() == 2
    assert rec.snapshots()[0].head == h1
    assert rec.snapshots()[1].head == h2


# ---------------------------------------------------------------------------
# n_snapshots correct
# ---------------------------------------------------------------------------

def test_n_snapshots_correct():
    rec = FieldRecorder(envtype="test", grid_shape=(2, 2))
    assert rec.n_snapshots() == 0
    rec.record(1, make_grid(2, 2, 1.0))
    assert rec.n_snapshots() == 1
    rec.record(2, make_grid(2, 2, 2.0))
    rec.record(3, make_grid(2, 2, 3.0))
    assert rec.n_snapshots() == 3


# ---------------------------------------------------------------------------
# save_npz creates file
# ---------------------------------------------------------------------------

def test_save_npz_creates_file():
    rec = FieldRecorder(envtype="test", grid_shape=(3, 3), record_every=1)
    rec.record(1, make_grid(3, 3, 1.0))
    rec.record(2, make_grid(3, 3, 2.0))

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        path = tmp.name
    try:
        rec.save_npz(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        # Verify it's a valid npz
        with np.load(path, allow_pickle=True) as data:
            assert "heads" in data
            assert "steps" in data
            assert data["heads"].shape == (2, 3, 3)
            assert len(data["steps"]) == 2
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# load_npz roundtrip
# ---------------------------------------------------------------------------

def test_load_npz_roundtrip_heads_preserved():
    rec = FieldRecorder(envtype="roundtrip", grid_shape=(2, 3), record_every=2)
    h1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    h2 = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    rec.record(2, h1)
    rec.record(4, h2)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        path = tmp.name
    try:
        rec.save_npz(path)
        loaded = FieldRecorder.load_npz(path)
        assert loaded.envtype == "roundtrip"
        assert loaded.grid_shape == (2, 3)
        assert loaded.record_every == 2
        assert loaded.n_snapshots() == 2
        # heads preserved
        assert np.allclose(loaded.snapshots()[0].head, h1)
        assert np.allclose(loaded.snapshots()[1].head, h2)
        assert loaded.snapshots()[0].step == 2
        assert loaded.snapshots()[1].step == 4
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# summary structure and values
# ---------------------------------------------------------------------------

def test_summary_structure_and_values():
    rec = FieldRecorder(envtype="test", grid_shape=(2, 2), record_every=1)
    rec.record(1, [[1.0, 2.0], [3.0, 4.0]])
    rec.record(2, [[5.0, 6.0], [7.0, 8.0]])

    s = rec.summary()
    assert s["n_snapshots"] == 2
    assert s["grid_shape"] == [2, 2]
    assert s["record_every"] == 1
    assert s["head_min"] == 1.0
    assert s["head_max"] == 8.0
    assert s["head_mean_overall"] == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# compute_field_stats correct mean
# ---------------------------------------------------------------------------

def test_compute_field_stats_correct_mean():
    head = [[1.0, 2.0], [3.0, 4.0]]
    snap = FieldSnapshot(step=1, head=head)
    stats = compute_field_stats(snap)
    assert stats["head_mean"] == pytest.approx(2.5)
    assert stats["head_std"] == pytest.approx(np.std([1, 2, 3, 4]))
    assert stats["head_min"] == 1.0
    assert stats["head_max"] == 4.0
    assert "vx_mean" not in stats
    assert "vy_mean" not in stats


# ---------------------------------------------------------------------------
# compute_field_stats with velocity
# ---------------------------------------------------------------------------

def test_compute_field_stats_with_velocity():
    head = [[1.0, 2.0], [3.0, 4.0]]
    vx = [[0.1, 0.2], [0.3, 0.4]]
    vy = [[-0.1, -0.2], [-0.3, -0.4]]
    snap = FieldSnapshot(step=1, head=head, velocity_x=vx, velocity_y=vy)
    stats = compute_field_stats(snap)
    assert stats["vx_mean"] == pytest.approx(0.25)
    assert stats["vy_mean"] == pytest.approx(-0.25)


# ---------------------------------------------------------------------------
# temporal_change basic
# ---------------------------------------------------------------------------

def test_temporal_change_basic():
    rec = FieldRecorder(envtype="test", grid_shape=(2, 2))
    rec.record(1, [[0.0, 0.0], [0.0, 0.0]])
    rec.record(2, [[1.0, 1.0], [1.0, 1.0]])
    rec.record(3, [[3.0, 3.0], [3.0, 3.0]])

    tc = temporal_change(rec)
    assert tc["n_transitions"] == 2
    # |1-0| = 1, |3-1| = 2 → mean = 1.5, max = 2.0
    assert tc["mean_abs_change"] == pytest.approx(1.5)
    assert tc["max_abs_change"] == pytest.approx(2.0)
    assert len(tc["cells_most_changed"]) == 4  # only 4 cells in 2x2 grid


# ---------------------------------------------------------------------------
# temporal_change single snapshot
# ---------------------------------------------------------------------------

def test_temporal_change_single_snapshot():
    rec = FieldRecorder(envtype="test", grid_shape=(2, 2))
    rec.record(1, [[1.0, 2.0], [3.0, 4.0]])
    tc = temporal_change(rec)
    assert tc["n_transitions"] == 0
    assert tc["mean_abs_change"] == 0.0
    assert tc["max_abs_change"] == 0.0
    assert tc["cells_most_changed"] == []


# ---------------------------------------------------------------------------
# compare_fields identical
# ---------------------------------------------------------------------------

def test_compare_fields_identical():
    head = [[1.0, 2.0], [3.0, 4.0]]
    snap_a = FieldSnapshot(step=1, head=head)
    snap_b = FieldSnapshot(step=2, head=head)
    cmp = compare_fields(snap_a, snap_b)
    assert cmp["abs_diff_max"] == 0.0
    assert cmp["abs_diff_mean"] == 0.0
    assert cmp["rel_diff_mean"] == 0.0


# ---------------------------------------------------------------------------
# compare_fields different
# ---------------------------------------------------------------------------

def test_compare_fields_different():
    snap_a = FieldSnapshot(step=1, head=[[0.0, 0.0], [0.0, 0.0]])
    snap_b = FieldSnapshot(step=2, head=[[1.0, 2.0], [3.0, 4.0]])
    cmp = compare_fields(snap_a, snap_b)
    assert cmp["abs_diff_max"] == pytest.approx(4.0)
    assert cmp["abs_diff_mean"] == pytest.approx(2.5)
    # rel_diff: for cell (0,0) denom=max(0,1,eps)=1, rel=1/1=1
    # for cell (0,1) denom=2, rel=2/2=1
    # for cell (1,0) denom=3, rel=3/3=1
    # for cell (1,1) denom=4, rel=4/4=1
    assert cmp["rel_diff_mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# format_field_report contains text
# ---------------------------------------------------------------------------

def test_format_field_report_contains_text():
    rec = FieldRecorder(envtype="5s-c-cost", grid_shape=(3, 3), record_every=2)
    rec.record(2, make_grid(3, 3, 1.0))
    rec.record(4, make_grid(3, 3, 2.0))

    report = format_field_report(rec)
    assert "Field Snapshot Report" in report
    assert "5s-c-cost" in report
    assert "3" in str(report)
    assert "Temporal change" in report


# ---------------------------------------------------------------------------
# 5x5 grid with 10 steps
# ---------------------------------------------------------------------------

def test_5x5_grid_with_10_steps():
    h, w = 5, 5
    rec = FieldRecorder(envtype="test", grid_shape=(h, w), record_every=1)
    for step in range(1, 11):
        grid = [[(r + c + step) % 7 for c in range(w)] for r in range(h)]
        rec.record(step, grid)

    assert rec.n_snapshots() == 10

    s = rec.summary()
    assert s["n_snapshots"] == 10
    assert s["grid_shape"] == [5, 5]

    tc = temporal_change(rec)
    assert tc["n_transitions"] == 9
    assert tc["max_abs_change"] > 0

    # save and load roundtrip
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        path = tmp.name
    try:
        rec.save_npz(path)
        loaded = FieldRecorder.load_npz(path)
        assert loaded.n_snapshots() == 10
        assert loaded.grid_shape == (5, 5)
        for i in range(10):
            assert np.allclose(loaded.snapshots()[i].head, rec.snapshots()[i].head)
    finally:
        os.unlink(path)
