"""Tests for flopyarcade.head_change module (ROADMAP V2)."""

import json
import math
import os
import tempfile

import pytest

from flopyarcade.head_change import (
    CellChange,
    HeadChangeGrid,
    compare_grids,
    format_head_report,
)


# ---------- CellChange ----------

def test_cell_change_creation():
    c = CellChange(row=1, col=2, delta_h=0.52, flow_direction=(1.0, 0.0))
    assert c.row == 1
    assert c.col == 2
    assert c.delta_h == pytest.approx(0.52)
    assert c.flow_direction == (1.0, 0.0)

    c2 = CellChange(row=0, col=0, delta_h=-0.3)
    assert c2.flow_direction is None


# ---------- HeadChangeGrid init ----------

def test_head_change_grid_init():
    g = HeadChangeGrid(rows=3, cols=5)
    assert g.rows == 3
    assert g.cols == 5
    # unset cells return zero
    c = g.cell(0, 0)
    assert c.delta_h == 0.0
    assert c.flow_direction is None


# ---------- set_change and cell retrieval ----------

def test_set_change_and_cell_retrieval():
    g = HeadChangeGrid(rows=3, cols=3)
    g.set_change(1, 1, 0.75, flow_dir=(0.6, 0.8))
    c = g.cell(1, 1)
    assert c.row == 1
    assert c.col == 1
    assert c.delta_h == pytest.approx(0.75)
    assert c.flow_direction == (0.6, 0.8)

    # overwriting
    g.set_change(1, 1, -0.25)
    c2 = g.cell(1, 1)
    assert c2.delta_h == pytest.approx(-0.25)
    assert c2.flow_direction is None

    # out of bounds
    with pytest.raises(IndexError):
        g.set_change(5, 0, 1.0)


# ---------- max_change ----------

def test_max_change_correct():
    g = HeadChangeGrid(rows=2, cols=2)
    g.set_change(0, 0, 0.1)
    g.set_change(0, 1, -3.5)
    g.set_change(1, 0, 2.0)
    g.set_change(1, 1, 0.5)
    mc = g.max_change()
    assert mc.row == 0
    assert mc.col == 1
    assert mc.delta_h == pytest.approx(-3.5)


# ---------- min_change ----------

def test_min_change_correct():
    g = HeadChangeGrid(rows=2, cols=2)
    g.set_change(0, 0, 0.1)
    g.set_change(0, 1, -3.5)
    g.set_change(1, 0, 2.0)
    g.set_change(1, 1, 0.5)
    mc = g.min_change()
    assert mc.row == 0
    assert mc.col == 0
    assert mc.delta_h == pytest.approx(0.1)


# ---------- mean_change ----------

def test_mean_change():
    g = HeadChangeGrid(rows=2, cols=2)
    g.set_change(0, 0, 1.0)
    g.set_change(0, 1, -1.0)
    g.set_change(1, 0, 2.0)
    g.set_change(1, 1, 4.0)
    assert g.mean_change() == pytest.approx(1.5)

    # empty grid
    g2 = HeadChangeGrid(rows=1, cols=1)
    assert g2.mean_change() == 0.0


# ---------- rms_change ----------

def test_rms_change():
    g = HeadChangeGrid(rows=2, cols=2)
    g.set_change(0, 0, 3.0)
    g.set_change(0, 1, 4.0)
    # rms = sqrt((9+16)/2) = sqrt(12.5)
    expected = math.sqrt(12.5)
    assert g.rms_change() == pytest.approx(expected)

    # empty
    g2 = HeadChangeGrid(rows=1, cols=1)
    assert g2.rms_change() == 0.0


# ---------- gradient_field shape ----------

def test_gradient_field_shape():
    g = HeadChangeGrid(rows=4, cols=6)
    for r in range(4):
        for c in range(6):
            g.set_change(r, c, float(r + c))
    grad = g.gradient_field()
    assert len(grad) == 4
    assert all(len(row) == 6 for row in grad)
    for row in grad:
        for val in row:
            assert len(val) == 2  # (dh_dr, dh_dc)


# ---------- gradient_field values (linear → constant gradient) ----------

def test_gradient_field_linear_field():
    # h = 2*r + 3*c  →  dh/dr = 2, dh/dc = 3 everywhere
    rows, cols = 5, 5
    g = HeadChangeGrid(rows=rows, cols=cols)
    for r in range(rows):
        for c in range(cols):
            g.set_change(r, c, 2.0 * r + 3.0 * c)
    grad = g.gradient_field()
    for r in range(rows):
        for c in range(cols):
            dh_dr, dh_dc = grad[r][c]
            assert dh_dr == pytest.approx(2.0, abs=1e-10)
            assert dh_dc == pytest.approx(3.0, abs=1e-10)


# ---------- flow_arrows count ----------

def test_flow_arrows_count():
    g = HeadChangeGrid(rows=3, cols=3)
    for r in range(3):
        for c in range(3):
            g.set_change(r, c, float(r * c))
    arrows = g.flow_arrows(max_arrow_cells=5)
    assert len(arrows) == 5

    # request more than available
    arrows_all = g.flow_arrows(max_arrow_cells=100)
    assert len(arrows_all) == 9


# ---------- flow_arrows unit direction vectors ----------

def test_flow_arrows_unit_directions():
    g = HeadChangeGrid(rows=4, cols=4)
    for r in range(4):
        for c in range(4):
            g.set_change(r, c, float((r - 1) * (c - 1)))
    arrows = g.flow_arrows(max_arrow_cells=10)
    for a in arrows:
        if a["magnitude"] > 1e-12:
            dx, dy = a["direction"]
            norm = math.sqrt(dx * dx + dy * dy)
            assert norm == pytest.approx(1.0, abs=1e-10)


# ---------- heatmap_data shape ----------

def test_heatmap_data_shape():
    g = HeadChangeGrid(rows=3, cols=5)
    g.set_change(0, 0, 1.5)
    data = g.heatmap_data()
    assert len(data) == 3
    assert all(len(row) == 5 for row in data)
    assert data[0][0] == pytest.approx(1.5)
    assert data[0][1] == 0.0


# ---------- annotation_text format ----------

def test_annotation_text_format():
    c = CellChange(row=0, col=0, delta_h=0.52, flow_direction=(1.0, 0.0))
    text = HeadChangeGrid(rows=1, cols=1).annotation_text(c)
    assert "Δh=" in text
    assert "+0.52" in text
    assert "(1.0,0.0)" in text

    # negative
    c2 = CellChange(row=0, col=0, delta_h=-0.33)
    text2 = HeadChangeGrid(rows=1, cols=1).annotation_text(c2)
    assert "Δh=" in text2
    assert "-0.33" in text2
    assert "→" not in text2


# ---------- save / load roundtrip ----------

def test_save_load_roundtrip():
    g = HeadChangeGrid(rows=3, cols=3)
    g.set_change(0, 0, 1.0, flow_dir=(1.0, 0.0))
    g.set_change(1, 1, -2.5)
    g.set_change(2, 2, 0.75, flow_dir=(0.0, -1.0))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        g.save_json(path)
        loaded = HeadChangeGrid.load_json(path)
        assert loaded.rows == 3
        assert loaded.cols == 3
        assert loaded.cell(0, 0).delta_h == pytest.approx(1.0)
        assert loaded.cell(0, 0).flow_direction == (1.0, 0.0)
        assert loaded.cell(1, 1).delta_h == pytest.approx(-2.5)
        assert loaded.cell(1, 1).flow_direction is None
        assert loaded.cell(2, 2).delta_h == pytest.approx(0.75)
        assert loaded.cell(2, 2).flow_direction == (0.0, -1.0)
        # unset cell
        assert loaded.cell(0, 1).delta_h == 0.0
    finally:
        os.unlink(path)


# ---------- compare_grids identical ----------

def test_compare_grids_identical():
    g1 = HeadChangeGrid(rows=3, cols=3)
    g2 = HeadChangeGrid(rows=3, cols=3)
    for r in range(3):
        for c in range(3):
            val = float(r * 10 + c)
            g1.set_change(r, c, val)
            g2.set_change(r, c, val)
    result = compare_grids(g1, g2)
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["max_cell_diff"] == pytest.approx(0.0)
    assert result["rms_diff"] == pytest.approx(0.0)


# ---------- compare_grids different ----------

def test_compare_grids_different():
    g1 = HeadChangeGrid(rows=2, cols=2)
    g2 = HeadChangeGrid(rows=2, cols=2)
    g1.set_change(0, 0, 1.0)
    g1.set_change(0, 1, 2.0)
    g1.set_change(1, 0, 3.0)
    g1.set_change(1, 1, 4.0)

    g2.set_change(0, 0, 2.0)
    g2.set_change(0, 1, 4.0)
    g2.set_change(1, 0, 3.0)
    g2.set_change(1, 1, 4.0)

    result = compare_grids(g1, g2)
    # diffs: -1, -2, 0, 0
    assert result["mean_diff"] == pytest.approx(-0.75)
    assert result["max_cell_diff"] == pytest.approx(2.0)
    assert result["rms_diff"] == pytest.approx(math.sqrt((1 + 4) / 4))

    # dimension mismatch
    g3 = HeadChangeGrid(rows=3, cols=3)
    with pytest.raises(ValueError):
        compare_grids(g1, g3)


# ---------- format_head_report ----------

def test_format_head_report_contains_text():
    g = HeadChangeGrid(rows=3, cols=3)
    g.set_change(0, 0, 5.0)
    g.set_change(1, 1, -2.0)
    g.set_change(2, 2, 1.0)
    report = format_head_report(g)
    assert "Head Change Report" in report
    assert "5x" in report or "3x3" in report
    assert "Max change" in report
    assert "Min change" in report
    assert "Mean change" in report
    assert "RMS change" in report
    assert "flow" in report.lower() or "arrow" in report.lower()

    # empty grid
    g_empty = HeadChangeGrid(rows=2, cols=2)
    report_empty = format_head_report(g_empty)
    assert "No changes" in report_empty


# ---------- 3x3 and 5x5 grids ----------

def test_3x3_grid():
    g = HeadChangeGrid(rows=3, cols=3)
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    idx = 0
    for r in range(3):
        for c in range(3):
            g.set_change(r, c, float(values[idx]))
            idx += 1
    assert g.max_change().delta_h == pytest.approx(9.0)
    assert g.min_change().delta_h == pytest.approx(1.0)
    assert g.mean_change() == pytest.approx(5.0)
    data = g.heatmap_data()
    assert data == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]


def test_5x5_grid():
    g = HeadChangeGrid(rows=5, cols=5)
    for r in range(5):
        for c in range(5):
            g.set_change(r, c, float(r * c))
    assert len(g.heatmap_data()) == 5
    assert len(g.heatmap_data()[0]) == 5
    grad = g.gradient_field()
    assert len(grad) == 5
    assert all(len(row) == 5 for row in grad)
    arrows = g.flow_arrows(max_arrow_cells=3)
    assert len(arrows) == 3
    # sorted by magnitude descending
    mags = [a["magnitude"] for a in arrows]
    assert mags == sorted(mags, reverse=True)
