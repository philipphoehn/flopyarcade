"""Tests for the Plotly dashboard module (ROADMAP V3)."""

import json
import os
import tempfile

import pytest

from flopyarcade.plotly_dashboard import (
    DashboardBuilder,
    DashboardConfig,
    FrameData,
    build_minimal_dashboard,
    format_dashboard_summary,
)


# ------------------------------------------------------------------
# DashboardConfig
# ------------------------------------------------------------------


def test_dashboard_config_creation():
    cfg = DashboardConfig(
        title="Test Dashboard",
        n_frames=3,
        grid_shape=(4, 4),
        well_positions=[(0, 0), (1, 1)],
    )
    assert cfg.title == "Test Dashboard"
    assert cfg.n_frames == 3
    assert cfg.grid_shape == (4, 4)
    assert cfg.well_positions == [(0, 0), (1, 1)]


def test_dashboard_config_defaults():
    cfg = DashboardConfig()
    assert cfg.title == "FloPyArcade Dashboard"
    assert cfg.n_frames == 5
    assert cfg.grid_shape == (5, 5)
    assert cfg.well_positions == []


# ------------------------------------------------------------------
# FrameData
# ------------------------------------------------------------------


def test_frame_data_creation():
    heads = [[1.0, 2.0], [3.0, 4.0]]
    fd = FrameData(t=0, heads=heads)
    assert fd.t == 0
    assert fd.heads == heads
    assert fd.particle_positions is None
    assert fd.active_wells is None


def test_frame_data_with_particles():
    heads = [[1.0]]
    particles = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    fd = FrameData(t=5, heads=heads, particle_positions=particles, active_wells=[0, 1])
    assert fd.t == 5
    assert fd.particle_positions == particles
    assert fd.active_wells == [0, 1]


# ------------------------------------------------------------------
# DashboardBuilder init & add_frame
# ------------------------------------------------------------------


def test_dashboard_builder_init():
    cfg = DashboardConfig(title="Init Test", grid_shape=(3, 3))
    builder = DashboardBuilder(cfg)
    assert builder.config is cfg
    assert builder.frames() == []


def test_add_frame_appends():
    cfg = DashboardConfig(grid_shape=(3, 3))
    builder = DashboardBuilder(cfg)
    heads = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    builder.add_frame(0, heads)
    builder.add_frame(1, heads, particle_positions=[[0, 0, 0]])
    assert len(builder.frames()) == 2
    assert builder.frames()[0].t == 0
    assert builder.frames()[1].t == 1
    assert builder.frames()[1].particle_positions == [[0, 0, 0]]


def test_frames_length():
    builder = build_minimal_dashboard()
    assert len(builder.frames()) == 5


# ------------------------------------------------------------------
# build_3d_scatter
# ------------------------------------------------------------------


def test_build_3d_scatter_structure():
    builder = build_minimal_dashboard()
    fig = builder.build_3d_scatter()
    assert "data" in fig
    assert "layout" in fig
    assert "frames" in fig
    assert "sliders" in fig["layout"]
    assert "updatemenus" in fig["layout"]
    assert "scene" in fig["layout"]
    assert "title" in fig["layout"]


def test_build_3d_scatter_has_scatter3d_trace():
    builder = build_minimal_dashboard()
    fig = builder.build_3d_scatter()
    assert len(fig["data"]) > 0
    trace = fig["data"][0]
    assert trace["type"] == "scatter3d"
    assert "x" in trace
    assert "y" in trace
    assert "z" in trace
    assert trace["mode"] == "markers"


# ------------------------------------------------------------------
# build_head_contour
# ------------------------------------------------------------------


def test_build_head_contour_structure():
    builder = build_minimal_dashboard()
    fig = builder.build_head_contour()
    assert "data" in fig
    assert "layout" in fig
    assert "sliders" in fig["layout"]
    trace = fig["data"][0]
    assert trace["type"] == "contour"
    assert "x" in trace
    assert "y" in trace
    assert "z" in trace


# ------------------------------------------------------------------
# build_well_selector
# ------------------------------------------------------------------


def test_build_well_selector_has_updatemenus():
    builder = build_minimal_dashboard()
    result = builder.build_well_selector()
    assert "updatemenus" in result
    assert len(result["updatemenus"]) > 0
    menu = result["updatemenus"][0]
    assert menu["type"] == "buttons"
    assert len(menu["buttons"]) >= 2  # "All wells" + at least one well


# ------------------------------------------------------------------
# build_combined
# ------------------------------------------------------------------


def test_build_combined_combines_traces():
    builder = build_minimal_dashboard()
    fig = builder.build_combined()
    assert "data" in fig
    assert "layout" in fig
    assert "frames" in fig
    # Should have both scatter3d and contour traces
    types = {t["type"] for t in fig["data"]}
    assert "scatter3d" in types
    assert "contour" in types
    # Should have updatemenus (well selector + play/pause)
    assert "updatemenus" in fig["layout"]


# ------------------------------------------------------------------
# save_html
# ------------------------------------------------------------------


def test_save_html_creates_file():
    builder = build_minimal_dashboard()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        path = tmp.name
    try:
        builder.save_html(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_save_html_contains_plotly_script_tag():
    builder = build_minimal_dashboard()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        path = tmp.name
    try:
        builder.save_html(path)
        with open(path, "r") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "cdn.plot.ly/plotly-latest.min.js" in content
        assert "Plotly.newPlot" in content
        assert "<div id='plot'></div>" in content
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ------------------------------------------------------------------
# to_json
# ------------------------------------------------------------------


def test_to_json_serializable():
    builder = build_minimal_dashboard()
    result = builder.to_json()
    # Must not raise
    serialized = json.dumps(result)
    assert len(serialized) > 0
    # Round-trip should work
    parsed = json.loads(serialized)
    assert "data" in parsed
    assert "layout" in parsed


# ------------------------------------------------------------------
# build_minimal_dashboard
# ------------------------------------------------------------------


def test_build_minimal_dashboard_has_5_frames():
    builder = build_minimal_dashboard()
    assert len(builder.frames()) == 5
    assert builder.config.grid_shape == (5, 5)
    assert len(builder.config.well_positions) == 3
    # Each frame should have particles and active wells
    for frame in builder.frames():
        assert frame.particle_positions is not None
        assert frame.active_wells is not None


# ------------------------------------------------------------------
# format_dashboard_summary
# ------------------------------------------------------------------


def test_format_dashboard_summary_contains_text():
    builder = build_minimal_dashboard()
    summary = format_dashboard_summary(builder)
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "FloPyArcade" in summary
    assert "5x5" in summary
    assert "5" in summary  # frame count
    assert "Wells" in summary


# ------------------------------------------------------------------
# 3x3 grid edge case
# ------------------------------------------------------------------


def test_3x3_grid_works():
    cfg = DashboardConfig(
        title="Small Grid",
        n_frames=2,
        grid_shape=(3, 3),
        well_positions=[(1, 1)],
    )
    builder = DashboardBuilder(cfg)
    heads = [[float(i * 3 + j) for j in range(3)] for i in range(3)]
    builder.add_frame(0, heads, particle_positions=[[1, 1, 5.0]])
    builder.add_frame(1, heads, particle_positions=[[2, 2, 4.0]])

    # All builders should work
    scatter = builder.build_3d_scatter()
    assert len(scatter["data"]) == 2

    contour = builder.build_head_contour()
    assert len(contour["data"]) == 2

    well_sel = builder.build_well_selector()
    assert len(well_sel["updatemenus"][0]["buttons"]) == 2  # All + Well 0

    combined = builder.build_combined()
    types = {t["type"] for t in combined["data"]}
    assert "scatter3d" in types
    assert "contour" in types

    json.dumps(builder.to_json())  # serializable

    summary = format_dashboard_summary(builder)
    assert "3x3" in summary
