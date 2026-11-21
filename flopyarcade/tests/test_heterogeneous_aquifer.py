"""Tests for S2 heterogeneous aquifer environment."""

import math
import os
import tempfile

import pytest

from flopyarcade.heterogeneous_aquifer import (
    AquiferConfig,
    HeterogeneousAquifer,
    KLayer,
    LensGeometry,
    format_aquifer_report,
    generate_combined_config,
    generate_layered_config,
    generate_lens_config,
)


# ── Dataclass creation ────────────────────────────────────────────────────────

def test_klayer_creation():
    layer = KLayer(name="top", k_value=5.0, row_start=0, row_end=2)
    assert layer.name == "top"
    assert layer.k_value == 5.0
    assert layer.row_start == 0
    assert layer.row_end == 2


def test_lensgeometry_creation():
    lens = LensGeometry(center=(5, 5), radius=3, k_multiplier=10.0)
    assert lens.center == (5, 5)
    assert lens.radius == 3
    assert lens.k_multiplier == 10.0


def test_aquiferconfig_defaults():
    cfg = AquiferConfig()
    assert cfg.rows == 10
    assert cfg.cols == 10
    assert cfg.layers == []
    assert cfg.lens is None
    assert cfg.base_k == 1.0
    assert cfg.envtype == '8s-d'


# ── HeterogeneousAquifer init ─────────────────────────────────────────────────

def test_heterogeneous_aquifer_init():
    cfg = AquiferConfig(rows=5, cols=5, base_k=2.0)
    aq = HeterogeneousAquifer(cfg)
    assert aq.config.rows == 5
    assert aq.config.cols == 5
    assert aq.config.base_k == 2.0


# ── k_grid ────────────────────────────────────────────────────────────────────

def test_k_grid_shape():
    cfg = AquiferConfig(rows=4, cols=6)
    aq = HeterogeneousAquifer(cfg)
    grid = aq.k_grid()
    assert len(grid) == 4
    assert all(len(row) == 6 for row in grid)


def test_k_grid_uniform_no_heterogeneity():
    cfg = AquiferConfig(rows=5, cols=5, base_k=3.0)
    aq = HeterogeneousAquifer(cfg)
    grid = aq.k_grid()
    for row in grid:
        for v in row:
            assert v == 3.0


def test_k_grid_layered():
    cfg = AquiferConfig(
        rows=6, cols=4, base_k=1.0,
        layers=[
            KLayer(name="top", k_value=2.0, row_start=0, row_end=1),
            KLayer(name="mid", k_value=4.0, row_start=2, row_end=3),
            KLayer(name="bot", k_value=0.5, row_start=4, row_end=5),
        ],
    )
    aq = HeterogeneousAquifer(cfg)
    grid = aq.k_grid()
    assert all(v == 2.0 for v in grid[0])
    assert all(v == 2.0 for v in grid[1])
    assert all(v == 4.0 for v in grid[2])
    assert all(v == 4.0 for v in grid[3])
    assert all(v == 0.5 for v in grid[4])
    assert all(v == 0.5 for v in grid[5])


def test_k_grid_with_lens():
    cfg = AquiferConfig(
        rows=10, cols=10, base_k=1.0,
        lens=LensGeometry(center=(5, 5), radius=2, k_multiplier=10.0),
    )
    aq = HeterogeneousAquifer(cfg)
    # Center cell should be base_k * multiplier
    assert aq.k_at(5, 5) == 10.0
    # Cell outside lens should be base_k
    assert aq.k_at(0, 0) == 1.0


def test_k_at_specific_cell():
    cfg = AquiferConfig(
        rows=4, cols=4, base_k=1.0,
        layers=[KLayer(name="l0", k_value=7.0, row_start=0, row_end=1)],
    )
    aq = HeterogeneousAquifer(cfg)
    assert aq.k_at(0, 0) == 7.0
    assert aq.k_at(1, 3) == 7.0
    assert aq.k_at(2, 0) == 1.0
    assert aq.k_at(3, 3) == 1.0


# ── contrast_ratio ────────────────────────────────────────────────────────────

def test_contrast_ratio_heterogeneous():
    cfg = generate_layered_config(rows=9, cols=9)
    aq = HeterogeneousAquifer(cfg)
    assert aq.contrast_ratio() > 1.0


def test_contrast_ratio_homogeneous():
    cfg = AquiferConfig(rows=5, cols=5, base_k=2.0)
    aq = HeterogeneousAquifer(cfg)
    assert aq.contrast_ratio() == 1.0


# ── lens_cells / lens_fraction ────────────────────────────────────────────────

def test_lens_cells_within_radius():
    cfg = AquiferConfig(
        rows=10, cols=10, base_k=1.0,
        lens=LensGeometry(center=(5, 5), radius=2, k_multiplier=5.0),
    )
    aq = HeterogeneousAquifer(cfg)
    cells = aq.lens_cells()
    assert (5, 5) in cells
    # All cells within radius 2 of (5,5)
    for r, c in cells:
        assert math.sqrt((r - 5)**2 + (c - 5)**2) <= 2
    # (5,5) is center, (3,5) is dist 2, (7,7) is dist ~2.83 > 2
    assert (3, 5) in cells
    assert (7, 7) not in cells


def test_lens_fraction_in_range():
    cfg = generate_lens_config(rows=10, cols=10)
    aq = HeterogeneousAquifer(cfg)
    frac = aq.lens_fraction()
    assert 0 < frac < 1


# ── homogeneous_equivalent / effective_k ─────────────────────────────────────

def test_homogeneous_equivalent_no_het():
    cfg = AquiferConfig(rows=5, cols=5, base_k=4.0)
    aq = HeterogeneousAquifer(cfg)
    assert aq.homogeneous_equivalent() == 4.0


def test_effective_k_less_than_arithmetic():
    cfg = generate_layered_config(rows=9, cols=9)
    aq = HeterogeneousAquifer(cfg)
    harmonic = aq.effective_k()
    arithmetic = aq.homogeneous_equivalent()
    # For varied K values, harmonic mean < arithmetic mean
    assert harmonic < arithmetic


# ── transmissivity ────────────────────────────────────────────────────────────

def test_transmissivity():
    cfg = AquiferConfig(rows=3, cols=3, base_k=2.0)
    aq = HeterogeneousAquifer(cfg)
    t = aq.transmissivity(thickness=5.0)
    for row in t:
        for v in row:
            assert v == 10.0  # 2.0 * 5.0


# ── capture_zone_efficiency ──────────────────────────────────────────────────

def test_capture_zone_efficiency_structure():
    cfg = generate_layered_config(rows=9, cols=9)
    aq = HeterogeneousAquifer(cfg)
    result = aq.capture_zone_efficiency(
        capture_cells=[(1, 1), (1, 2), (2, 1)],
        source_pos=(0, 0),
    )
    assert 'n_capture_cells' in result
    assert 'mean_k_capture' in result
    assert 'mean_k_global' in result
    assert 'ratio' in result
    assert 'efficiency_estimate' in result
    assert result['n_capture_cells'] == 3
    assert result['efficiency_estimate'] == result['ratio'] ** 0.5


# ── flow_direction_estimate ──────────────────────────────────────────────────

def test_flow_direction_estimate():
    cfg = AquiferConfig(rows=5, cols=5, base_k=1.0)
    aq = HeterogeneousAquifer(cfg)
    path = aq.flow_direction_estimate(source=(0, 0), sink=(4, 4))
    assert len(path) > 0
    assert path[0]['row'] == 0
    assert path[0]['col'] == 0
    assert path[0]['step'] == 0
    # Last element should be at or near the sink
    last = path[-1]
    assert last['row'] == 4
    assert last['col'] == 4
    # Each entry has required keys
    for entry in path:
        assert 'row' in entry
        assert 'col' in entry
        assert 'k' in entry
        assert 'step' in entry


# ── compare_homogeneous ──────────────────────────────────────────────────────

def test_compare_homogeneous_structure():
    cfg = generate_combined_config(rows=9, cols=9)
    aq = HeterogeneousAquifer(cfg)
    result = aq.compare_homogeneous()
    assert 'het_mean_k' in result
    assert 'het_effective_k' in result
    assert 'homo_k' in result
    assert 'contrast' in result
    assert 'n_layers' in result
    assert 'has_lens' in result
    assert result['homo_k'] == 1.0
    assert result['n_layers'] == 3
    assert result['has_lens'] is True


# ── generate configs ─────────────────────────────────────────────────────────

def test_generate_layered_config():
    cfg = generate_layered_config(rows=9, cols=9, n_layers=3)
    assert len(cfg.layers) == 3
    assert cfg.layers[0].k_value == 1.0
    assert cfg.layers[1].k_value == 5.0
    assert cfg.layers[2].k_value == 0.2
    # Layers should cover all rows
    assert cfg.layers[0].row_start == 0
    assert cfg.layers[-1].row_end == 8


def test_generate_lens_config():
    cfg = generate_lens_config(rows=10, cols=10, lens_center=(5, 5), radius=3, k_mult=10.0)
    assert cfg.lens is not None
    assert cfg.lens.center == (5, 5)
    assert cfg.lens.radius == 3
    assert cfg.lens.k_multiplier == 10.0
    assert cfg.layers == []


# ── save/load roundtrip ──────────────────────────────────────────────────────

def test_save_load_roundtrip():
    cfg = generate_combined_config(rows=10, cols=10)
    aq = HeterogeneousAquifer(cfg)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    try:
        aq.save_config(path)
        loaded = HeterogeneousAquifer.load_config(path)
        assert loaded.config.rows == 10
        assert loaded.config.cols == 10
        assert len(loaded.config.layers) == 3
        assert loaded.config.lens is not None
        assert loaded.config.lens.center == (5, 5)
        assert loaded.config.base_k == 1.0
        assert loaded.config.envtype == '8s-d'
        # K grids should match
        grid1 = aq.k_grid()
        grid2 = loaded.k_grid()
        assert len(grid1) == len(grid2)
        for r in range(10):
            for c in range(10):
                assert abs(grid1[r][c] - grid2[r][c]) < 1e-12
    finally:
        os.unlink(path)


# ── format_aquifer_report ────────────────────────────────────────────────────

def test_format_aquifer_report():
    cfg = generate_combined_config(rows=10, cols=10)
    aq = HeterogeneousAquifer(cfg)
    report = format_aquifer_report(aq)
    assert '8s-d' in report
    assert 'n_layers' in report
    assert 'contrast' in report
    assert 'effective_k' in report
    assert 'lens' in report


# ── n_layers / has_lens ─────────────────────────────────────────────────────

def test_n_layers_and_has_lens():
    cfg = generate_layered_config(rows=9, cols=9)
    aq = HeterogeneousAquifer(cfg)
    assert aq.n_layers() == 3
    assert aq.has_lens() is False

    cfg2 = generate_lens_config()
    aq2 = HeterogeneousAquifer(cfg2)
    assert aq2.n_layers() == 0
    assert aq2.has_lens() is True
