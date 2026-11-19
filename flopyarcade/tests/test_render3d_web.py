#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade - Browser 3D Viewer tests
# V5: Self-contained HTML+Three.js 3D visualisation

"""Tests for flopyarcade.render3d_web module."""

import json
import os
import tempfile

import numpy as np
import pytest

from flopyarcade.render3d_web import (
    Render3DWeb,
    _terrain_colormap,
    make_viewer,
)


# --- Mock environment ---

class MockEnv:
    """Minimal env duck-type for testing."""
    def __init__(self, nrow=5, ncol=7, seed=42):
        rng = np.random.default_rng(seed)
        self.nrow = nrow
        self.ncol = ncol
        self.extentX = 500.0
        self.extentY = 400.0
        self.zTop = 40.0
        self.zBot = 0.0
        self.heads = 20 + rng.normal(0, 3, (nrow, ncol))


# --- Colormap ---

class TestColormap:
    def test_length(self):
        cmap = _terrain_colormap(256)
        assert len(cmap) == 256

    def test_rgb_range(self):
        cmap = _terrain_colormap(64)
        for r, g, b in cmap:
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0

    def test_monotonic_brightness(self):
        """Higher values should generally be brighter (sum increases)."""
        cmap = _terrain_colormap(128)
        brightness = [sum(c) for c in cmap]
        # Allow small non-monotonicity at boundaries
        increasing = sum(1 for i in range(len(brightness) - 1)
                         if brightness[i + 1] >= brightness[i] - 0.05)
        assert increasing > len(brightness) * 0.8

    def test_starts_dark_ends_bright(self):
        cmap = _terrain_colormap(256)
        assert sum(cmap[0]) < sum(cmap[-1])


# --- Render3DWeb init ---

class TestInit:
    def test_basic(self):
        env = MockEnv()
        v = Render3DWeb(env)
        assert v.nrow == 5
        assert v.ncol == 7
        assert v.extentX == 500.0
        assert v.extentY == 400.0
        assert v.zTop == 40.0
        assert v.zBot == 0.0
        assert v.headMin <= v.headMax

    def test_trajectory_with_y(self):
        env = MockEnv()
        traj = [{'x': 100, 'z': 200, 'y': 25.0}]
        v = Render3DWeb(env, trajectory=traj)
        assert v.trajectory[0]['y'] == 25.0

    def test_trajectory_without_y(self):
        env = MockEnv()
        traj = [{'x': 100, 'z': 200}]
        v = Render3DWeb(env, trajectory=traj)
        assert 0.0 <= v.trajectory[0]['y'] <= 40.0

    def test_trajectory_empty(self):
        env = MockEnv()
        v = Render3DWeb(env, trajectory=None)
        assert v.trajectory == []

    def test_wells_normalised(self):
        env = MockEnv()
        wells = [{'x': 100, 'z': 200, 'q': 5.0, 'name': 'P1'}]
        v = Render3DWeb(env, wells=wells)
        assert v.wells[0]['x'] == 100.0
        assert v.wells[0]['q'] == 5.0
        assert v.wells[0]['name'] == 'P1'

    def test_wells_empty(self):
        env = MockEnv()
        v = Render3DWeb(env, wells=None)
        assert v.wells == []

    def test_custom_z(self):
        env = MockEnv()
        v = Render3DWeb(env, z_top=100.0, z_bot=-10.0)
        assert v.zTop == 100.0
        assert v.zBot == -10.0

    def test_head_shape_mismatch_resized(self):
        """If heads shape doesn't match nrow/ncol, scipy zoom is used."""
        env = MockEnv(nrow=5, ncol=7)
        # Override heads with different shape
        env.heads = np.random.default_rng(0).normal(20, 3, (3, 4))
        v = Render3DWeb(env)  # should zoom to (5,7)
        assert v.heads.shape == (5, 7)


# --- to_dict ---

class TestToDict:
    def test_keys(self):
        env = MockEnv()
        v = Render3DWeb(env)
        d = v.to_dict()
        expected_keys = {
            'nrow', 'ncol', 'extentX', 'extentY',
            'zTop', 'zBot', 'zScale',
            'heads', 'headMin', 'headMax',
            'trajectory', 'wells', 'colormap',
        }
        assert expected_keys.issubset(d.keys())

    def test_heads_shape(self):
        env = MockEnv(nrow=4, ncol=6)
        v = Render3DWeb(env)
        d = v.to_dict()
        assert len(d['heads']) == 4
        assert len(d['heads'][0]) == 6

    def test_colormap_length(self):
        env = MockEnv()
        v = Render3DWeb(env)
        d = v.to_dict()
        assert len(d['colormap']) == 128

    def test_json_serialisable(self):
        env = MockEnv()
        v = Render3DWeb(env, trajectory=[{'x': 1, 'z': 2}],
                        wells=[{'x': 10, 'z': 20, 'q': 1}])
        d = v.to_dict()
        s = json.dumps(d)
        assert len(s) > 100
        parsed = json.loads(s)
        assert parsed['nrow'] == 5


# --- to_html ---

class TestToHtml:
    def test_contains_doctype(self):
        env = MockEnv()
        v = Render3DWeb(env)
        html = v.to_html()
        assert html.startswith('<!DOCTYPE html>')

    def test_contains_threejs_import(self):
        env = MockEnv()
        v = Render3DWeb(env)
        html = v.to_html()
        assert 'three@0.164.1' in html
        assert 'OrbitControls' in html

    def test_contains_data(self):
        env = MockEnv()
        v = Render3DWeb(env)
        html = v.to_html()
        assert 'nrow' in html
        assert 'headMin' in html
        assert 'colormap' in html

    def test_no_placeholder_left(self):
        env = MockEnv()
        v = Render3DWeb(env)
        html = v.to_html()
        assert '__DATA_JSON__' not in html

    def test_contains_cyberpunk_colors(self):
        env = MockEnv()
        v = Render3DWeb(env)
        html = v.to_html()
        assert '0x00ffcc' in html or '00ffcc' in html


# --- save ---

class TestSave:
    def test_creates_file(self):
        env = MockEnv()
        v = Render3DWeb(env)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'viewer.html')
            result = v.save(path)
            assert os.path.exists(path)
            assert str(result) == os.path.join(tmp, 'viewer.html')
            size = os.path.getsize(path)
            assert size > 1000

    def test_creates_parent_dirs(self):
        env = MockEnv()
        v = Render3DWeb(env)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'a', 'b', 'c', 'viewer.html')
            v.save(path)
            assert os.path.exists(path)

    def test_file_is_valid_html(self):
        env = MockEnv()
        v = Render3DWeb(env)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'v.html')
            v.save(path)
            with open(path) as f:
                content = f.read()
            assert content.strip().startswith('<!DOCTYPE html>')
            assert '</html>' in content


# --- make_viewer ---

class TestMakeViewer:
    def test_returns_path(self):
        env = MockEnv()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'out.html')
            result = make_viewer(env, out_path=path)
            assert os.path.exists(path)

    def test_with_trajectory_and_wells(self):
        env = MockEnv()
        traj = [{'x': 50, 'z': 100}, {'x': 100, 'z': 200}]
        wells = [{'x': 250, 'z': 350, 'q': -2.0, 'name': 'W1'}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'full.html')
            make_viewer(env, trajectory=traj, wells=wells, out_path=path)
            with open(path) as f:
                content = f.read()
            assert 'trajectory' in content
            assert 'wells' in content


# --- Edge cases ---

class TestEdgeCases:
    def test_single_cell(self):
        env = MockEnv(nrow=1, ncol=1)
        v = Render3DWeb(env)
        d = v.to_dict()
        assert len(d['heads']) == 1
        assert len(d['heads'][0]) == 1

    def test_uniform_heads(self):
        env = MockEnv()
        env.heads = np.full((5, 7), 25.0)
        v = Render3DWeb(env)
        assert v.headMin == 25.0
        assert v.headMax == 25.0
        html = v.to_html()
        assert '__DATA_JSON__' not in html

    def test_negative_heads(self):
        env = MockEnv()
        env.heads = np.full((5, 7), -10.0)
        v = Render3DWeb(env)
        assert v.headMin == -10.0
        assert v.headMax == -10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
