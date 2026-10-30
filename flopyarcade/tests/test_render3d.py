"""Tests for the FloPyArcade 3D visualization module.

Uses direct module imports to avoid triggering flopyarcade/__init__.py
(which pulls in tensorflow and other heavy dependencies).
"""

import os
import sys
import numpy as np
import pytest

# Add the flopyarcade package directory to path so we can import
# render3d and play3d as top-level modules
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_DIR)

# Import directly (bypasses flopyarcade/__init__.py)
import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

render3d = _import_module('render3d', os.path.join(_PKG_DIR, 'render3d.py'))
play3d = _import_module('play3d', os.path.join(_PKG_DIR, 'play3d.py'))

MockEnv = play3d.MockEnv
Render3D = render3d.Render3D
render_environment_3d = render3d.render_environment_3d


class TestMockEnv:
    """Test the synthetic environment used for 3D demos."""

    def test_init_shape(self):
        env = MockEnv(nrow=20, ncol=20)
        assert env.heads.shape == (20, 20)
        assert env.nrow == 20
        assert env.ncol == 20
        assert env.extentX == 20 * 100.0
        assert env.extentY == 20 * 100.0

    def test_head_range(self):
        env = MockEnv(nrow=25, ncol=25)
        assert env.heads.min() > 50
        assert env.heads.max() < 65

    def test_particle_starts_west(self):
        env = MockEnv()
        assert env.particleCoords[0] < env.extentX * 0.1

    def test_well_in_domain(self):
        env = MockEnv()
        wx, wy = env.wellCoords[0], env.wellCoords[1]
        assert 0 < wx < env.extentX
        assert 0 < wy < env.extentY

    def test_step_moves_particle(self):
        env = MockEnv()
        x0 = env.particleX
        env.step()
        assert env.particleX > x0

    def test_step_completes(self):
        env = MockEnv(nrow=10, ncol=10)
        steps = 0
        while not env.done and steps < 200:
            env.step()
            steps += 1
        assert env.done
        assert steps < 200

    def test_reset(self):
        env = MockEnv()
        env.step()
        env.step()
        env.reset()
        assert env.particleX < env.extentX * 0.1
        assert not env.done


class TestRender3D:
    """Test the 3D renderer with synthetic data."""

    def test_build_grid(self):
        env = MockEnv(nrow=10, ncol=10)
        renderer = Render3D(env, headless=True)
        result = renderer._build_grid()
        assert isinstance(result, tuple)
        X, Y, Z, head_nodes = result
        assert X.shape == (10 + 1, 10 + 1)
        assert Z.shape == (10 + 1, 10 + 1)

    def test_particle_position(self):
        env = MockEnv()
        renderer = Render3D(env, headless=True)
        ppos = renderer._get_particle_position()
        assert len(ppos) == 3
        assert ppos[0] >= 0
        assert ppos[1] >= 0
        assert ppos[2] > 0

    def test_well_position(self):
        env = MockEnv()
        renderer = Render3D(env, headless=True)
        wpos = renderer._get_well_position()
        assert len(wpos) == 3
        assert wpos[2] > 0

    def test_render_to_file(self, tmp_path):
        env = MockEnv(nrow=10, ncol=10)
        renderer = Render3D(env, headless=True)
        out = str(tmp_path / "test_frame.png")
        result = renderer.render(save_path=out)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000

    def test_render_convenience_function(self, tmp_path):
        env = MockEnv(nrow=10, ncol=10)
        out = str(tmp_path / "test_frame2.png")
        result = render_environment_3d(env, output_path=out)
        assert result == out
        assert os.path.exists(out)

    def test_trajectory_recording(self):
        env = MockEnv(nrow=10, ncol=10)
        renderer = Render3D(env, headless=True)
        assert len(renderer.trajectory) == 0
        env.step()
        renderer.record_trajectory()
        env.step()
        renderer.record_trajectory()
        assert len(renderer.trajectory) == 2

    def test_elevation_scale(self):
        env = MockEnv(nrow=10, ncol=10)
        r1 = Render3D(env, headless=True, elevation_scale=1.0)
        r2 = Render3D(env, headless=True, elevation_scale=3.0)
        p1 = r1._get_particle_position()
        p2 = r2._get_particle_position()
        assert abs(p2[2] / p1[2] - 3.0) < 0.01


class TestAnimation:
    """Test animation generation."""

    def test_demo_small(self, tmp_path):
        out = str(tmp_path / "anim_test.gif")
        result = play3d.run_demo(output_path=out, nsteps=5, nrow=8, ncol=8)
        assert result == out
        assert os.path.exists(out)
        size = os.path.getsize(out)
        assert size > 5000, f"GIF too small: {size} bytes"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
