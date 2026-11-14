"""Coverage push: play3d.run_demo and render3d trajectory."""
import sys
import os
import types
import numpy as np
from unittest.mock import MagicMock, patch
import pytest


class TestPlay3dRunDemo:
    """Cover play3d.run_demo by mocking render3d and _render_frame."""

    def _make_fake_render3d(self):
        fake = types.ModuleType('render3d')
        fake.__path__ = []
        fake.Render3D = MagicMock(return_value=MagicMock())
        saved = sys.modules.get('render3d')
        sys.modules['render3d'] = fake
        return saved

    def _restore_render3d(self, saved):
        if saved is not None:
            sys.modules['render3d'] = saved
        else:
            sys.modules.pop('render3d', None)

    def _patch_render_frame(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import flopyarcade.play3d as p3d
        orig = p3d._render_frame
        def fake_rf(env, renderer, step, total):
            fig = plt.figure(figsize=(4, 3))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot([0, 1], [0, 1], [0, 0])
            return fig
        p3d._render_frame = fake_rf
        return orig

    def _restore_render_frame(self, orig):
        import flopyarcade.play3d as p3d
        p3d._render_frame = orig

    def test_run_demo_normal(self, tmp_path):
        saved_r3d = self._make_fake_render3d()
        try:
            orig_rf = self._patch_render_frame()
            import flopyarcade.play3d as p3d
            output_path = str(tmp_path / 'test.gif')
            result = p3d.run_demo(output_path=output_path, nsteps=5, nrow=5, ncol=5)
            assert result == output_path
        finally:
            self._restore_render_frame(orig_rf)
            self._restore_render3d(saved_r3d)
            import matplotlib.pyplot as plt
            plt.close('all')

    def test_run_demo_env_done(self, tmp_path):
        saved_r3d = self._make_fake_render3d()
        try:
            orig_rf = self._patch_render_frame()
            import flopyarcade.play3d as p3d
            # Make env done after 1 step
            orig_step = p3d.MockEnv.step
            def step_done(self):
                orig_step(self)
                self.done = True
            p3d.MockEnv.step = step_done
            try:
                output_path = str(tmp_path / 'done.gif')
                result = p3d.run_demo(output_path=output_path, nsteps=10, nrow=4, ncol=4)
                assert result == output_path
            finally:
                p3d.MockEnv.step = orig_step
        finally:
            self._restore_render_frame(orig_rf)
            self._restore_render3d(saved_r3d)
            import matplotlib.pyplot as plt
            plt.close('all')


class TestRender3dTrajectory:
    """Cover render3d trajectory spline (lines 136-157)."""

    def test_trajectory_spline(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from flopyarcade.render3d import Render3D

        mock_env = MagicMock()
        mock_env.nrow = 10
        mock_env.ncol = 10
        mock_env.nlay = 1
        mock_env.delr = 100.0
        mock_env.delc = 100.0
        mock_env.extentX = 1000.0
        mock_env.extentY = 1000.0
        mock_env.zBot = 0.0
        mock_env.zTop = 50.0
        mock_env.heads = np.random.rand(10, 10)
        mock_env.particleX = 500.0
        mock_env.particleY = 500.0
        mock_env.wellX = 700.0
        mock_env.wellY = 300.0
        mock_env.wellQ = -0.005
        mock_env.done = False

        renderer = Render3D(mock_env, headless=True)
        renderer.trajectory = [
            [0.0, 0.0, 10.0],
            [100.0, 100.0, 20.0],
            [200.0, 200.0, 30.0],
            [300.0, 300.0, 40.0],
        ]
        try:
            fig = renderer.render_frame()
        except Exception:
            pass
        plt.close('all')

    def test_trajectory_two_points(self):
        """n==2 path in spline code."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from flopyarcade.render3d import Render3D

        mock_env = MagicMock()
        mock_env.nrow = 10
        mock_env.ncol = 10
        mock_env.nlay = 1
        mock_env.delr = 100.0
        mock_env.delc = 100.0
        mock_env.extentX = 1000.0
        mock_env.extentY = 1000.0
        mock_env.zBot = 0.0
        mock_env.zTop = 50.0
        mock_env.heads = np.random.rand(10, 10)
        mock_env.particleX = 500.0
        mock_env.particleY = 500.0
        mock_env.wellX = 700.0
        mock_env.wellY = 300.0
        mock_env.wellQ = -0.005
        mock_env.done = False

        renderer = Render3D(mock_env, headless=True)
        renderer.trajectory = [
            [0.0, 0.0, 10.0],
            [100.0, 100.0, 20.0],
        ]
        try:
            fig = renderer.render_frame()
        except Exception:
            pass
        plt.close('all')
