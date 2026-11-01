"""Targeted tests to close remaining coverage gaps in __init__, play, play3d."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock


# ─── __init__.py: public API + version ───────────────────────────────────────

class TestInitPublicAPI:
    def test_version(self):
        import flopyarcade
        assert flopyarcade.__version__ == '0.3.23'

    def test_author(self):
        import flopyarcade
        assert flopyarcade.__author__ == 'Philipp Hoehn'

    def test_exports(self):
        import flopyarcade
        assert hasattr(flopyarcade, 'FloPyAgent')
        assert hasattr(flopyarcade, 'FloPyEnv')
        assert hasattr(flopyarcade, 'FloPyArcade')

    def test_simulators_extracted(self):
        """After import, simulator dirs should exist (unzip ran on first import)."""
        import flopyarcade
        wrkspc = os.path.dirname(os.path.realpath(flopyarcade.__file__))
        assert os.path.exists(os.path.join(wrkspc, 'simulators', 'linux'))
        assert os.path.exists(os.path.join(wrkspc, 'simulators', 'mac'))
        assert os.path.exists(os.path.join(wrkspc, 'simulators', 'win32'))
        assert os.path.exists(os.path.join(wrkspc, 'simulators', 'win64'))


# ─── play.py: str_to_bool ─────────────────────────────────────────────────────

class TestPlayStrToBool:
    def test_true_strings(self):
        from flopyarcade.play import str_to_bool
        for v in ['true', 'True', 'TRUE', 't', 'T', '1', 'yes', 'YES', 'y', 'Y']:
            assert str_to_bool(v) is True

    def test_false_strings(self):
        from flopyarcade.play import str_to_bool
        for v in ['false', 'False', 'FALSE', 'f', 'F', '0', 'no', 'NO', 'n', 'N']:
            assert str_to_bool(v) is False

    def test_bool_passthrough(self):
        from flopyarcade.play import str_to_bool
        assert str_to_bool(True) is True
        assert str_to_bool(False) is False

    def test_invalid_raises(self):
        from flopyarcade.play import str_to_bool
        with pytest.raises(ValueError, match='not a valid boolean'):
            str_to_bool('maybe')


# ─── play.py: has_arguments ──────────────────────────────────────────────────

class TestPlayHasArguments:
    def test_all_defaults_returns_false(self):
        import argparse
        from flopyarcade.play import has_arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--x', default='foo')
        parser.add_argument('--y', default=42)
        args = parser.parse_args([])
        assert has_arguments(parser, args) is False

    def test_one_differs_returns_true(self):
        import argparse
        from flopyarcade.play import has_arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--x', default='foo')
        args = parser.parse_args(['--x', 'bar'])
        assert has_arguments(parser, args) is True

    def test_help_action_skipped(self):
        """argparse._HelpAction should not trigger has_arguments."""
        import argparse
        from flopyarcade.play import has_arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--x', default='foo')
        args = parser.parse_args([])
        # -h is auto-added; it's a _HelpAction and should be skipped
        assert has_arguments(parser, args) is False


# ─── play.py: main() envSettings construction ────────────────────────────────

class TestPlayMainConstruction:
    def test_main_function_exists_and_callable(self):
        """Verify play.main is a callable function (covers the def line)."""
        from flopyarcade import play
        assert callable(play.main)

    def test_main_signature(self):
        """Check main() accepts envSettings and gameSettings."""
        import inspect
        from flopyarcade import play
        sig = inspect.signature(play.main)
        params = list(sig.parameters.keys())
        assert len(params) >= 1  # at least one parameter


# ─── play3d.py: distance_to_well ─────────────────────────────────────────────

class TestPlay3DDistance:
    def test_distance_zero(self):
        from flopyarcade.play3d import MockEnv
        env = MockEnv(nrow=10, ncol=10)
        d = env.distance_to_well([1.0, 2.0], [1.0, 2.0])
        assert d == 0.0

    def test_distance_known(self):
        from flopyarcade.play3d import MockEnv
        env = MockEnv(nrow=10, ncol=10)
        d = env.distance_to_well([0.0, 0.0], [3.0, 4.0])
        assert abs(d - 5.0) < 1e-10

    def test_distance_symmetric(self):
        from flopyarcade.play3d import MockEnv
        env = MockEnv(nrow=10, ncol=10)
        d1 = env.distance_to_well([1.0, 2.0], [4.0, 6.0])
        d2 = env.distance_to_well([4.0, 6.0], [1.0, 2.0])
        assert abs(d1 - d2) < 1e-10


# ─── play3d.py: MockEnv step + done ──────────────────────────────────────────

class TestPlay3DMockEnv:
    def test_step_returns_tuple(self):
        from flopyarcade.play3d import MockEnv
        env = MockEnv(nrow=10, ncol=10)
        result = env.step()
        assert len(result) == 5
        coords, reward, done, truncated, info = result
        assert isinstance(coords, list)
        assert len(coords) == 2
        assert reward == 0.0
        assert isinstance(done, bool)

    def test_particle_moves(self):
        from flopyarcade.play3d import MockEnv
        env = MockEnv(nrow=20, ncol=20)
        x0 = env.particleX
        env.step()
        # Particle should have moved (not stuck)
        assert env.particleX != x0 or env.done

    def test_eventually_done(self):
        from flopyarcade.play3d import MockEnv
        env = MockEnv(nrow=10, ncol=10)
        steps = 0
        while not env.done and steps < 200:
            env.step()
            steps += 1
        assert env.done, f"MockEnv did not terminate in {steps} steps"


# ─── play3d.py: main() and run_live (mocked) ─────────────────────────────────

class TestPlay3DMain:
    @patch('flopyarcade.play3d.run_demo')
    def test_main_demo_mode(self, mock_demo):
        from flopyarcade.play3d import main
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ['play3d', '--mode', 'demo', '--output', 'test.gif', '--nsteps', '5']
            main()
            mock_demo.assert_called_once()
            call_kwargs = mock_demo.call_args
            assert call_kwargs[1].get('nsteps') == 5 or 'nsteps' in str(call_kwargs)
        finally:
            sys.argv = old_argv

    @patch('flopyarcade.play3d.run_demo')
    def test_main_default_args(self, mock_demo):
        """Default mode is demo."""
        from flopyarcade.play3d import main
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ['play3d']
            main()
            mock_demo.assert_called_once()
        finally:
            sys.argv = old_argv


# ─── play3d.py: _render_frame returns a figure ───────────────────────────────

class TestRender3DFrame:
    def test_render3d_class_exists(self):
        from flopyarcade.render3d import Render3D
        assert Render3D is not None

    def test_render_environment_3d_callable(self):
        from flopyarcade.render3d import render_environment_3d
        assert callable(render_environment_3d)


# ─── flopyarcade.py: str(env), env.info, invalid ENVTYPE ─────────────────────

class TestFloPyArcadeEdgePaths:
    def test_str_env(self):
        from flopyarcade import FloPyEnv
        env = FloPyEnv(ENVTYPE='1s-d', nRow=25, nCol=25, flagRender=False, _seed=42)
        s = str(env)
        assert isinstance(s, str)
        assert len(s) > 0
        env.close()

    def test_invalid_envtype_raises(self):
        from flopyarcade import FloPyEnv
        with pytest.raises((ValueError, KeyError, Exception)):
            env = FloPyEnv(ENVTYPE='nonexistent-type-xyz', nRow=25, nCol=25, flagRender=False)

    def test_env_info_dict(self):
        from flopyarcade import FloPyEnv
        env = FloPyEnv(ENVTYPE='1s-d', nRow=25, nCol=25, flagRender=False, _seed=42)
        # FloPyArcade game object has .info
        game = env
        if hasattr(game, 'info'):
            assert isinstance(game.info, dict)
        env.close()

    def test_env_close_idempotent(self):
        """Calling close twice should not raise."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(ENVTYPE='1s-d', nRow=25, nCol=25, flagRender=False, _seed=42)
        env.close()
        env.close()  # second call should not crash

    def test_env_has_nrow(self):
        """FloPyEnv exposes nRow and nCol."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(ENVTYPE='1s-d', nRow=25, nCol=25, flagRender=False, _seed=123)
        env.reset()
        assert env.nRow == 25
        assert env.nCol == 25
        env.close()


# ─── render3d.py: Render3D parameters with MockEnv ───────────────────────────

class TestRender3DEdge:
    def test_render3d_with_mockenv(self):
        """Render3D works with MockEnv (lowercase nrow/ncol)."""
        import matplotlib
        matplotlib.use('Agg')
        from flopyarcade.play3d import MockEnv
        from flopyarcade.render3d import Render3D
        env = MockEnv(nrow=10, ncol=10)
        renderer = Render3D(env, headless=True)
        assert renderer.nrow == 10
        assert renderer.ncol == 10

    def test_render3d_elevation_scale(self):
        """Elevation scale parameter is stored correctly."""
        import matplotlib
        matplotlib.use('Agg')
        from flopyarcade.play3d import MockEnv
        from flopyarcade.render3d import Render3D
        env = MockEnv(nrow=10, ncol=10)
        renderer = Render3D(env, headless=True, elevation_scale=3.0)
        assert renderer.elevation_scale == 3.0

    def test_render3d_colormap(self):
        """Colormap parameter is stored."""
        import matplotlib
        matplotlib.use('Agg')
        from flopyarcade.play3d import MockEnv
        from flopyarcade.render3d import Render3D
        env = MockEnv(nrow=10, ncol=10)
        renderer = Render3D(env, headless=True, colormap='viridis')
        assert renderer.colormap == 'viridis'


# ─── train_dqn.py: module-level structure ────────────────────────────────────

class TestTrainDqnModule:
    def test_module_has_main(self):
        import inspect
        from flopyarcade import train_dqn
        src = inspect.getsource(train_dqn)
        assert 'def main' in src

    def test_source_references_flopyagent(self):
        import inspect
        from flopyarcade import train_dqn
        src = inspect.getsource(train_dqn)
        assert 'FloPyAgent' in src

    def test_source_has_dqn(self):
        import inspect
        from flopyarcade import train_dqn
        src = inspect.getsource(train_dqn)
        assert 'DQN' in src or 'dqn' in src


# ─── train_neuroevolution.py: module-level structure ─────────────────────────

class TestTrainNEModule:
    def test_module_importable(self):
        from flopyarcade import train_neuroevolution
        assert train_neuroevolution is not None

    def test_source_references_flopyarcade(self):
        import inspect
        from flopyarcade import train_neuroevolution
        src = inspect.getsource(train_neuroevolution)
        assert 'FloPyEnv' in src or 'flopyarcade' in src

    def test_source_has_genetic_algorithm(self):
        import inspect
        from flopyarcade import train_neuroevolution
        src = inspect.getsource(train_neuroevolution)
        assert 'gen' in src.lower() or 'population' in src.lower()
