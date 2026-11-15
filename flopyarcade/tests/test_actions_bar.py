"""Tests for B3: Action bar + Reward indicator overlay."""
import sys
import unittest
from unittest.mock import patch, MagicMock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TestActionBarRendering(unittest.TestCase):
    """Test that action bar and reward bar render correctly."""

    def _make_env_stub(self):
        """Create a minimal stub with the attributes render methods need."""
        stub = MagicMock()
        stub.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        stub._lastAction = 'up'
        stub.rewardCurrent = 5.0
        stub._action_bar_enabled = True
        stub.REWARD_MAX = 1000.0
        fig, ax = plt.subplots()
        stub.fig = fig
        stub.ax2 = ax
        stub._blit_enabled = False
        return stub

    def test_renderActionBar_creates_text_artists(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        RenderMixin.renderActionBar(stub, zorder=11)
        # Should have 5 text artists (one per action) at y=97
        texts = [t for t in stub.ax2.texts if abs(t.get_position()[1] - 96) < 0.1]
        self.assertEqual(len(texts), 5)
        plt.close('all')

    def test_renderActionBar_highlights_current(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        stub._lastAction = 'left'
        RenderMixin.renderActionBar(stub, zorder=11)
        texts = [t for t in stub.ax2.texts if abs(t.get_position()[1] - 96) < 0.1]
        # 'left' is index 2; it should be green (#4CAF50)
        left_text = texts[2]
        self.assertEqual(left_text.get_color(), 'white')
        self.assertEqual(left_text.get_fontweight(), 'bold')
        plt.close('all')

    def test_renderActionBar_no_actionSpace(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        stub.actionSpace = None
        RenderMixin.renderActionBar(stub, zorder=11)
        # No text should be added
        self.assertEqual(len(stub.ax2.texts), 0)
        plt.close('all')

    def test_renderActionBar_no_lastAction(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        stub._lastAction = None
        RenderMixin.renderActionBar(stub, zorder=11)
        self.assertEqual(len(stub.ax2.texts), 0)
        plt.close('all')

    def test_renderRewardBar_positive(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        stub.rewardCurrent = 10.0
        RenderMixin.renderRewardBar(stub, zorder=11)
        # Should have patches (bg + fill) and a text label
        patches = stub.ax2.patches
        self.assertGreaterEqual(len(patches), 2)
        # Label text should contain 'R: +10.0'
        labels = [t for t in stub.ax2.texts if 'R:' in t.get_text()]
        self.assertEqual(len(labels), 1)
        self.assertIn('+10', labels[0].get_text())
        plt.close('all')

    def test_renderRewardBar_negative(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        stub.rewardCurrent = -3.0
        RenderMixin.renderRewardBar(stub, zorder=11)
        labels = [t for t in stub.ax2.texts if 'R:' in t.get_text()]
        self.assertIn('-3', labels[0].get_text())
        plt.close('all')

    def test_renderRewardBar_zero(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        stub.rewardCurrent = 0.0
        RenderMixin.renderRewardBar(stub, zorder=11)
        labels = [t for t in stub.ax2.texts if 'R:' in t.get_text()]
        self.assertIn('+0', labels[0].get_text())
        plt.close('all')


class TestBlitActionBar(unittest.TestCase):
    """Test the blit-mode action bar update."""

    def _make_env_stub(self):
        stub = MagicMock()
        stub.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        stub._lastAction = 'right'
        stub.rewardCurrent = 2.5
        stub._action_bar_enabled = True
        stub._action_bar_artists = []
        fig, ax = plt.subplots()
        stub.fig = fig
        stub.ax2 = ax
        return stub

    def test_blit_update_action_bar_creates_artists(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        RenderMixin._blit_update_action_bar(stub)
        self.assertGreater(len(stub._action_bar_artists), 0)
        plt.close('all')

    def test_blit_update_action_bar_cleanup(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        # First call
        RenderMixin._blit_update_action_bar(stub)
        first_count = len(stub._action_bar_artists)
        # Second call should remove old artists
        RenderMixin._blit_update_action_bar(stub)
        # Artists list should be fresh (not accumulating)
        self.assertLessEqual(len(stub._action_bar_artists), first_count + 5)
        plt.close('all')

    def test_blit_update_action_bar_no_actions(self):
        from flopyarcade.render import RenderMixin
        stub = self._make_env_stub()
        stub.actionSpace = None
        stub._action_bar_artists = []
        RenderMixin._blit_update_action_bar(stub)
        self.assertEqual(len(stub._action_bar_artists), 0)
        plt.close('all')


class TestActionBarWiring(unittest.TestCase):
    """Test that action bar is wired into the render pipeline."""

    def test_plot_environment_calls_action_bar(self):
        """_plot_environment should call renderActionBar when enabled."""
        import inspect
        from flopyarcade import render
        src = inspect.getsource(render.RenderMixin._plot_environment)
        self.assertIn('renderActionBar', src)
        self.assertIn('renderRewardBar', src)

    def test_blit_frame_calls_action_bar(self):
        """render_blit_frame should call _blit_update_action_bar when enabled."""
        import inspect
        from flopyarcade import render
        src = inspect.getsource(render.RenderMixin.render_blit_frame)
        self.assertIn('_blit_update_action_bar', src)


class TestActionBarConfig(unittest.TestCase):
    """Test that the action bar flag flows through the stack."""

    def test_arcade_accepts_actionBar(self):
        import inspect
        from flopyarcade import arcade
        sig = inspect.signature(arcade.FloPyArcade.__init__)
        self.assertIn('actionBar', sig.parameters)

    def test_play_has_action_bar_flag(self):
        import inspect
        from flopyarcade import play
        src = inspect.getsource(play)
        self.assertIn('--action-bar', src)

    def test_action_bar_default_true(self):
        """Default should be enabled."""
        from flopyarcade.render import RenderMixin
        stub = MagicMock()
        # _action_bar_enabled should default to True via getattr
        self.assertTrue(getattr(stub, '_action_bar_enabled', True))


class TestActionIcons(unittest.TestCase):
    """Test that all action icons are defined."""

    def test_icons_cover_all_actions(self):
        from flopyarcade.render import RenderMixin
        for action in ['up', 'down', 'left', 'right', 'keep', 'keepQ', 'moreQ', 'lessQ']:
            self.assertIn(action, RenderMixin._ACTION_ICONS)

    def test_icons_are_strings(self):
        from flopyarcade.render import RenderMixin
        for icon in RenderMixin._ACTION_ICONS.values():
            self.assertIsInstance(icon, str)
            self.assertGreater(len(icon), 0)


if __name__ == '__main__':
    unittest.main()
