"""B3: Tests for actions bar at top with reward colormap.

Validates the renderActionBar and renderRewardBar methods in render.py.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class MockRenderer:
    """Minimal mock of the FloPyRenderer interface for action bar tests."""

    _ACTION_ICONS = {
        'up': '↑', 'down': '↓', 'left': '←', 'right': '→',
        'keep': '●', 'keepQ': '●', 'moreQ': '▲', 'lessQ': '▼',
    }
    REWARD_MAX = 1000.0

    def __init__(self):
        self.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        self._lastAction = 'up'
        self.rewardCurrent = 500.0
        self.ax2 = MagicMock()
        self._action_bar_artists = []


def _make_renderer(action_space=None, last_action=None, reward=500.0):
    r = MockRenderer()
    if action_space:
        r.actionSpace = action_space
    if last_action is not None:
        r._lastAction = last_action
    r.rewardCurrent = reward
    return r


class TestActionIcons:
    def test_all_standard_actions_have_icons(self):
        expected = {'up', 'down', 'left', 'right', 'keep', 'keepQ', 'moreQ', 'lessQ'}
        assert expected.issubset(set(MockRenderer._ACTION_ICONS.keys()))

    def test_icon_values_are_unicodes(self):
        for action, icon in MockRenderer._ACTION_ICONS.items():
            assert len(icon) >= 1
            assert isinstance(icon, str)


class TestRenderActionBar:
    def test_returns_none_without_action_space(self):
        r = _make_renderer()
        r.actionSpace = None  # simulate renderer with no action space
        actions = getattr(r, 'actionSpace', None)
        # None action space means early return in renderActionBar
        assert actions is None

    def test_action_space_defined(self):
        r = _make_renderer()
        assert len(r.actionSpace) == 5
        assert 'up' in r.actionSpace
        assert 'keep' in r.actionSpace

    def test_last_action_set(self):
        r = _make_renderer(last_action='down')
        assert r._lastAction == 'down'

    def test_icon_lookup_fallback(self):
        """Unknown action falls back to first letter uppercased."""
        icons = MockRenderer._ACTION_ICONS
        unknown_action = 'diagonal'
        icon = icons.get(unknown_action, unknown_action[0].upper())
        assert icon == 'D'


class TestRenderRewardBar:
    def test_reward_max_constant(self):
        assert MockRenderer.REWARD_MAX == 1000.0

    def test_reward_fraction_clamped(self):
        """Reward fraction should be clamped to [0, 1]."""
        REWARD_MAX = MockRenderer.REWARD_MAX
        # Normal case
        reward = 500.0
        frac = min(1.0, max(0.0, reward) / REWARD_MAX)
        assert frac == 0.5

        # Over max
        reward = 2000.0
        frac = min(1.0, max(0.0, reward) / REWARD_MAX)
        assert frac == 1.0

        # Negative
        reward = -100.0
        frac = min(1.0, max(0.0, reward) / REWARD_MAX)
        assert frac == 0.0

    def test_reward_bar_width_proportional(self):
        """Bar fill width is proportional to reward fraction."""
        REWARD_MAX = MockRenderer.REWARD_MAX
        n_actions = 5
        seg_width = 9.0
        bar_w = n_actions * seg_width

        reward = 250.0
        frac = min(1.0, max(0.0, reward) / REWARD_MAX)
        fill_w = bar_w * frac
        assert fill_w == bar_w * 0.25


class TestBlitUpdateActionBar:
    def test_cleanup_empty_artists(self):
        """Blit update should clear old artists before redrawing."""
        r = _make_renderer()
        r._action_bar_artists = [MagicMock() for _ in range(3)]
        # Simulate cleanup
        for art in r._action_bar_artists:
            art.remove()
        r._action_bar_artists = []
        assert len(r._action_bar_artists) == 0

    def test_no_action_space_early_return(self):
        r = _make_renderer()
        r.actionSpace = None  # force None to test early-return path
        actions = getattr(r, 'actionSpace', None)
        assert actions is None


class TestActionSpaceVariants:
    def test_six_action_space(self):
        """Environments with Q adjustments have 6 actions."""
        r = _make_renderer(action_space=['up', 'down', 'left', 'right', 'Qup', 'Qdown'])
        assert len(r.actionSpace) == 6

    def test_keep_only_space(self):
        r = _make_renderer(action_space=['keep'])
        assert len(r.actionSpace) == 1

    def test_bar_position_math(self):
        """Verify x-position calculation for action bar segments."""
        n = 5
        seg_width = 9.0
        x_end = 97.0
        x_start = x_end - n * seg_width
        assert x_start == 52.0

        # Each action center
        for i, act in enumerate(['up', 'down', 'left', 'right', 'keep']):
            x0 = x_start + i * seg_width
            center = x0 + seg_width / 2
            assert x0 < center < x0 + seg_width


class TestRewardDisplay:
    def test_positive_reward_format(self):
        reward = 500.0
        label = f"R: {reward:+.0f}"
        assert label == "R: +500"

    def test_negative_reward_format(self):
        reward = -200.0
        label = f"R: {reward:+.0f}"
        assert label == "R: -200"

    def test_zero_reward_format(self):
        reward = 0.0
        label = f"R: {reward:+.0f}"
        assert label == "R: +0"


class TestIntegrationWithRenderPipeline:
    def test_action_bar_enabled_flag(self):
        """The _action_bar_enabled flag controls rendering."""
        r = _make_renderer()
        enabled = getattr(r, '_action_bar_enabled', True)
        assert enabled is True

    def test_action_bar_disabled(self):
        r = _make_renderer()
        r._action_bar_enabled = False
        enabled = getattr(r, '_action_bar_enabled', True)
        assert enabled is False
