#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Tests for B2 split-screen P2 modes (human / replay / policy)."""

import json
import os
import sys
import tempfile
import unittest

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from logging import getLogger, FATAL
getLogger('tensorflow').setLevel(FATAL)

from unittest.mock import patch, MagicMock


class TestP2ModeLabels(unittest.TestCase):
    """Verify P2 mode label mapping."""

    def test_all_modes_have_labels(self):
        from flopyarcade.play_split import P2_MODE_LABELS
        self.assertIn('human', P2_MODE_LABELS)
        self.assertIn('replay', P2_MODE_LABELS)
        self.assertIn('policy', P2_MODE_LABELS)

    def test_labels_are_nonempty_strings(self):
        from flopyarcade.play_split import P2_MODE_LABELS
        for mode, label in P2_MODE_LABELS.items():
            self.assertIsInstance(label, str)
            self.assertTrue(len(label) > 0)


class TestReplayMode(unittest.TestCase):
    """Test replay action loading and resolution."""

    def _make_replay_file(self, actions):
        """Create a temporary replay JSON file."""
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(actions, f)
        return path

    def test_replay_loads_plain_list(self):
        actions = ['up', 'left', 'keep', 'down', 'right']
        path = self._make_replay_file(actions)
        try:
            from flopyarcade.play_split import SplitScreenGame
            with patch('flopyarcade.play_split.FloPyEnv', MagicMock()):
                with patch('flopyarcade.play_split.mpl_figure') as mock_fig:
                    mock_fig.return_value = MagicMock()
                    mock_canvas = MagicMock()
                    mock_fig.return_value.canvas = mock_canvas
                    game = SplitScreenGame(p2_mode='replay', p2_replay=path)
                    self.assertEqual(game._p2_actions, actions)
        finally:
            os.unlink(path)

    def test_replay_loads_dict_with_actions_key(self):
        actions = ['up', 'keep', 'down']
        path = self._make_replay_file({'actions': actions, 'envtype': '1s-d'})
        try:
            from flopyarcade.play_split import SplitScreenGame
            with patch('flopyarcade.play_split.FloPyEnv', MagicMock()):
                with patch('flopyarcade.play_split.mpl_figure') as mock_fig:
                    mock_fig.return_value = MagicMock()
                    mock_canvas = MagicMock()
                    mock_fig.return_value.canvas = mock_canvas
                    game = SplitScreenGame(p2_mode='replay', p2_replay=path)
                    self.assertEqual(game._p2_actions, actions)
        finally:
            os.unlink(path)

    def test_replay_file_not_found_raises(self):
        from flopyarcade.play_split import SplitScreenGame
        with self.assertRaises(FileNotFoundError):
            with patch('flopyarcade.play_split.FloPyEnv', MagicMock()):
                with patch('flopyarcade.play_split.mpl_figure') as mock_fig:
                    mock_fig.return_value = MagicMock()
                    mock_fig.return_value.canvas = MagicMock()
                    game = SplitScreenGame(p2_mode='replay', p2_replay='/nonexistent/file.json')

    def test_replay_invalid_action_raises(self):
        path = self._make_replay_file(['up', 'diagonal', 'keep'])
        try:
            from flopyarcade.play_split import SplitScreenGame
            with self.assertRaises(ValueError):
                with patch('flopyarcade.play_split.FloPyEnv', MagicMock()):
                    with patch('flopyarcade.play_split.mpl_figure') as mock_fig:
                        mock_fig.return_value = MagicMock()
                        mock_fig.return_value.canvas = MagicMock()
                        game = SplitScreenGame(p2_mode='replay', p2_replay=path)
        finally:
            os.unlink(path)

    def test_replay_no_path_raises(self):
        from flopyarcade.play_split import SplitScreenGame
        with self.assertRaises(ValueError):
            with patch('flopyarcade.play_split.FloPyEnv', MagicMock()):
                with patch('flopyarcade.play_split.mpl_figure') as mock_fig:
                    mock_fig.return_value = MagicMock()
                    mock_fig.return_value.canvas = MagicMock()
                    game = SplitScreenGame(p2_mode='replay', p2_replay=None)

    def test_replay_action_resolution_within_bounds(self):
        actions = ['up', 'left', 'keep']
        path = self._make_replay_file(actions)
        try:
            from flopyarcade.play_split import SplitScreenGame
            with patch('flopyarcade.play_split.FloPyEnv') as mock_env_cls:
                mock_env = MagicMock()
                mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
                mock_env_cls.return_value = mock_env
                with patch('flopyarcade.play_split.mpl_figure') as mock_fig:
                    mock_fig.return_value = MagicMock()
                    mock_fig.return_value.canvas = MagicMock()
                    game = SplitScreenGame(p2_mode='replay', p2_replay=path)
                    self.assertEqual(game._get_p2_action(0, None), 'up')
                    self.assertEqual(game._get_p2_action(1, None), 'left')
                    self.assertEqual(game._get_p2_action(2, None), 'keep')
                    # Out of bounds returns 'keep'
                    self.assertEqual(game._get_p2_action(99, None), 'keep')
        finally:
            os.unlink(path)


class TestPolicyModeValidation(unittest.TestCase):
    """Test policy mode initialization validation."""

    def test_policy_no_model_raises(self):
        from flopyarcade.play_split import SplitScreenGame
        with self.assertRaises(ValueError):
            with patch('flopyarcade.play_split.FloPyEnv', MagicMock()):
                with patch('flopyarcade.play_split.mpl_figure') as mock_fig:
                    mock_fig.return_value = MagicMock()
                    mock_fig.return_value.canvas = MagicMock()
                    game = SplitScreenGame(p2_mode='policy', p2_model=None)


class TestKeyRoutingP2Mode(unittest.TestCase):
    """Verify that arrow keys are only captured when P2 is human."""

    def test_arrow_keys_captured_in_human_mode(self):
        from flopyarcade.play_split import SplitScreenGame, RIGHT_KEYS
        with patch('flopyarcade.play_split.FloPyEnv', MagicMock()):
            with patch('flopyarcade.play_split.mpl_figure') as mock_fig:
                mock_fig.return_value = MagicMock()
                mock_fig.return_value.canvas = MagicMock()
                game = SplitScreenGame(p2_mode='human')
                event = MagicMock()
                event.key = 'up'
                game._capture_key(event)
                self.assertEqual(game._pending_right, 'up')

    def test_arrow_keys_ignored_in_replay_mode(self):
        from flopyarcade.play_split import SplitScreenGame, RIGHT_KEYS
        # Create a valid empty replay file
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump([], f)
        try:
            with patch('flopyarcade.play_split.FloPyEnv', MagicMock()):
                with patch('flopyarcade.play_split.mpl_figure') as mock_fig:
                    mock_fig.return_value = MagicMock()
                    mock_fig.return_value.canvas = MagicMock()
                    game = SplitScreenGame(p2_mode='replay', p2_replay=path)
                    # Even if key fires, _pending_right should stay None
                    event = MagicMock()
                    event.key = 'up'
                    game._capture_key(event)
                    self.assertIsNone(game._pending_right)
        finally:
            os.unlink(path)


class TestCLIArgs(unittest.TestCase):
    """Verify CLI argument parsing for P2 modes."""

    def test_policy_requires_model(self):
        from flopyarcade.play_split import main_split_screen
        import sys
        # Simulate: --p2-mode policy without --p2-model should error
        old_argv = sys.argv
        sys.argv = ['play_split.py', '--p2-mode', 'policy']
        try:
            with self.assertRaises(SystemExit) as ctx:
                main_split_screen()
            self.assertEqual(ctx.exception.code, 2)
        finally:
            sys.argv = old_argv

    def test_replay_requires_replay_file(self):
        from flopyarcade.play_split import main_split_screen
        import sys
        # Simulate: --p2-mode replay without --p2-replay should error
        old_argv = sys.argv
        sys.argv = ['play_split.py', '--p2-mode', 'replay']
        try:
            with self.assertRaises(SystemExit) as ctx:
                main_split_screen()
            self.assertEqual(ctx.exception.code, 2)
        finally:
            sys.argv = old_argv


if __name__ == '__main__':
    unittest.main()
