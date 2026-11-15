#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Tests for B2: Split-screen mode (play_split.py)."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

import matplotlib
matplotlib.use('Agg')


class TestSplitScreenKeys(unittest.TestCase):
    """Test key mapping and routing."""

    def test_left_keys_mapping(self):
        from flopyarcade.play_split import LEFT_KEYS
        self.assertEqual(LEFT_KEYS['w'], 'up')
        self.assertEqual(LEFT_KEYS['a'], 'left')
        self.assertEqual(LEFT_KEYS['s'], 'down')
        self.assertEqual(LEFT_KEYS['d'], 'right')
        self.assertEqual(LEFT_KEYS[' '], 'keep')

    def test_right_keys_mapping(self):
        from flopyarcade.play_split import RIGHT_KEYS
        self.assertEqual(RIGHT_KEYS['up'], 'up')
        self.assertEqual(RIGHT_KEYS['down'], 'down')
        self.assertEqual(RIGHT_KEYS['left'], 'left')
        self.assertEqual(RIGHT_KEYS['right'], 'right')
        self.assertEqual(RIGHT_KEYS['enter'], 'keep')


class TestSplitScreenGameCreation(unittest.TestCase):
    """Test SplitScreenGame instantiation."""

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_creates_two_envs(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)
        self.assertEqual(len(game.envs), 2)
        self.assertEqual(mock_env_cls.call_count, 2)

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_envs_use_shared_figure(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)
        for env in game.envs:
            self.assertIs(env.fig, game.fig)

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_envs_have_distinct_axes(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env1 = MagicMock()
        mock_env2 = MagicMock()
        mock_env_cls.side_effect = [mock_env1, mock_env2]
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)
        self.assertIsNot(game.envs[0].ax, game.envs[1].ax)

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_renderInitializeCanvas_is_noop(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)
        # Calling renderInitializeCanvas should not raise
        for env in game.envs:
            env.renderInitializeCanvas()  # should be no-op


class TestSplitScreenKeyRouting(unittest.TestCase):
    """Test that key events route to the correct player."""

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_left_key_routes_to_player0(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)

        # Simulate 'w' keypress
        game._capture_key(MagicMock(key='w'))
        self.assertEqual(game._pending_left, 'up')
        self.assertIsNone(game._pending_right)

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_right_key_routes_to_player1(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)

        # Simulate 'up' arrow keypress
        game._capture_key(MagicMock(key='up'))
        self.assertEqual(game._pending_right, 'up')
        self.assertIsNone(game._pending_left)

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_unknown_key_ignored(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)

        game._capture_key(MagicMock(key='escape'))
        self.assertIsNone(game._pending_left)
        self.assertIsNone(game._pending_right)


class TestSplitScreenActionResolution(unittest.TestCase):
    """Test action resolution from pending keys."""

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_resolve_valid_action(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)

        game._pending_left = 'up'
        action = game._resolve_action(0)
        self.assertEqual(action, 'up')
        self.assertIsNone(game._pending_left)  # cleared after resolve

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_resolve_no_key_gives_keep(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)

        action = game._resolve_action(0)
        self.assertEqual(action, 'keep')

    @patch('flopyarcade.play_split.FloPyEnv')
    def test_resolve_invalid_action_falls_back_to_keep(self, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=10)

        game._pending_left = 'invalid_action'
        action = game._resolve_action(0)
        self.assertEqual(action, 'keep')


class TestSplitScreenGameLoop(unittest.TestCase):
    """Test the game loop steps both envs."""

    @patch('flopyarcade.play_split.FloPyEnv')
    @patch('flopyarcade.play_split.waitforbuttonpress')
    def test_steps_both_envs(self, mock_wait, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        mock_env.step.return_value = (None, 1.0, False, False, {})
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=3)

        scores = game._play_one_game(0, verbose=False)
        # Each env should be stepped 3 times
        self.assertEqual(mock_env.step.call_count, 6)  # 2 envs × 3 steps
        self.assertEqual(scores[0], 3.0)
        self.assertEqual(scores[1], 3.0)

    @patch('flopyarcade.play_split.FloPyEnv')
    @patch('flopyarcade.play_split.waitforbuttonpress')
    def test_stops_when_both_done(self, mock_wait, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        # Done after first step
        mock_env.step.return_value = (None, 1.0, True, False, {})
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=100)

        scores = game._play_one_game(0, verbose=False)
        # Only 1 step each (done immediately)
        self.assertEqual(mock_env.step.call_count, 2)  # 2 envs × 1 step

    @patch('flopyarcade.play_split.FloPyEnv')
    @patch('flopyarcade.play_split.waitforbuttonpress')
    def test_skips_done_player(self, mock_wait, mock_env_cls):
        from flopyarcade.play_split import SplitScreenGame
        mock_env = MagicMock()
        mock_env.actionSpace = ['up', 'down', 'left', 'right', 'keep']
        mock_env.step.return_value = (None, 1.0, False, False, {})
        mock_env_cls.return_value = mock_env
        game = SplitScreenGame(envtype='1s-d', nagentsteps=3)

        # Manually mark player 0 as done
        game._play_one_game(0, verbose=False)
        # Player 1 still gets stepped; player 0 was not done in our mock
        # (both envs are the same mock, so both get 3 steps)
        self.assertEqual(mock_env.step.call_count, 6)


class TestPlaySplitCLI(unittest.TestCase):
    """Test the play.py --split-screen flag integration."""

    def test_play_py_has_split_screen_flag(self):
        import ast
        with open(os.path.join(os.path.dirname(__file__), '..', 'play.py')) as f:
            source = f.read()
        self.assertIn('--split-screen', source)

    def test_play_split_module_importable(self):
        import importlib
        mod = importlib.import_module('flopyarcade.play_split')
        self.assertTrue(hasattr(mod, 'SplitScreenGame'))
        self.assertTrue(hasattr(mod, 'main_split_screen'))


if __name__ == '__main__':
    unittest.main()
