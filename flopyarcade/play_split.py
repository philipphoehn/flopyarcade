#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade — B2: Split-screen mode (human vs human / replay / policy)
# Left player: WASD | Right player: Arrow keys (or auto: replay/policy)
# author: Philipp Hoehn

import argparse
import json
import os
import sys
from argparse import ArgumentParser
from time import time, sleep
from numpy.random import randint

# Suppress TF/flopy noise before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from logging import getLogger, FATAL
getLogger('tensorflow').setLevel(FATAL)
getLogger('flopy').setLevel(FATAL)

from matplotlib import use as mpl_use
try:
    mpl_use('TkAgg')
except Exception:
    pass
from matplotlib.pyplot import figure as mpl_figure, show, close, waitforbuttonpress

from flopyarcade import FloPyEnv
from flopyarcade.agent import FloPyAgent


# ─── Key mappings ───────────────────────────────────────────────────────────────

LEFT_KEYS = {
    'w': 'up', 'a': 'left', 's': 'down', 'd': 'right', ' ': 'keep'
}
RIGHT_KEYS = {
    'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right', 'enter': 'keep'
}

# ─── P2 mode labels ─────────────────────────────────────────────────────────────

P2_MODE_LABELS = {
    'human': 'P2 (Arrows)',
    'replay': 'P2 (Replay)',
    'policy': 'P2 (Policy AI)',
}


class SplitScreenGame:
    """Two-player split-screen FloPyArcade.

    One matplotlib figure with two side-by-side subplots.
    Left player uses WASD. Right player can be:
      - human (default): arrow keys
      - replay: actions loaded from a JSON file
      - policy: a saved DQN/keras model
    Both players step simultaneously; the game ends when both are done.
    """

    def __init__(self, envtype='1s-d', seed=1, nlay=1, nrow=100, ncol=100,
                 nagentsteps=200, manualcontrol_time=1.5,
                 blit=False, smooth_frames=4, smooth_subdiv=10,
                 pathmf2005=None, pathmp6=None, modelname='unittest',
                 p2_mode='human', p2_model=None, p2_replay=None):
        self.envtype = envtype
        self.seed = seed
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.nagentsteps = nagentsteps
        self.wait_time = manualcontrol_time
        self.modelname = modelname
        self.p2_mode = p2_mode
        self.p2_model = p2_model
        self.p2_replay = p2_replay

        # ── P2 setup ──
        self._p2_agent = None      # FloPyAgent instance (policy mode)
        self._p2_model_loaded = None  # keras model (policy mode)
        self._p2_actions = None   # list of actions (replay mode)

        if p2_mode == 'policy':
            self._init_policy(p2_model)
        elif p2_mode == 'replay':
            self._init_replay(p2_replay)

        # ── Shared figure ──
        p2_label = P2_MODE_LABELS.get(p2_mode, P2_MODE_LABELS['human'])
        self.fig = mpl_figure(figsize=(14, 7), dpi=100)
        self.fig.suptitle(
            f'FloPyArcade  \u2014  Split Screen  '
            f'(Left: W A S D   |   Right: {p2_label})',
            fontsize=13, y=0.97
        )
        self.ax_left = self.fig.add_subplot(1, 2, 1, aspect='equal', adjustable='box')
        self.ax_right = self.fig.add_subplot(1, 2, 2, aspect='equal', adjustable='box')

        # ── Create two env instances ──
        self.envs = []
        for i, ax in enumerate([self.ax_left, self.ax_right]):
            env = FloPyEnv(
                ENVTYPE=envtype,
                PATHMF2005=pathmf2005,
                PATHMP6=pathmp6,
                _seed=seed,
                MODELNAME=f'{modelname}_p{i + 1}',
                ANIMATIONFOLDER=None,
                flagSavePlot=False,
                flagManualControl=False,
                flagRender=True,
                NAGENTSTEPS=nagentsteps,
                nLay=nlay,
                nRow=nrow,
                nCol=ncol,
                OBSPREP=True,
                initWithSolution=True
            )
            # Redirect to shared figure / subplot
            env.fig = self.fig
            env.ax = ax
            env.ax3 = ax.twinx()
            env.ax2 = env.ax3.twiny()
            env.canvasInitialized = True
            env._blit_enabled = blit
            env.SMOOTH_TRAJ_FRAMES = smooth_frames
            env.SMOOTH_TRAJ_SUBDIV = smooth_subdiv
            env.keyPressed = None
            env.renderInitializeCanvas = lambda: None
            # Label the subplot
            if i == 0:
                ax.set_title('P1 (WASD)', fontsize=11, fontweight='bold')
            else:
                ax.set_title(p2_label, fontsize=11, fontweight='bold')
            self.envs.append(env)

        # ── Shared key handler ──
        self._pending_left = None
        self._pending_right = None
        self.fig.canvas.mpl_connect('key_press_event', self._capture_key)

    def _init_policy(self, model_name):
        """Load a policy model for P2."""
        if not model_name:
            raise ValueError('--p2-model is required when --p2-mode=policy')

        # Resolve model path
        model_prefix = model_name
        if os.path.isabs(model_name):
            # If absolute path given, use its directory as modelpth and basename as prefix
            dirpath = os.path.dirname(model_name)
            model_prefix = os.path.basename(model_name)
        else:
            dirpath = None

        # Create a FloPyAgent to use its model-loading machinery
        agent_settings = {
            'ENVTYPE': self.envtype,
            'MODELNAME': self.modelname,
            'NAGENTSTEPS': self.nagentsteps,
            'NLAY': self.nlay,
            'NROW': self.nrow,
            'NCOL': self.ncol,
            'SEEDENV': self.seed,
        }
        self._p2_agent = FloPyAgent(agent_settings)

        # If we have a directory override, patch modelpth
        if dirpath:
            self._p2_agent.modelpth = dirpath

        # Load the model once (avoids per-step loading overhead)
        self._p2_model_loaded = self._p2_agent.loadAgentModel(model_prefix)
        print(f'[B2-policy] Loaded P2 model: {model_prefix}')

    def _init_replay(self, replay_path):
        """Load a replay action file for P2."""
        if not replay_path:
            raise ValueError('--p2-replay is required when --p2-mode=replay')
        if not os.path.exists(replay_path):
            raise FileNotFoundError(f'Replay file not found: {replay_path}')

        with open(replay_path) as f:
            data = json.load(f)

        # Accept either a plain list of actions or a dict with 'actions' key
        if isinstance(data, dict):
            if 'actions' in data:
                self._p2_actions = data['actions']
            else:
                raise ValueError("Replay JSON must be a list or have an 'actions' key")
        elif isinstance(data, list):
            self._p2_actions = data
        else:
            raise ValueError('Replay JSON must be a list or dict with actions key')

        # Validate actions
        valid_actions = {'up', 'down', 'left', 'right', 'keep'}
        for i, a in enumerate(self._p2_actions):
            if a not in valid_actions:
                raise ValueError(f'Replay action #{i} "{a}" not in {valid_actions}')

        print(f'[B2-replay] Loaded {len(self._p2_actions)} actions from {replay_path}')

    def _get_p2_action(self, step, obs):
        """Get the action for P2 based on mode."""
        if self.p2_mode == 'human':
            # Fall back to key-based input
            action = self._pending_right
            self._pending_right = None
            env = self.envs[1]
            if action is None:
                return 'keep'
            if action in env.actionSpace:
                return action
            return 'keep'
        elif self.p2_mode == 'replay':
            if self._p2_actions and step < len(self._p2_actions):
                return self._p2_actions[step]
            return 'keep'
        elif self.p2_mode == 'policy':
            if self._p2_model_loaded is not None:
                actionIdx = int(max(range(len(self.envs[1].actionSpace)),
                                    key=lambda i: self._p2_model_loaded.predict_on_batch(
                                        __import__('numpy').array(obs).reshape(-1, *__import__('numpy').shape(obs))
                                    )[0][i]))
                return self.envs[1].actionSpace[actionIdx]
            return 'keep'
        return 'keep'

    def _capture_key(self, event):
        """Route key events to the correct player."""
        key = event.key
        if key in LEFT_KEYS:
            self._pending_left = LEFT_KEYS[key]
        elif key in RIGHT_KEYS and self.p2_mode == 'human':
            self._pending_right = RIGHT_KEYS[key]

    def _resolve_action(self, player_idx, step=0, obs=None):
        """Return the action for a player and clear the pending key."""
        if player_idx == 0:
            action = self._pending_left
            self._pending_left = None
            if action is None:
                return 'keep'
            env = self.envs[0]
            if action in env.actionSpace:
                return action
            return 'keep'
        else:
            return self._get_p2_action(step, obs)

    def play(self, ngames=1, verbose=True):
        """Run ngames split-screen episodes."""
        all_scores = []
        for game in range(ngames):
            scores = self._play_one_game(game, verbose)
            all_scores.append(scores)
        close(self.fig)
        return all_scores

    def _play_one_game(self, game_idx, verbose=True):
        """Play a single split-screen episode."""
        p2_label = P2_MODE_LABELS.get(self.p2_mode, P2_MODE_LABELS['human'])
        if verbose:
            print(f'\n=== Split-Screen Game {game_idx + 1} ===')
            print(f'  P1:  W=up  A=left  S=down  D=right  SPACE=keep')
            if self.p2_mode == 'human':
                print(f'  P2:  \u2191=up  \u2193=down  \u2190=left  \u2192=right  ENTER=keep')
            elif self.p2_mode == 'replay':
                print(f'  P2:  replay from {self.p2_replay}')
            elif self.p2_mode == 'policy':
                print(f'  P2:  policy model ({self.p2_model})')

        t0 = time()

        # Reset both envs
        for env in self.envs:
            env.reset(seed=self.seed, MODELNAME=env.MODELNAME)
            env.timeStep = 0

        # Initial render
        self.fig.canvas.draw()
        show(block=False)
        self.fig.canvas.flush_events()

        done = [False, False]
        scores = [0.0, 0.0]
        steps_played = 0

        for step in range(self.nagentsteps):
            if all(done):
                break

            # Get observation for P2 (needed for policy/replay)
            obs = None
            if self.p2_mode in ('policy', 'replay') and not done[1]:
                try:
                    obs = self.envs[1].obs  # current observation
                except Exception:
                    obs = None

            # Resolve actions
            actions = [self._resolve_action(0), self._resolve_action(1, step, obs)]

            # Step each active env
            for i in range(2):
                if done[i]:
                    continue
                env = self.envs[i]
                obs_r, reward, done_flag, truncated, info = env.step(actions[i])
                scores[i] += reward
                done[i] = done_flag

            steps_played += 1

            # Unified pause (both players see the same frame)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            waitforbuttonpress(timeout=self.wait_time)

        # Results
        elapsed = time() - t0
        if verbose:
            print(f'\n  Steps played: {steps_played}')
            print(f'  P1 (WASD):        {scores[0]:.4f}')
            print(f'  P2 ({p2_label}):  {scores[1]:.4f}')
            if scores[0] > scores[1]:
                print('  \u2705 P1 wins!')
            elif scores[1] > scores[0]:
                print(f'  \u2705 P2 ({p2_label}) wins!')
            else:
                print('  \u2610 Tie!')
            print(f'  Time: {elapsed:.1f}s')

        return scores


def main_split_screen():
    """CLI entry point for split-screen mode."""
    parser = ArgumentParser(description='FloPyArcade split-screen (2-player)')
    parser.add_argument('--envtype', default='1s-d', type=str)
    parser.add_argument('--seedenv', default=1, type=int)
    parser.add_argument('--nlay', default=1, type=int)
    parser.add_argument('--nrow', default=100, type=int)
    parser.add_argument('--ncol', default=100, type=int)
    parser.add_argument('--nagentsteps', default=200, type=int)
    parser.add_argument('--ngames', default=1, type=int)
    parser.add_argument('--wait', default=1.5, type=float,
                        help='seconds to wait per step for player input (default: 1.5)')
    parser.add_argument('--smooth-frames', default=4, type=int)
    parser.add_argument('--smooth-subdiv', default=10, type=int)
    parser.add_argument('--pathmf2005', default=None, type=str)
    parser.add_argument('--pathmp6', default=None, type=str)
    parser.add_argument('--modelname', default='unittest', type=str)
    # ── P2 mode ──
    parser.add_argument('--p2-mode', default='human', type=str,
                        choices=['human', 'replay', 'policy'],
                        help='P2 control mode: human (arrows), replay (JSON), policy (model)')
    parser.add_argument('--p2-model', default=None, type=str,
                        help='Model name/path for --p2-mode=policy (e.g. 1s-d or /path/to/model)')
    parser.add_argument('--p2-replay', default=None, type=str,
                        help='JSON file with action list for --p2-mode=replay')
    args = parser.parse_args()

    # Validate P2 mode
    if args.p2_mode == 'policy' and not args.p2_model:
        parser.error('--p2-model is required when --p2-mode=policy')
    if args.p2_mode == 'replay' and not args.p2_replay:
        parser.error('--p2-replay is required when --p2-mode=replay')

    game = SplitScreenGame(
        envtype=args.envtype,
        seed=args.seedenv,
        nlay=args.nlay,
        nrow=args.nrow,
        ncol=args.ncol,
        nagentsteps=args.nagentsteps,
        manualcontrol_time=args.wait,
        smooth_frames=args.smooth_frames,
        smooth_subdiv=args.smooth_subdiv,
        pathmf2005=args.pathmf2005,
        pathmp6=args.pathmp6,
        modelname=args.modelname,
        p2_mode=args.p2_mode,
        p2_model=args.p2_model,
        p2_replay=args.p2_replay
    )
    scores = game.play(ngames=args.ngames)
    return scores


if __name__ == '__main__':
    main_split_screen()
