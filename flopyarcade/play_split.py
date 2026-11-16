#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade — B2: Split-screen mode (two human players side-by-side)
# Left player: WASD | Right player: Arrow keys
# author: Philipp Hoehn

import argparse
from argparse import ArgumentParser
from time import time, sleep
from numpy.random import randint

# Suppress TF/flopy noise before imports
import os
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


class SplitScreenGame:
    """Two-player split-screen FloPyArcade.

    One matplotlib figure with two side-by-side subplots.
    Left player uses WASD, right player uses arrow keys.
    Both players step simultaneously; the game ends when both are done.
    """

    def __init__(self, envtype='1s-d', seed=1, nlay=1, nrow=100, ncol=100,
                 nagentsteps=200, manualcontrol_time=1.5,
                 blit=False, smooth_frames=4, smooth_subdiv=10,
                 pathmf2005=None, pathmp6=None, modelname='unittest'):
        self.envtype = envtype
        self.seed = seed
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.nagentsteps = nagentsteps
        self.wait_time = manualcontrol_time
        self.modelname = modelname

        # ── Shared figure ──
        self.fig = mpl_figure(figsize=(14, 7), dpi=100)
        self.fig.suptitle(
            'FloPyArcade  \u2014  Split Screen  '
            '(Left: W A S D   |   Right: \u2191 \u2193 \u2190 \u2192)',
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
            # Prevent per-env canvas re-initialisation
            env.renderInitializeCanvas = lambda: None
            # Label the subplot
            ax.set_title(f'Player {i + 1}', fontsize=11, fontweight='bold')
            self.envs.append(env)

        # ── Shared key handler ──
        self._pending_left = None
        self._pending_right = None
        self.fig.canvas.mpl_connect('key_press_event', self._capture_key)

    def _capture_key(self, event):
        """Route key events to the correct player."""
        key = event.key
        if key in LEFT_KEYS:
            self._pending_left = LEFT_KEYS[key]
        elif key in RIGHT_KEYS:
            self._pending_right = RIGHT_KEYS[key]

    def _resolve_action(self, player_idx):
        """Return the action for a player and clear the pending key."""
        if player_idx == 0:
            action = self._pending_left
            self._pending_left = None
        else:
            action = self._pending_right
            self._pending_right = None
        # Validate against action space
        env = self.envs[player_idx]
        if action is None:
            return 'keep'
        if action in env.actionSpace:
            return action
        return 'keep'

    def play(self, ngames=1, verbose=True):
        """Run ngames split-screen episodes."""
        all_scores = []
        for game in range(ngames):
            scores = self._play_one_game(game, verbose)
            all_scores.append(scores)
        close(self.fig)
        return all_scores

    def _play_one_game(self, game_idx, verbose):
        """Play a single split-screen episode."""
        if verbose:
            print(f'\n=== Split-Screen Game {game_idx + 1} ===')
            print('  Left:  W=up  A=left  S=down  D=right  SPACE=keep')
            print('  Right: \u2191=up  \u2193=down  \u2190=left  \u2192=right  ENTER=keep')

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

            # Resolve actions from captured keys
            actions = [self._resolve_action(0), self._resolve_action(1)]

            # Step each active env
            for i in range(2):
                if done[i]:
                    continue
                env = self.envs[i]
                obs, reward, done_flag, truncated, info = env.step(actions[i])
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
            print(f'  Player 1 (WASD):    {scores[0]:.4f}')
            print(f'  Player 2 (Arrows):  {scores[1]:.4f}')
            if scores[0] > scores[1]:
                print('  \u2705 Player 1 wins!')
            elif scores[1] > scores[0]:
                print('  \u2705 Player 2 wins!')
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
    args = parser.parse_args()

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
        modelname=args.modelname
    )
    scores = game.play(ngames=args.ngames)
    return scores


if __name__ == '__main__':
    main_split_screen()
