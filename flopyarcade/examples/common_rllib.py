#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for FloPyArcade training examples.

Provides common argparse setup, environment factory, logging,
evaluation loop, and net-arch parsing used across all training scripts.
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import abspath, dirname, join

import numpy as np

from flopyarcade import FloPyEnv


def str_to_bool(value):
    """Parse a string into a boolean (for argparse type=)."""
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value!r} is not a valid boolean value')


def parse_net_arch(s):
    """Parse a comma-separated string like '512,256,128' into [512, 256, 128]."""
    if s is None:
        return None
    return [int(x) for x in s.split(',')]


def add_common_args(parser, default_envtype='3s-c', default_timesteps=1_000_000):
    """Add the common set of arguments shared by all training scripts."""
    parser.add_argument('--envtype', default=default_envtype, type=str,
        help=f'environment type string (default: {default_envtype})')
    parser.add_argument('--seed', default=1, type=int,
        help='random seed for the algorithm (default: 1)')
    parser.add_argument('--seed-env', default=None, type=int,
        help='seed for environment initialization (default: same as --seed)')
    parser.add_argument('--timesteps', default=default_timesteps, type=int,
        help=f'total timesteps for training (default: {default_timesteps:,})')
    parser.add_argument('--lr', default=3e-4, type=float,
        help='learning rate (default: 0.0003)')
    parser.add_argument('--gamma', default=0.99, type=float,
        help='discount factor (default: 0.99)')
    parser.add_argument('--net-arch', default=None, type=str,
        help='comma-separated hidden layer sizes, e.g. "512,256,128" (default: algorithm default)')
    parser.add_argument('--cpus', default=4, type=int,
        help='number of CPU workers (default: 4)')
    parser.add_argument('--gpus', default=0, type=int,
        help='number of GPUs (default: 0)')
    parser.add_argument('--render', default=False, type=str_to_bool,
        help='enable rendering / benchmark visualization')
    parser.add_argument('--outdir', default=None, type=str,
        help='output directory for results (default: ./results/<env>-<algo>)')
    parser.add_argument('--reward-threshold', default=990.0, type=float,
        help='stop training when mean episode reward reaches this (default: 990)')
    parser.add_argument('--eval-episodes', default=5, type=int,
        help='number of evaluation episodes (default: 5)')
    parser.add_argument('--save-freq', default=100_000, type=int,
        help='checkpoint save frequency in timesteps (default: 100000)')
    parser.add_argument('--max-steps-eval', default=500, type=int,
        help='max steps per evaluation episode (safety, default: 500)')
    parser.add_argument('--resume', default=None, type=str,
        help='path to a saved model to resume training from')
    parser.add_argument('--eval-only', default=None, type=str,
        help='path to a saved model; run evaluation only (no training)')
    parser.add_argument('--quick-test', action='store_true',
        help='run a minimal 200-timestep smoke test to verify wiring')
    parser.add_argument('--verbose', default=1, type=int, choices=[0, 1, 2, 3],
        help='verbosity level 0-3 (default: 1)')
    return parser


def make_flopy_env(envtype, render=False, seed_env=None):
    """Create a FloPyArcade environment with the given config."""
    env_config = {'ENVTYPE': envtype}
    if render:
        env_config['flagRender'] = True
    if seed_env is not None:
        env_config['seed'] = seed_env
    return FloPyEnv(env_config=env_config)


def setup_logging(verbose=1):
    """Configure logging based on verbosity level."""
    import logging
    levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=levels.get(verbose, logging.WARNING),
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')


def run_evaluation(model, env, max_episodes=5, max_steps=500, verbose=1):
    """Run N evaluation episodes and return mean reward.
    
    Works with both SB3 models (model.predict) and generic predict interfaces.
    """
    import logging
    logger = logging.getLogger('eval')
    rewards = []

    for ep in range(max_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
        rewards.append(total_reward)
        if verbose >= 2:
            logger.info(f'  Episode {ep+1}: reward={total_reward:.2f} ({step} steps)')

    mean_reward = float(np.mean(rewards))
    logger.info(f'Mean reward over {max_episodes} episodes: {mean_reward:.2f}')
    return mean_reward


def get_outdir(args, algo_name):
    """Resolve the output directory."""
    if args.outdir:
        outdir = args.outdir
    else:
        wrkspc = abspath(dirname(__file__))
        outdir = join(wrkspc, 'results', f'{args.envtype}-{algo_name}')
    makedirs(outdir, exist_ok=True)
    return outdir


def effective_seed(args):
    """Return the seed to use, defaulting env seed to algorithm seed."""
    if args.seed_env is not None:
        return args.seed_env
    return args.seed
