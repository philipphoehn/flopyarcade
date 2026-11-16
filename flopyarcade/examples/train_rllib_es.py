#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FloPyArcade — Ray RLlib ES (Evolution Strategies) training.

Evolution Strategies: lightweight, no critic, discrete or continuous.

Requires: ray[default]>=2.9, flopy>=3.9.2, gymnasium, tensorflow
"""

import logging
logging.disable(logging.WARNING)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore', 'SelectableGroups dict interface')
warnings.filterwarnings('ignore', category=DeprecationWarning)

from argparse import ArgumentParser
from common import (
    add_common_args, make_flopy_env, setup_logging,
    get_outdir, effective_seed, parse_net_arch
)
from ray import tune
from ray import init as rayInit
from ray.tune.registry import register_env
from ray.rllib.algorithms.es import ESConfig

from flopyarcade import FloPyEnv

ENVTYPE = None


def env_creator(env_config):
    env_config = env_config or {}
    env_config['ENVTYPE'] = ENVTYPE
    return FloPyEnv(env_config=env_config)


def main():
    global ENVTYPE

    parser = ArgumentParser(description=__doc__)
    add_common_args(parser, default_envtype='1s-d', default_timesteps=2_000_000)
    parser.add_argument('--lr', default=0.01, type=float,
        help='learning rate (default: 0.01)')
    parser.add_argument('--popsize', default=512, type=int,
        help='population size (default: 512)')
    parser.add_argument('--sigma', default=0.5, type=float,
        help='mutation noise sigma (default: 0.5)')
    parser.add_argument('--framework', default='tf', type=str,
        choices=['tf', 'torch'], help='deep learning framework (default: tf)')
    args = parser.parse_args()

    ENVTYPE = args.envtype
    setup_logging(args.verbose)

    if args.quick_test:
        args.timesteps = 200
        args.eval_episodes = 1
        args.popsize = 32

    outdir = get_outdir(args, 'ES')

    rayInit(log_to_driver=not args.render, logging_level=logging.FATAL if args.render else logging.ERROR)
    register_env('flopy_es', env_creator)

    model_cfg = {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
    }
    net_arch = parse_net_arch(args.net_arch)
    if net_arch:
        model_cfg["fcnet_hiddens"] = net_arch

    config = (
        ESConfig()
        .environment('flopy_es')
        .seed(args.seed)
        .num_workers(args.cpus)
        .num_gpus(args.gpus)
        .framework(args.framework)
        .lr(args.lr)
        .gamma(args.gamma)
        .train_batch_size(args.popsize)
        .model(model_cfg)
        .evaluation_config({
            "n_steps": args.max_steps_eval,
            "episode_reward_threshold": args.reward_threshold,
            "evaluation_interval": 1,
            "evaluation_num_episodes": args.eval_episodes,
        })
    )
    # ES-specific: mutation sigma
    config["sigma"] = args.sigma

    stop = {
        "timesteps_total": args.timesteps,
        "episode_reward_mean": args.reward_threshold,
    }

    results = tune.run(
        "ES",
        config=config.to_dict(),
        stop=stop,
        name=f'{ENVTYPE}-ES',
        local_dir=outdir,
        verbose=2,
        checkpoint_at_end=True,
    )
    print(f'\nTraining complete. Best trial: {results.best_trial}')
    print(f'Results saved to: {outdir}')


if __name__ == '__main__':
    main()
