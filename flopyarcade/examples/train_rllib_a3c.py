#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FloPyArcade — Ray RLlib A3C training.

Asynchronous Advantage Actor-Critic: on-policy, discrete actions.

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
from ray.rllib.algorithms.a3c import A3CConfig

from flopyarcade import FloPyEnv

ENVTYPE = None


def env_creator(env_config):
    env_config = env_config or {}
    env_config['ENVTYPE'] = ENVTYPE
    return FloPyEnv(env_config=env_config)


def main():
    global ENVTYPE

    parser = ArgumentParser(description=__doc__)
    add_common_args(parser, default_envtype='1s-d', default_timesteps=5_000_000)
    parser.add_argument('--lr', default=5e-5, type=float,
        help='learning rate (default: 0.00005)')
    parser.add_argument('--vf-loss-coef', default=0.5, type=float,
        help='value function loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', default=0.01, type=float,
        help='entropy coefficient (default: 0.01)')
    parser.add_argument('--framework', default='tf', type=str,
        choices=['tf', 'torch'], help='deep learning framework (default: tf)')
    args = parser.parse_args()

    ENVTYPE = args.envtype
    setup_logging(args.verbose)

    if args.quick_test:
        args.timesteps = 200
        args.eval_episodes = 1

    outdir = get_outdir(args, 'A3C')

    rayInit(log_to_driver=not args.render, logging_level=logging.FATAL if args.render else logging.ERROR)
    register_env('flopy_a3c', env_creator)

    model_cfg = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    }
    net_arch = parse_net_arch(args.net_arch)
    if net_arch:
        model_cfg["fcnet_hiddens"] = net_arch

    config = (
        A3CConfig()
        .environment('flopy_a3c')
        .seed(args.seed)
        .num_workers(args.cpus)
        .num_gpus(args.gpus)
        .framework(args.framework)
        .lr(args.lr)
        .gamma(args.gamma)
        .vf_loss_coeff(args.vf_loss_coef)
        .entropy_coeff(args.entropy_coef)
        .rollout_fragment_length(128)
        .train_batch_size(512)
        .timesteps_per_iteration(10000)
        .min_iter_time_s(30)
        .model(model_cfg)
        .evaluation_config({
            "n_steps": args.max_steps_eval,
            "episode_reward_threshold": args.reward_threshold,
            "evaluation_interval": 1,
            "evaluation_num_episodes": args.eval_episodes,
        })
    )

    stop = {
        "timesteps_total": args.timesteps,
        "episode_reward_mean": args.reward_threshold,
    }

    results = tune.run(
        "A3C",
        config=config.to_dict(),
        stop=stop,
        name=f'{ENVTYPE}-A3C',
        local_dir=outdir,
        verbose=2,
        checkpoint_at_end=True,
    )
    print(f'\nTraining complete. Best trial: {results.best_trial}')
    print(f'Results saved to: {outdir}')


if __name__ == '__main__':
    main()
