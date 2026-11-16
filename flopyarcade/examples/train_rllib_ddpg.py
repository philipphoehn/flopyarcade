#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FloPyArcade — Ray RLlib DDPG training.

Deep Deterministic Policy Gradient: off-policy, continuous actions.

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
from os.path import join
from common import (
    add_common_args, make_flopy_env, setup_logging,
    get_outdir, effective_seed, parse_net_arch
)
from ray import tune
from ray import init as rayInit
from ray.tune.registry import register_env
from ray.rllib.algorithms.ddpg import DDPGConfig

from flopyarcade import FloPyEnv

ENVTYPE = None


def env_creator(env_config):
    env_config = env_config or {}
    env_config['ENVTYPE'] = ENVTYPE
    return FloPyEnv(env_config=env_config)


def main():
    global ENVTYPE

    parser = ArgumentParser(description=__doc__)
    add_common_args(parser, default_envtype='3s-c', default_timesteps=10_000_000)
    parser.add_argument('--lr', default=5e-5, type=float,
        help='learning rate (default: 0.00005)')
    parser.add_argument('--buffer-size', default=3_000_000, type=int,
        help='replay buffer capacity (default: 3000000)')
    parser.add_argument('--noise-sigma', default=0.1, type=float,
        help='exploration noise sigma (default: 0.1)')
    parser.add_argument('--framework', default='tf', type=str,
        choices=['tf', 'torch'], help='deep learning framework (default: tf)')
    args = parser.parse_args()

    ENVTYPE = args.envtype
    setup_logging(args.verbose)

    if args.quick_test:
        args.timesteps = 200
        args.eval_episodes = 1

    outdir = get_outdir(args, 'DDPG')
    seed = args.seed

    rayInit(log_to_driver=not args.render, logging_level=logging.FATAL if args.render else logging.ERROR)
    register_env('flopy_ddpg', env_creator)

    model_cfg = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    }
    net_arch = parse_net_arch(args.net_arch)
    if net_arch:
        model_cfg["fcnet_hiddens"] = net_arch

    config = (
        DDPGConfig()
        .environment('flopy_ddpg')
        .seed(seed)
        .num_workers(args.cpus)
        .num_gpus(args.gpus)
        .framework(args.framework)
        .lr(args.lr)
        .gamma(args.gamma)
        .replay_buffer_config({"capacity": args.buffer_size})
        .exploration_config({"type": "GaussianNoise", "sigma": args.noise_sigma})
        .train_batch_size(256)
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

    if args.resume:
        print(f'Resuming from {args.resume}')
        trainer = DDPGConfig().restore_from_path(args.resume) if False else None
        # RLlib resume: load checkpoint and continue
        algo_cls = __import__('ray.rllib.algorithms.ddpg', fromlist=['DDPG']).DDPG
        trainer = algo_cls.from_checkpoint(args.resume)
        trainer.train()
        trainer.save()
        print(f'Checkpoint saved to {trainer.checkpoint_path}')
    else:
        results = tune.run(
            "DDPG",
            config=config.to_dict(),
            stop=stop,
            name=f'{ENVTYPE}-DDPG',
            local_dir=outdir,
            verbose=2,
            checkpoint_at_end=True,
        )
        print(f'\nTraining complete. Best trial: {results.best_trial}')
        print(f'Results saved to: {outdir}')


if __name__ == '__main__':
    main()
