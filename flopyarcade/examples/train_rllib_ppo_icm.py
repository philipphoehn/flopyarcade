#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FloPyArcade — Ray RLlib PPO + ICM training.

Proximal Policy Optimization with Intrinsic Curiosity Module:
on-policy, discrete or continuous, exploration bonus.

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
from common_rllib import (
    add_common_args, make_flopy_env, setup_logging,
    get_outdir, effective_seed, parse_net_arch
)
from ray import tune
from ray import init as rayInit
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from flopyarcade import FloPyEnv

ENVTYPE = None


def env_creator(env_config):
    env_config = env_config or {}
    env_config['ENVTYPE'] = ENVTYPE
    return FloPyEnv(env_config=env_config)


def main():
    global ENVTYPE

    parser = ArgumentParser(description=__doc__)
    add_common_args(parser, default_envtype='6r-c', default_timesteps=10_000_000)
    parser.add_argument('--lr', default=5e-5, type=float,
        help='learning rate (default: 0.00005)')
    parser.add_argument('--clip-range', default=0.2, type=float,
        help='PPO clip range (default: 0.2)')
    parser.add_argument('--icm-eta', default=1.0, type=float,
        help='weight for intrinsic curiosity reward (default: 1.0)')
    parser.add_argument('--icm-embed-dim', default=256, type=int,
        help='embedding dimension for ICM feature network (default: 256)')
    parser.add_argument('--icm-batch', default=256, type=int,
        help='ICM forward/inverse net batch size (default: 256)')
    parser.add_argument('--framework', default='tf', type=str,
        choices=['tf', 'torch'], help='deep learning framework (default: tf)')
    args = parser.parse_args()

    ENVTYPE = args.envtype
    setup_logging(args.verbose)

    if args.quick_test:
        args.timesteps = 200
        args.eval_episodes = 1

    outdir = get_outdir(args, 'PPO-ICM')

    rayInit(log_to_driver=not args.render, logging_level=logging.FATAL if args.render else logging.ERROR)
    register_env('flopy_ppo_icm', env_creator)

    model_cfg = {
        "fcnet_hiddens": [512, 256],
        "fcnet_activation": "relu",
    }
    net_arch = parse_net_arch(args.net_arch)
    if net_arch:
        model_cfg["fcnet_hiddens"] = net_arch

    config = (
        PPOConfig()
        .environment('flopy_ppo_icm')
        .seed(args.seed)
        .num_workers(args.cpus)
        .num_gpus(args.gpus)
        .framework(args.framework)
        .lr(args.lr)
        .gamma(args.gamma)
        .cliprange(args.clip_range)
        .train_batch_size(512)
        .rollout_fragment_length(50)
        .timesteps_per_iteration(10000)
        .min_iter_time_s(30)
        .model(model_cfg)
        .exploration_config({
            "type": "Curiosity",
            "eta": args.icm_eta,
            "lr": 0.001,
            "feature_dim": args.icm_embed_dim,
            "feature_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
            },
            "inverse_net_hiddens": [args.icm_batch],
            "inverse_net_activation": "relu",
            "forward_net_hiddens": [args.icm_batch],
            "forward_net_activation": "relu",
            "beta": 0.2,
            "sub_exploration": {
                "type": "StochasticSampling",
            }
        })
        .evaluation_config({
            "n_steps": args.max_steps_eval,
            "episode_reward_threshold": args.reward_threshold,
            "evaluation_interval": 1,
            "evaluation_num_episodes": args.eval_episodes,
        })
    )

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        import ray.rllib.algorithms as alg
        PPO_algo = alg.PPO.from_checkpoint(args.resume)
        PPO_algo.train()
        PPO_algo.save()
        print(f"Checkpoint saved to {PPO_algo.checkpoint_path}")
        return

    stop = {
        "timesteps_total": args.timesteps,
        "episode_reward_mean": args.reward_threshold,
    }

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        name=f'{ENVTYPE}-PPO-ICM',
        local_dir=outdir,
        verbose=2,
        checkpoint_at_end=True,
    )
    print(f'\nTraining complete. Best trial: {results.best_trial}')
    print(f'Results saved to: {outdir}')


if __name__ == '__main__':
    main()
