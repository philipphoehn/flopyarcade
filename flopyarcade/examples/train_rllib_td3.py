#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - Ray RLlib TD3 training
# Twin Delayed Deep Deterministic Policy Gradient for continuous actions

import os
import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser
from os.path import abspath, dirname, join, exists
from os import makedirs
from ray import tune
from ray import init as rayInit
from ray.tune.registry import register_env
from ray.rllib.algorithms.ddpg import DDPGConfig
import warnings
warnings.filterwarnings('ignore', 'SelectableGroups dict interface')
warnings.filterwarnings('ignore', category=DeprecationWarning)

from flopyarcade import FloPyEnv


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = ArgumentParser(description='FloPyArcade training with TD3 (Ray RLlib)')
parser.add_argument('--envtype', default='3s-c', type=str,
    help='string defining environment type')
parser.add_argument('--seed', default=1, type=int,
    help='random seed for reproducibility')
parser.add_argument('--cpus', default=4, type=int,
    help='number of CPU workers')
parser.add_argument('--gpus', default=0, type=int,
    help='number of GPUs')
parser.add_argument('--timesteps', default=1000000, type=int,
    help='total timesteps for training')
parser.add_argument('--lr', default=0.00005, type=float,
    help='learning rate')
parser.add_argument('--buffer-size', default=3000000, type=int,
    help='replay buffer size')
parser.add_argument('--render', default=False, type=str_to_bool,
    help='enable rendering for benchmark visualization')
args = parser.parse_args()

ENVTYPE = args.envtype


def env_creator(env_config):
    env_config = env_config or {}
    env_config['ENVTYPE'] = ENVTYPE
    return FloPyEnv(env_config=env_config)


def main():
    if not args.render:
        rayInit()
    else:
        rayInit(log_to_driver=False, logging_level=logging.FATAL)

    register_env('flopy_td3', env_creator)

    config = (
        DDPGConfig()
        .environment('flopy_td3')
        .seed(args.seed)
        .num_workers(args.cpus - 1 if args.cpus >= 2 else 1)
        .num_gpus(args.gpus)
        .framework("tf")
        .lr(args.lr)
        .replay_buffer_config({"capacity": args.buffer_size})
        .exploration_config({"type": "GaussianNoise", "sigma": 0.1})
        .train_batch_size(256)
        .timesteps_per_iteration(10000)
        .min_iter_time_s(30)
        .model({
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        })
        .log_with({
            "tune": {
                "metrics": {
                    "min_success_rate": 0.0
                }
            }
        })
    )

    # Use TD3 variant: DDPG with target policy smoothing
    # In Ray RLlib, TD3 is available as a variant of DDPG
    config["target_network_update_freq"] = 5
    config["td3"] = True  # Enable TD3 policy smoothing if supported

    stop = {
        "timesteps_total": args.timesteps,
        "episode_reward_mean": 990,
    }

    wrkspc = abspath(dirname(__file__))
    outdir = join(wrkspc, 'results', f'{ENVTYPE}-TD3')

    results = tune.run(
        "DDPG",
        config=config.to_dict(),
        stop=stop,
        name=f'{ENVTYPE}-TD3',
        local_dir=outdir,
        verbose=2,
        checkpoint_at_end=True,
    )

    print(f'\nTraining complete. Best trial: {results.best_trial}')
    print(f'Results saved to: {outdir}')


if __name__ == '__main__':
    main()
