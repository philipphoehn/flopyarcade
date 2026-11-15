#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - Ray RLlib A3C training
# Asynchronous Advantage Actor-Critic for discrete actions

import os
import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser
from os.path import abspath, dirname, join
from os import makedirs
from ray import tune
from ray import init as rayInit
from ray.tune.registry import register_env
from ray.rllib.algorithms.a3c import A3CConfig
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


parser = ArgumentParser(description='FloPyArcade training with A3C (Ray RLlib)')
parser.add_argument('--envtype', default='6s-c', type=str,
    help='string defining environment type')
parser.add_argument('--seed', default=1, type=int,
    help='random seed for reproducibility')
parser.add_argument('--cpus', default=8, type=int,
    help='number of CPU workers')
parser.add_argument('--gpus', default=0, type=int,
    help='number of GPUs')
parser.add_argument('--timesteps', default=1000000, type=int,
    help='total timesteps for training')
parser.add_argument('--lr', default=0.00005, type=float,
    help='learning rate')
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

    register_env('flopy_a3c', env_creator)

    config = (
        A3CConfig()
        .environment('flopy_a3c')
        .seed(args.seed)
        .num_workers(args.cpus)
        .num_gpus(args.gpus)
        .framework("tf")
        .lr(args.lr)
        .train_batch_size(512)
        .rollout_fragment_length(50)
        .timesteps_per_iteration(10000)
        .min_iter_time_s(30)
        .model({
            "fcnet_hiddens": [512, 256],
            "fcnet_activation": "relu",
        })
    )

    stop = {
        "timesteps_total": args.timesteps,
        "episode_reward_mean": 990,
    }

    wrkspc = abspath(dirname(__file__))
    outdir = join(wrkspc, 'results', f'{ENVTYPE}-A3C')

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
