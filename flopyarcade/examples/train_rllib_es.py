#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - Ray RLlib ES training
# Evolution Strategies — lightweight, GPU-friendly, no critic needed

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
from ray.rllib.algorithms.es import ESConfig
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


parser = ArgumentParser(description='FloPyArcade training with ES (Ray RLlib)')
parser.add_argument('--envtype', default='6s-c', type=str,
    help='string defining environment type')
parser.add_argument('--seed', default=1, type=int,
    help='random seed for reproducibility')
parser.add_argument('--cpus', default=16, type=int,
    help='number of CPU workers (ES benefits from many workers)')
parser.add_argument('--gpus', default=0, type=int,
    help='number of GPUs')
parser.add_argument('--timesteps', default=500000, type=int,
    help='total timesteps for training')
parser.add_argument('--sigma', default=0.2, type=float,
    help='perturbation sigma for ES')
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

    register_env('flopy_es', env_creator)

    config = (
        ESConfig()
        .environment('flopy_es')
        .seed(args.seed)
        .num_workers(args.cpus)
        .num_gpus(args.gpus)
        .framework("tf")
        .learning_rate(0.01)
        .train_batch_size(1)
        .timesteps_per_iteration(1000)
        .min_iter_time_s(10)
        .model({
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
        })
        # ES-specific
        .sigma(args.sigma)
    )

    stop = {
        "timesteps_total": args.timesteps,
        "episode_reward_mean": 990,
    }

    wrkspc = abspath(dirname(__file__))
    outdir = join(wrkspc, 'results', f'{ENVTYPE}-ES')

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
