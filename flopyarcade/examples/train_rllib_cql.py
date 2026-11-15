#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - Ray RLlib CQL training
# Conservative Q-Learning for discrete/continuous actions
# Reduces overestimation bias in offline/online RL

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


parser = ArgumentParser(description='FloPyArcade training with CQL (Ray RLlib)')
parser.add_argument('--envtype', default='6s-c', type=str,
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
parser.add_argument('--cql-temp', default=1.0, type=float,
    help='CQL temperature parameter')
parser.add_argument('--cql-alpha', default=1.0, type=float,
    help='CQL penalty coefficient')
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

    register_env('flopy_cql', env_creator)

    config = {
        "env": "flopy_cql",
        "seed": args.seed,
        "num_workers": args.cpus - 1 if args.cpus >= 2 else 1,
        "num_gpus": args.gpus,
        "framework": "tf",
        "lr": args.lr,
        "train_batch_size": 256,
        "timesteps_per_iteration": 10000,
        "min_iter_time_s": 30,
        "buffer_size": args.buffer_size,
        "learning_starts": 50000,
        "target_network_update_freq": 10000,
        "model": {
            "fcnet_hiddens": [512, 256],
            "fcnet_activation": "relu",
        },
        # CQL-specific parameters
        "cql_temp": args.cql_temp,
        "cql_alpha": args.cql_alpha,
        "cql_num_actions": 10,
        "exploration_config": {"type": "EpsilonGreedy", "epsilon": 0.05},
    }

    stop = {
        "timesteps_total": args.timesteps,
        "episode_reward_mean": 990,
    }

    wrkspc = abspath(dirname(__file__))
    outdir = join(wrkspc, 'results', f'{ENVTYPE}-CQL')

    results = tune.run(
        "DQN",  # CQL built on DQN in RLlib
        config=config,
        stop=stop,
        name=f'{ENVTYPE}-CQL',
        local_dir=outdir,
        verbose=2,
        checkpoint_at_end=True,
    )

    print(f'\nTraining complete. Best trial: {results.best_trial}')
    print(f'Results saved to: {outdir}')


if __name__ == '__main__':
    main()
