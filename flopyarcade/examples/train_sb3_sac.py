#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - Stable-Baselines3 SAC training
# Soft Actor-Critic for continuous action spaces

from argparse import ArgumentParser
from os import makedirs
from os.path import abspath, dirname, join, exists

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback,
    StopTrainingOnRewardThreshold, CallbackList
)

from flopyarcade import FloPyEnv


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = ArgumentParser(description='FloPyArcade training with SAC (Stable-Baselines3)')
parser.add_argument('--envtype', default='3s-c', type=str,
    help='string defining environment type')
parser.add_argument('--seed', default=1, type=int,
    help='random seed for reproducibility')
parser.add_argument('--timesteps', default=1000000, type=int,
    help='total timesteps for training')
parser.add_argument('--lr', default=0.0003, type=float,
    help='learning rate')
parser.add_argument('--batchsize', default=256, type=int,
    help='mini-batch size')
parser.add_argument('--buffer-size', default=1000000, type=int,
    help='replay buffer size')
parser.add_argument('--render', default=False, type=str_to_bool,
    help='enable rendering during training')
args = parser.parse_args()


def make_env(envtype, render=False):
    """Create a FloPyArcade environment."""
    env_config = {'ENVTYPE': envtype}
    if render:
        env_config['flagRender'] = True
    return FloPyEnv(env_config=env_config)


def main():
    wrkspc = abspath(dirname(__file__))
    outdir = join(wrkspc, 'results', f'{args.envtype}-SAC')
    makedirs(outdir, exist_ok=True)

    env = make_env(args.envtype)
    eval_env = make_env(args.envtype)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=100000,
        save_path=outdir,
        name_prefix=f'sac_{args.envtype}'
    )
    stop_on_reward = StopTrainingOnRewardThreshold(reward_threshold=990, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=join(outdir, 'best_model'),
        log_path=join(outdir, 'results'),
        n_eval_episodes=5,
        eval_freq=100000,
        callback_on_new_best=stop_on_reward,
        verbose=1
    )
    callback = CallbackList([checkpoint_cb, eval_cb])

    model = SAC(
        'MlpPolicy',
        env,
        action_noise=action_noise,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batchsize,
        seed=args.seed,
        tensorboard_log=outdir,
        verbose=1
    )

    model.learn(
        total_timesteps=args.timesteps,
        log_interval=10,
        callback=callback
    )

    model.save(join(outdir, 'sac_final'))
    print(f'Model saved to {outdir}/sac_final')

    # Quick evaluation
    obs, _ = eval_env.reset()
    done = False
    reward_total = 0.
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = eval_env.step(action)
        reward_total += reward
    print(f'Final evaluation reward: {reward_total:.2f}')


if __name__ == '__main__':
    main()
