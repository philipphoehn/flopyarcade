#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - Stable-Baselines3 PPO training
# Proximal Policy Optimization (works for both discrete and continuous)

from argparse import ArgumentParser
from os import makedirs
from os.path import abspath, dirname, join

from stable_baselines3 import PPO
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


parser = ArgumentParser(description='FloPyArcade training with PPO (Stable-Baselines3)')
parser.add_argument('--envtype', default='6r-c', type=str,
    help='string defining environment type')
parser.add_argument('--seed', default=1, type=int,
    help='random seed for reproducibility')
parser.add_argument('--timesteps', default=1000000, type=int,
    help='total timesteps for training')
parser.add_argument('--lr', default=0.00005, type=float,
    help='learning rate')
parser.add_argument('--batchsize', default=256, type=int,
    help='mini-batch size')
parser.add_argument('--nepochs', default=10, type=int,
    help='gradient descent epochs per update')
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
    outdir = join(wrkspc, 'results', f'{args.envtype}-PPO-SB3')
    makedirs(outdir, exist_ok=True)

    env = make_env(args.envtype)
    eval_env = make_env(args.envtype)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=100000,
        save_path=outdir,
        name_prefix=f'ppo_{args.envtype}'
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

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=args.lr,
        batch_size=args.batchsize,
        n_epochs=args.nepochs,
        seed=args.seed,
        device='cpu',
        tensorboard_log=outdir,
        verbose=1
    )

    model.learn(
        total_timesteps=args.timesteps,
        log_interval=10,
        callback=callback
    )

    model.save(join(outdir, 'ppo_final'))
    print(f'Model saved to {outdir}/ppo_final')

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
