#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FloPyArcade — Stable-Baselines3 DDPG training.

Deep Deterministic Policy Gradient: off-policy, continuous actions.

Requires: stable-baselines3>=2.0, flopy>=3.9.2, gymnasium, numpy
"""

from argparse import ArgumentParser
from common import (
    add_common_args, make_flopy_env, setup_logging,
    run_evaluation, get_outdir, effective_seed, parse_net_arch
)

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback,
    StopTrainingOnRewardThreshold, CallbackList
)
import numpy as np
from os.path import join


def main():
    parser = ArgumentParser(description=__doc__)
    add_common_args(parser, default_envtype='3s-c', default_timesteps=30_000_000)
    parser.add_argument('--lr', default=5e-5, type=float,
        help='learning rate (default: 0.00005)')
    # DDPG-specific
    parser.add_argument('--tau', default=0.005, type=float,
        help='soft update coefficient (default: 0.005)')
    parser.add_argument('--buffer-size', default=3_000_000, type=int,
        help='replay buffer capacity (default: 3000000)')
    parser.add_argument('--batch-size', default=256, type=int,
        help='mini-batch size (default: 256)')
    parser.add_argument('--learning-starts', default=50_000, type=int,
        help='steps before training starts (default: 50000)')
    parser.add_argument('--noise-sigma', default=0.1, type=float,
        help='exploration noise sigma (default: 0.1)')
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.quick_test:
        args.timesteps = 200
        args.eval_episodes = 1
        args.save_freq = 100
        args.learning_starts = 100

    outdir = get_outdir(args, 'DDPG')
    env = make_flopy_env(args.envtype, seed_env=effective_seed(args))
    eval_env = make_flopy_env(args.envtype, seed_env=effective_seed(args))

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.noise_sigma * np.ones(n_actions)
    )

    policy_kwargs = {}
    net_arch = parse_net_arch(args.net_arch)
    if net_arch:
        policy_kwargs['net_arch'] = net_arch

    if args.resume:
        print(f'Resuming from {args.resume}')
        model = DDPG.load(args.resume, env=env)
    else:
        model = DDPG(
            'MlpPolicy',
            env,
            action_noise=action_noise,
            learning_rate=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            optimize_memory_usage=True,
            seed=args.seed,
            tensorboard_log=outdir,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
        )

    if args.eval_only:
        print(f'Eval-only from {args.eval_only}')
        model = DDPG.load(args.eval_only, env=env)
        run_evaluation(model, eval_env, max_episodes=args.eval_episodes,
                       max_steps=args.max_steps_eval, verbose=args.verbose)
        return

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=outdir,
        name_prefix=f'ddpg_{args.envtype}',
        save_replay_buffer=True,
    )
    stop_on_reward = StopTrainingOnRewardThreshold(
        reward_threshold=args.reward_threshold, verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=join(outdir, 'best_model'),
        log_path=join(outdir, 'results'),
        n_eval_episodes=args.eval_episodes,
        eval_freq=args.save_freq,
        callback_on_new_best=stop_on_reward,
        verbose=1,
    )
    callback = CallbackList([checkpoint_cb, eval_cb])

    model.learn(
        total_timesteps=args.timesteps,
        log_interval=10,
        callback=callback,
        reset_num_timesteps=(args.resume is None),
    )

    final_path = join(outdir, 'ddpg_final')
    model.save(final_path)
    print(f'\nModel saved to {final_path}')
    run_evaluation(model, eval_env, max_episodes=args.eval_episodes,
                   max_steps=args.max_steps_eval, verbose=args.verbose)


if __name__ == '__main__':
    main()
