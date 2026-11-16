#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FloPyArcade — Stable-Baselines3 SAC training.

Soft Actor-Critic: off-policy, continuous actions, entropy-regularized.

Requires: stable-baselines3>=2.0, flopy>=3.9.2, gymnasium, numpy
"""

from argparse import ArgumentParser
from common_sb3 import (
    add_common_args, make_flopy_env, setup_logging,
    run_evaluation, get_outdir, effective_seed, parse_net_arch
)

from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback,
    StopTrainingOnRewardThreshold, CallbackList
)
import numpy as np
from os.path import join


def main():
    parser = ArgumentParser(description=__doc__)
    add_common_args(parser, default_envtype='3s-c')
    # SAC-specific
    parser.add_argument('--tau', default=0.005, type=float,
        help='soft update coefficient for target networks (default: 0.005)')
    parser.add_argument('--n-critics', default=2, type=int,
        help='number of critics (default: 2)')
    parser.add_argument('--train-freq', default=1, type=int,
        help='train N gradient steps per env step (default: 1)')
    parser.add_argument('--buffer-size', default=1_000_000, type=int,
        help='replay buffer capacity (default: 1000000)')
    parser.add_argument('--batch-size', default=256, type=int,
        help='mini-batch size (default: 256)')
    parser.add_argument('--noise-sigma', default=0.1, type=float,
        help='exploration noise sigma (default: 0.1)')
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Quick test mode
    if args.quick_test:
        args.timesteps = 200
        args.eval_episodes = 1
        args.save_freq = 100

    outdir = get_outdir(args, 'SAC')
    seed = args.seed
    env_seed = effective_seed(args)

    env = make_flopy_env(args.envtype, seed_env=env_seed)
    eval_env = make_flopy_env(args.envtype, seed_env=env_seed)

    # Noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.noise_sigma * np.ones(n_actions)
    )

    # Build policy_kwargs
    policy_kwargs = {}
    net_arch = parse_net_arch(args.net_arch)
    if net_arch:
        policy_kwargs['net_arch'] = net_arch
    if args.n_critics != 2:
        policy_kwargs['n_critics'] = args.n_critics

    # Resume or fresh
    if args.resume:
        print(f'Resuming from {args.resume}')
        model = SAC.load(args.resume, env=env)
        model.noise = action_noise
    else:
        model = SAC(
            'MlpPolicy',
            env,
            action_noise=action_noise,
            learning_rate=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            train_freq=(args.train_freq, 'steps'),
            seed=seed,
            tensorboard_log=outdir,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
        )

    # Eval-only mode
    if args.eval_only:
        print(f'Eval-only from {args.eval_only}')
        model = SAC.load(args.eval_only, env=env)
        run_evaluation(model, eval_env, max_episodes=args.eval_episodes,
                       max_steps=args.max_steps_eval, verbose=args.verbose)
        return

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=outdir,
        name_prefix=f'sac_{args.envtype}',
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

    # Train
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=10,
        callback=callback,
        reset_num_timesteps=(args.resume is None),
    )

    # Save final
    final_path = join(outdir, 'sac_final')
    model.save(final_path)
    print(f'\nModel saved to {final_path}')

    # Final evaluation
    run_evaluation(model, eval_env, max_episodes=args.eval_episodes,
                   max_steps=args.max_steps_eval, verbose=args.verbose)


if __name__ == '__main__':
    main()
