#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FloPyArcade — Stable-Baselines3 PPO training.

Proximal Policy Optimization: on-policy, discrete or continuous.

Requires: stable-baselines3>=2.0, flopy>=3.9.2, gymnasium
"""

from argparse import ArgumentParser
from common_sb3 import (
    add_common_args, make_flopy_env, setup_logging,
    run_evaluation, get_outdir, effective_seed, parse_net_arch
)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback,
    StopTrainingOnRewardThreshold, CallbackList
)
from os.path import join


def main():
    parser = ArgumentParser(description=__doc__)
    add_common_args(parser, default_envtype='6r-c', default_timesteps=30_000_000)
    parser.add_argument('--lr', default=5e-5, type=float,
        help='learning rate (default: 0.00005)')
    # PPO-specific
    parser.add_argument('--n-epochs', default=10, type=int,
        help='gradient descent epochs per update (default: 10)')
    parser.add_argument('--batch-size', default=256, type=int,
        help='mini-batch size (default: 256)')
    parser.add_argument('--clip-range', default=0.2, type=float,
        help='PPO clip range (default: 0.2)')
    parser.add_argument('--ent-coef', default=0.0, type=float,
        help='entropy loss coefficient (default: 0.0)')
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.quick_test:
        args.timesteps = 200
        args.eval_episodes = 1
        args.save_freq = 100

    outdir = get_outdir(args, 'PPO-SB3')
    env = make_flopy_env(args.envtype, seed_env=effective_seed(args))
    eval_env = make_flopy_env(args.envtype, seed_env=effective_seed(args))

    policy_kwargs = {}
    net_arch = parse_net_arch(args.net_arch)
    if net_arch:
        policy_kwargs['net_arch'] = net_arch

    if args.resume:
        print(f'Resuming from {args.resume}')
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            seed=args.seed,
            device='cpu',
            tensorboard_log=outdir,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
        )

    if args.eval_only:
        print(f'Eval-only from {args.eval_only}')
        model = PPO.load(args.eval_only, env=env)
        run_evaluation(model, eval_env, max_episodes=args.eval_episodes,
                       max_steps=args.max_steps_eval, verbose=args.verbose)
        return

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=outdir,
        name_prefix=f'ppo_{args.envtype}',
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

    final_path = join(outdir, 'ppo_final')
    model.save(final_path)
    print(f'\nModel saved to {final_path}')
    run_evaluation(model, eval_env, max_episodes=args.eval_episodes,
                   max_steps=args.max_steps_eval, verbose=args.verbose)


if __name__ == '__main__':
    main()
