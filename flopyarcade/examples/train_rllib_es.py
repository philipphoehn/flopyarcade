#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade - Ray RLlib ES Neuroevolution
# A11: External neuroevolution examples
# Uses Ray RLlib Evolution Strategies (ES) trainer to evolve network weights.

"""Evolve FloPyArcade neural network weights using Ray RLlib ES.

Usage
-----
    python -m flopyarcade.examples.train_rllib_es \
        --envtype 3s-d --nrow 50 --ncol 50 \
        --popsize 64 --maxiters 100 --num-gpus 1

This is a **code-only** script. Run with ``--maxiters 1`` for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    if value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FloPyArcade Ray RLlib ES neuroevolution training",
    )
    parser.add_argument("--envtype", default="3s-d", type=str,
                        help="Environment type string")
    parser.add_argument("--seedenv", default=1, type=int,
                        help="Environment seed")
    parser.add_argument("--nrow", default=50, type=int,
                        help="Grid rows")
    parser.add_argument("--ncol", default=50, type=int,
                        help="Grid columns")
    parser.add_argument("--nlay", default=2, type=int,
                        help="Number of hidden layers")
    parser.add_argument("--hidden", default=[64, 64],
                        type=lambda s: [int(x) for x in s.split(',')],
                        help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--popsize", default=64, type=int,
                        help="ES population size")
    parser.add_argument("--maxiters", default=100, type=int,
                        help="Maximum RLlib iterations")
    parser.add_argument("--num-gpus", default=0, type=int,
                        help="Number of GPUs for RLlib")
    parser.add_argument("--num-threads", default=4, type=int,
                        help="Number of CPU threads")
    parser.add_argument("--render", default=False, type=str_to_bool,
                        help="Render environment")
    parser.add_argument("--output", default=None, type=str,
                        help="Output directory")
    parser.add_argument("--nagents", default=10, type=int,
                        help="Episodes per agent evaluation")
    parser.add_argument("--use-es", default=True, type=str_to_bool,
                        help="Use ES (True) or PPO (False) RLlib trainer")
    return parser.parse_args()


def build_env_config(args) -> dict:
    """Build the Ray environment registration config."""
    return {
        "envtype": args.envtype,
        "seedenv": args.seedenv,
        "nrow": args.nrow,
        "ncol": args.ncol,
        "nlay": args.nlay,
        "hidden": args.hidden,
        "render": args.render,
        "nagents": args.nagents,
    }


def get_rl_config(args) -> dict:
    """Build the Ray RLlib training configuration."""
    obs_size = args.nrow * args.ncol
    action_size = 5

    if args.use_es:
        # Evolution Strategies (ES) — weight-level search
        config = {
            "env": "FloPyArcade",
            "env_config": build_env_config(args),
            "framework": "tf",
            "num_gpus": args.num_gpus,
            "num_cpus_per_learner_worker": 1,
            "rollouts_from_batch_size": 4096,
            "lr": 0.0,
            "train_batch_size": 1,
            "trainable": False,
            "sgd_max_batch_size": 1,
            "gradient_clipping_value": 1.0,
            "num_sgd_iter": 0,
            "use_gae": False,
            "vf_loss_coeff": 0.0,
            "kl_coeff": 0.0,
            "kl_target": 1e-4,
            "clip_rewards": False,
            "gamma": 0.99,
            "lambda": 1.0,
            "rollout_fragment_length": 128,
            "sample_async": False,
            "use_pytorch": False,
            "use_tf": True,
            "use_gpu": args.num_gpus > 0,
            "num_workers": 0,
            "num_learners": args.num_threads,
            "worker_max_concurrent_trials": args.num_threads,
            "explore": True,
            "timesteps_per_iteration": args.popsize * 128,
            "exploration_config": {
                "type": "StochasticPolicy",
                "attributes": {
                    "explorer": "GaussianExplorer",
                    "stddev": 0.1,
                },
            },
        }
    else:
        # PPO fallback for comparison
        config = {
            "env": "FloPyArcade",
            "env_config": build_env_config(args),
            "framework": "tf",
            "num_gpus": args.num_gpus,
            "num_cpus_per_learner_worker": 1,
            "lr": 0.0003,
            "train_batch_size": args.popsize * 128,
            "gradient_clipping_value": 0.5,
            "use_gae": True,
            "vf_loss_coeff": 0.5,
            "kl_coeff": 0.0,
            "kl_target": 1e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "rollout_fragment_length": 128,
            "sample_async": False,
            "use_pytorch": False,
            "use_tf": True,
            "use_gpu": args.num_gpus > 0,
            "num_workers": 0,
            "num_learners": args.num_threads,
        }

    return config


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output or 'runs/rllib_es')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RLlib-ES] envtype={args.envtype} popsize={args.popsize}")
    print(f"[RLlib-ES] maxiters={args.maxiters} gpus={args.num_gpus}")
    print(f"[RLlib-ES] use_es={args.use_es}")

    import ray

    ray.init(num_gpus=args.num_gpus, num_cpus=args.num_threads,
             ignore_reinit_error=True, log_to_driver=False)

    # Register custom environment
    from gymnasium import spaces

    class FloPyArcadeEnv:
        """Lightweight wrapper around FloPyEnv for RLlib."""
        def __init__(self, env_config=None):
            cfg = env_config or {}
            from flopyarcade import FloPyEnv
            self.env = FloPyEnv(
                envtype=cfg.get('envtype', '3s-d'),
                seedenv=cfg.get('seedenv', 1),
                nrow=cfg.get('nrow', 50),
                ncol=cfg.get('ncol', 50),
                render=False,
            )
            nrow = cfg.get('nrow', 50)
            ncol = cfg.get('ncol', 50)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(nrow * ncol,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(5)

        def reset(self, **kwargs):
            obs, _ = self.env.reset()
            return np.asarray(obs, dtype=np.float32).ravel(), {}

        def step(self, action):
            actions = ['keep', 'up', 'down', 'left', 'right']
            act_str = actions[int(action) % len(actions)]
            obs, reward, done, truncated, info = self.env.step(act_str)
            obs = np.asarray(obs, dtype=np.float32).ravel()
            return obs, float(reward), bool(done or truncated), info

        def close(self):
            self.env.close()

    ray.register_env_creator(env_name="FloPyArcade",
                             env_creator=lambda cfg: FloPyArcadeEnv(cfg))

    # Select trainer
    trainer_name = "ES" if args.use_es else "PPO"
    config = get_rl_config(args)

    print(f"[RLlib-ES] Starting {trainer_name} trainer...")
    trainer = ray.tune.get_trainer(trainer_name)
    algorithm = trainer(config)

    # Training loop
    for i in range(args.maxiters):
            result = algorithm.train()
            print(f"  iter {i}: {result}")

    # Export best policy
    checkpoint = algorithm.export_api_to_path(str(output_dir))
    print(f"[RLlib-ES] Exported to {output_dir}")

    results = {
        'trainer': trainer_name,
        'envtype': args.envtype,
        'popsize': args.popsize,
        'maxiters': args.maxiters,
        'nrow': args.nrow,
        'ncol': args.ncol,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    ray.shutdown()
    print(f"[RLlib-ES] Done. Results in {output_dir}")


if __name__ == '__main__':
    main()
