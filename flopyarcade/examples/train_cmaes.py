#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade - CMA-ES Neuroevolution
# A11: External neuroevolution examples
# Uses the `cma` package to evolve neural network weights via CMA-ES.

"""Evolve FloPyArcade neural network weights using CMA-ES (cma package).

Usage
-----
    python -m flopyarcade.examples.train_cmaes \
        --envtype 3s-d --nrow 50 --ncol 50 \
        --popsize 60 --maxiter 100 --seed 1

This script does **not** execute long training runs by default.
Set ``--maxiter`` to a small value (e.g. 3) for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
        description="FloPyArcade CMA-ES neuroevolution training",
    )
    parser.add_argument("--envtype", default="3s-d", type=str,
                        help="Environment type string")
    parser.add_argument("--seedenv", default=1, type=int,
                        help="Environment seed")
    parser.add_argument("--seedagent", default=1, type=int,
                        help="Agent / CMA-ES seed")
    parser.add_argument("--nrow", default=50, type=int,
                        help="Grid rows")
    parser.add_argument("--ncol", default=50, type=int,
                        help="Grid columns")
    parser.add_argument("--nlay", default=2, type=int,
                        help="Number of hidden layers")
    parser.add_argument("--hidden", default=[64, 64], type=lambda s: [int(x) for x in s.split(',')],
                        help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--popsize", default=60, type=int,
                        help="CMA-ES population size")
    parser.add_argument("--maxiter", default=100, type=int,
                        help="Maximum CMA-ES iterations (generations)")
    parser.add_argument("--render", default=False, type=str_to_bool,
                        help="Render environment")
    parser.add_argument("--output", default=None, type=str,
                        help="Output directory for results")
    parser.add_argument("--nagents", default=10, type=int,
                        help="Number of agent evaluations per generation")
    return parser.parse_args()


def build_env(envtype: str, seed: int, nrow: int, ncol: int):
    """Create a FloPyArcade environment."""
    from flopyarcade import FloPyEnv
    env = FloPyEnv(
        envtype=envtype,
        seedenv=seed,
        nrow=nrow,
        ncol=ncol,
        render=False,
    )
    return env


def get_action_space_size(envtype: str) -> int:
    """Return the number of discrete actions for the environment."""
    if 'd' in envtype:
        return 5  # up, down, left, right, keep
    return 5


def build_model(nrow: int, ncol: int, nlay: int, hidden: list[int],
                action_size: int, seed: int = 42):
    """Build a small MLP for weight evolution."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.initializers import glorot_normal

    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    obs_size = nrow * ncol
    initializer = glorot_normal(seed=seed)

    model = Sequential()
    for i, h in enumerate(hidden[:nlay]):
        model.add(Dense(h, input_shape=(obs_size,),
                        kernel_initializer=initializer,
                        bias_initializer=initializer))
        model.add(Activation('relu'))
    model.add(Dense(action_size, activation='linear',
                     kernel_initializer=initializer))
    return model


def flatten_weights(model) -> np.ndarray:
    """Flatten all model weights into a 1-D vector."""
    weights = model.get_weights()
    return np.concatenate([w.ravel() for w in weights])


def set_weights_from_flat(model, flat: np.ndarray) -> None:
    """Restore model weights from a flat vector."""
    shapes = [w.shape for w in model.get_weights()]
    offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    new_weights = []
    for i, shape in enumerate(shapes):
        chunk = flat[offsets[i]:offsets[i + 1]].reshape(shape)
        new_weights.append(chunk)
    model.set_weights(new_weights)


def evaluate_agent(model, env, n_agents: int = 10) -> float:
    """Evaluate the agent by playing episodes and return mean reward."""
    rewards = []
    for _ in range(n_agents):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            obs_arr = np.asarray(obs).ravel()
            if obs_arr.size == 0:
                break
            pred = model.predict(obs_arr[np.newaxis, :], verbose=0)
            action_idx = int(np.argmax(pred[0]))
            # Map index to action string
            actions = env.actionSpace if hasattr(env, 'actionSpace') else ['keep', 'up', 'down', 'left', 'right']
            action = actions[action_idx % len(actions)]
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return float(np.mean(rewards))


def cmaes_objective(x: np.ndarray, model, env, n_agents: int) -> float:
    """CMA-ES objective: negative mean reward (minimisation)."""
    set_weights_from_flat(model, x)
    reward = evaluate_agent(model, env, n_agents)
    return -reward


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output or 'runs/cmaes')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CMA-ES] envtype={args.envtype} seed={args.seedagent}")
    print(f"[CMA-ES] popsize={args.popsize} maxiter={args.maxiter}")
    print(f"[CMA-ES] nrow={args.nrow} ncol={args.ncol} nlay={args.nlay}")

    # Build environment and model
    env = build_env(args.envtype, args.seedenv, args.nrow, args.ncol)
    action_size = get_action_space_size(args.envtype)
    model = build_model(args.nrow, args.ncol, args.nlay, args.hidden,
                        action_size, seed=args.seedagent)

    # Initial weight vector
    x0 = flatten_weights(model)
    sigma0 = 0.1  # initial step size

    print(f"[CMA-ES] weight vector size: {len(x0)}")

    # Run CMA-ES
    import cma

    result_dict = cma.fmin(
        cmaes_objective,
        x0=x0,
        sigma0=sigma0,
        args=(model, env, args.nagents),
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seedagent,
        verbose=0,
        bounds=[-5.0] * len(x0) + [5.0] * len(x0),
    )

    best_x = result_dict.xopt
    best_reward = -result_dict.fbest

    # Save results
    set_weights_from_flat(model, best_x)
    model.save(str(output_dir / 'best_model.keras'))

    results = {
        'best_reward': float(best_reward),
        'iterations': int(result.result[10] if hasattr(result, 'result') else args.maxiter),
        'popsize': args.popsize,
        'envtype': args.envtype,
        'nrow': args.nrow,
        'ncol': args.ncol,
        'weight_dim': len(x0),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[CMA-ES] Done. Best reward: {best_reward:.4f}")
    print(f"[CMA-ES] Results saved to {output_dir}")


if __name__ == '__main__':
    main()
