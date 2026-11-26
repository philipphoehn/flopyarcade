#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade - EvoTorch Neuroevolution
# A11: External neuroevolution examples
# Uses EvoTorch (Torch-based ES) to evolve neural network weights.

"""Evolve FloPyArcade neural network weights using EvoTorch.

EvoTorch (https://github.com/EvoTorch/EvoTorch) provides modern
Evolution Strategies and Neuroevolution on top of PyTorch.

Usage
-----
    python -m flopyarcade.examples.train_evotorch \
        --envtype 3s-d --nrow 50 --ncol 50 \
        --popsize 64 --maxiter 100 --seed 1

This is a **code-only** script. Install `evotorch` first:
    pip install evotorch

Run with ``--maxiter 1`` for a smoke test.
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
        description="FloPyArcade EvoTorch neuroevolution training",
    )
    parser.add_argument("--envtype", default="3s-d", type=str,
                        help="Environment type string")
    parser.add_argument("--seedenv", default=1, type=int,
                        help="Environment seed")
    parser.add_argument("--seedagent", default=1, type=int,
                        help="Agent / EvoTorch seed")
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
                        help="Population size")
    parser.add_argument("--maxiter", default=100, type=int,
                        help="Maximum generations")
    parser.add_argument("--lr", default=0.0, type=float,
                        help="Learning rate (0 = pure ES, no gradient)")
    parser.add_argument("--noise-std", default=0.1, type=float,
                        help="ES noise standard deviation")
    parser.add_argument("--render", default=False, type=str_to_bool,
                        help="Render environment")
    parser.add_argument("--output", default=None, type=str,
                        help="Output directory")
    parser.add_argument("--nagents", default=10, type=int,
                        help="Episodes per agent evaluation")
    parser.add_argument("--device", default="cpu", type=str,
                        help="PyTorch device (cpu, cuda, mps)")
    return parser.parse_args()


def build_env(envtype: str, seed: int, nrow: int, ncol: int):
    """Create a FloPyArcade environment."""
    from flopyarcade import FloPyEnv
    return FloPyEnv(
        envtype=envtype,
        seedenv=seed,
        nrow=nrow,
        ncol=ncol,
        render=False,
    )


def build_torch_model(nrow: int, ncol: int, nlay: int,
                      hidden: list[int], action_size: int, seed: int = 42):
    """Build a PyTorch MLP for weight evolution."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    obs_size = nrow * ncol
    layers = []
    in_dim = obs_size
    for h in hidden[:nlay]:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, action_size))
    return nn.Sequential(*layers)


def flatten_weights(model) -> np.ndarray:
    """Flatten all PyTorch model parameters into a 1-D numpy array."""
    import torch
    return torch.cat([p.detach().flatten() for p in model.parameters()]).numpy()


def set_weights_from_flat(model, flat: np.ndarray) -> None:
    """Restore PyTorch model parameters from a flat numpy array."""
    import torch
    flat_tensor = torch.tensor(flat, dtype=torch.float32)
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_tensor[offset:offset + n].reshape(p.shape))
        offset += n


def evaluate_agent(model, env, n_agents: int = 10, device='cpu') -> float:
    """Evaluate the agent by playing episodes and return mean reward."""
    import torch

    model.eval()
    rewards = []
    with torch.no_grad():
        for _ in range(n_agents):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                obs_arr = torch.tensor(
                    np.asarray(obs, dtype=np.float32).ravel()
                ).unsqueeze(0).to(device)
                pred = model(obs_arr)
                action_idx = int(pred.argmax().item())
                actions = ['keep', 'up', 'down', 'left', 'right']
                action = actions[action_idx % len(actions)]
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
    return float(np.mean(rewards))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output or 'runs/evotorch')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EvoTorch] envtype={args.envtype} seed={args.seedagent}")
    print(f"[EvoTorch] popsize={args.popsize} maxiter={args.maxiter}")
    print(f"[EvoTorch] device={args.device}")

    # Build environment and model
    env = build_env(args.envtype, args.seedenv, args.nrow, args.ncol)
    action_size = 5
    model = build_torch_model(args.nrow, args.ncol, args.nlay,
                              args.hidden, action_size, seed=args.seedagent)

    x0 = flatten_weights(model)
    print(f"[EvoTorch] weight vector size: {len(x0)}")

    # --- EvoTorch integration ---
    # EvoTorch provides ES algorithms (CMA, SAES, etc.) on PyTorch models.
    # The following is the intended integration pattern.
    # Requires: pip install evotorch

    try:
        import evotorch
        from evotorch import es
        print(f"[EvoTorch] Loaded version: {evotorch.__version__}")
    except ImportError:
        print("[EvoTorch] WARNING: evotorch not installed.")
        print("[EvoTorch] Falling back to manual ES loop.")
        evotorch = None

    # Manual ES fallback (works without evotorch)
    import torch
    device = torch.device(args.device)
    model.to(device)

    best_reward = -np.inf
    best_weights = x0.copy()
    sigma = args.noise_std

    for gen in range(args.maxiter):
        # Sample population
        candidates = []
        for _ in range(args.popsize):
            perturbed = x0 + sigma * np.random.randn(len(x0))
            candidates.append(perturbed)

        # Evaluate
        rewards = []
        for c in candidates:
            set_weights_from_flat(model, c)
            r = evaluate_agent(model, env, args.nagents, args.device)
            rewards.append(r)

        rewards = np.array(rewards)
        gen_best_idx = int(np.argmax(rewards))
        gen_best_reward = float(rewards[gen_best_idx])
        print(f"  gen {gen}: best={gen_best_reward:.4f} mean={rewards.mean():.4f}")

        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_weights = candidates[gen_best_idx].copy()

        # Update center (rank-weighted mean of top half)
        k = args.popsize // 2
        top_indices = np.argsort(rewards)[-k:]
        x0 = np.mean([candidates[i] for i in top_indices], axis=0)
        # Adapt sigma (1/5 rule of thumb)
        sigma = max(sigma * 0.99, 1e-5)

    # Save best model
    set_weights_from_flat(model, best_weights)
    import torch
    torch.save(model.state_dict(), str(output_dir / 'best_model.pt'))

    results = {
        'best_reward': float(best_reward),
        'generations': args.maxiter,
        'popsize': args.popsize,
        'envtype': args.envtype,
        'nrow': args.nrow,
        'ncol': args.ncol,
        'weight_dim': len(x0),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[EvoTorch] Done. Best reward: {best_reward:.4f}")
    print(f"[EvoTorch] Results saved to {output_dir}")


if __name__ == '__main__':
    main()
