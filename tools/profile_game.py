#!/usr/bin/env python3
"""Standalone cProfile wrapper for FloPyArcade game loop.

Usage:
    python tools/profile_game.py --envtype 1s-d --episodes 10 --output profile_results.html

This does NOT modify the game code. It wraps the play loop externally
and reports the top-N hotspots.
"""

import argparse
import cProfile
import pstats
import sys
import io
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Profile FloPyArcade game loop")
    p.add_argument("--envtype", default="1s-d", help="Environment type (default: 1s-d)")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes (default: 10)")
    p.add_argument("--steps", type=int, default=100, help="Max steps per episode (default: 100)")
    p.add_argument("--output", default="profile_results.html", help="Output HTML file")
    p.add_argument("--sort", default="cumulative", help="Sort key (cumulative, tottime, ncalls)")
    p.add_argument("--top", type=int, default=30, help="Top N functions to show (default: 30)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def run_game_loop(envtype, n_episodes, max_steps, seed):
    """Run the actual game loop to profile."""
    from flopyarcade import FloPyArcade
    import numpy as np

    total_reward = 0.0
    for ep in range(n_episodes):
        game = FloPyArcade(ENVTYPE=envtype, flagRender=False, NAGENTSTEPS=max_steps)
        obs = game.reset(seed=seed + ep)
        for step in range(max_steps):
            action = game.action_space.sample()
            obs, reward, done, truncated, info = game.step(action)
            total_reward += reward
            if done or truncated:
                break
        game.close()
    return total_reward


def main():
    args = parse_args()
    print(f"Profiling: {args.envtype}, {args.episodes} episodes, {args.steps} steps max")
    print(f"Output: {args.output}")

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    total_reward = run_game_loop(args.envtype, args.episodes, args.steps, args.seed)
    profiler.disable()
    elapsed = time.time() - t0

    # Print top functions to stdout
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(args.sort)
    stats.print_stats(args.top)
    print(stream.getvalue())

    # Save HTML
    profiler.dump_stats(args.output + ".stats")
    # Simple HTML report
    html = f"""<html><body>
<h1>FloPyArcade Profile: {args.envtype}</h1>
<p>Episodes: {args.episodes} | Max steps: {args.steps} | Wall time: {elapsed:.2f}s | Total reward: {total_reward:.2f}</p>
<h2>Top {args.top} functions (by {args.sort})</h2>
<pre>{stream.getvalue()}</pre>
</body></html>"""
    Path(args.output).write_text(html)
    print(f"\nHTML report: {args.output}")
    print(f"Stats file:  {args.output}.stats (open with snakeviz: snakeviz {args.output}.stats)")


if __name__ == "__main__":
    main()
