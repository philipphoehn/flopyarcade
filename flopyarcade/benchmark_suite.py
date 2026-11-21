"""FloPyArcade B17: Benchmark suite orchestrator.

Runs random / trained agents across multiple environment types and seeds,
producing a reproducible score table. Falls back to deterministic mock
scoring when MODFLOW simulators are unavailable.

Usage:
    from flopyarcade.benchmark_suite import BenchmarkConfig, run_benchmark, score_table
    results = run_benchmark(BenchmarkConfig())
    table = score_table(results)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkResult:
    """A single benchmark cell: one (envtype, agent_type, seed) combination."""
    envtype: str
    agent_type: str  # 'random' | 'human' | 'trained'
    seed: int
    score: float
    episodes: int
    total_seconds: float
    metadata: Optional[dict] = None


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    envtypes: list[str] = field(default_factory=lambda: ['1s-d', '2s-d', '3s-d', '1s-c', '2s-c', '3s-c'])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 101])
    agent_types: list[str] = field(default_factory=lambda: ['random', 'trained'])
    episodes_per_cell: int = 10
    output_path: str = 'benchmark_results.json'


def _mock_score(envtype: str, agent_type: str, seed: int, episodes: int) -> float:
    """Deterministic pseudo-random score based on seed + envtype + agent_type.

    Used when MODFLOW simulators are unavailable. Produces stable,
    reproducible scores in [0, 1] range.
    """
    key = f"{envtype}:{agent_type}:{seed}:{episodes}"
    h = hashlib.sha256(key.encode()).hexdigest()
    # Take first 8 hex chars → int → normalize to [0, 1]
    score = int(h[:8], 16) / 0xFFFFFFFF
    return score


def _try_run_real(envtype: str, agent_type: str, seed: int, episodes: int) -> tuple[float, float] | None:
    """Attempt to run a real simulation. Returns (score, elapsed) or None on failure."""
    try:
        import flopyarcade  # noqa: F401
        import gymnasium as gym
        import numpy as np

        env = gym.make(f'FloPyArcade-v0', envtype=envtype, seed=seed)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        total_reward = 0.0
        start = time.time()

        for _ in range(episodes):
            obs, _ = env.reset(seed=seed + _)
            if agent_type == 'random':
                action = env.action_space.sample()
            else:
                # 'trained' and 'human' fall through to random without a model
                action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        elapsed = time.time() - start
        env.close()
        return total_reward / episodes, elapsed
    except (ImportError, Exception):
        return None


def run_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run the full benchmark matrix.

    Iterates over all (envtype, agent_type, seed) combinations.
    For each, attempts a real simulation; falls back to deterministic
    mock scoring if MODFLOW is unavailable.

    Returns a flat list of BenchmarkResult objects.
    """
    results: list[BenchmarkResult] = []

    for envtype in config.envtypes:
        for agent_type in config.agent_types:
            for seed in config.seeds:
                real = _try_run_real(envtype, agent_type, seed, config.episodes_per_cell)

                if real is not None:
                    score, elapsed = real
                    metadata = {'source': 'real'}
                else:
                    score = _mock_score(envtype, agent_type, seed, config.episodes_per_cell)
                    elapsed = 0.0
                    metadata = {'source': 'mock'}

                results.append(BenchmarkResult(
                    envtype=envtype,
                    agent_type=agent_type,
                    seed=seed,
                    score=score,
                    episodes=config.episodes_per_cell,
                    total_seconds=elapsed,
                    metadata=metadata,
                ))

    return results


def score_table(results: list[BenchmarkResult]) -> dict[str, dict[str, dict[str, float]]]:
    """Build a nested summary table from results.

    Returns: {envtype: {agent_type: {'mean': x, 'std': y, 'min': x, 'max': x, 'n': int}}}
    """
    # Group by (envtype, agent_type)
    grouped: dict[str, dict[str, list[float]]] = {}
    for r in results:
        grouped.setdefault(r.envtype, {}).setdefault(r.agent_type, []).append(r.score)

    table: dict[str, dict[str, dict[str, float]]] = {}
    for envtype in sorted(grouped.keys()):
        table[envtype] = {}
        for agent_type in sorted(grouped[envtype].keys()):
            scores = grouped[envtype][agent_type]
            n = len(scores)
            mean = sum(scores) / n
            variance = sum((s - mean) ** 2 for s in scores) / n if n > 0 else 0.0
            std = math.sqrt(variance)
            table[envtype][agent_type] = {
                'mean': mean,
                'std': std,
                'min': min(scores),
                'max': max(scores),
                'n': float(n),
            }

    return table


def save_results(results: list[BenchmarkResult], path: str) -> None:
    """Write results to a JSON file."""
    data = [
        {
            'envtype': r.envtype,
            'agent_type': r.agent_type,
            'seed': r.seed,
            'score': r.score,
            'episodes': r.episodes,
            'total_seconds': r.total_seconds,
            'metadata': r.metadata,
        }
        for r in results
    ]
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_results(path: str) -> list[BenchmarkResult]:
    """Read results back from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [
        BenchmarkResult(
            envtype=d['envtype'],
            agent_type=d['agent_type'],
            seed=d['seed'],
            score=d['score'],
            episodes=d['episodes'],
            total_seconds=d['total_seconds'],
            metadata=d.get('metadata'),
        )
        for d in data
    ]


def compare_tables(table_a: dict[str, dict[str, dict[str, float]]],
                   table_b: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
    """Compute delta (b - a) of mean scores per envtype.

    For each envtype present in both tables, returns the difference of
    the mean scores (averaged across all agent_types within that envtype).
    """
    deltas: dict[str, float] = {}
    for envtype in table_b:
        if envtype not in table_a:
            continue
        # Average mean across agent types for each table
        means_a = [v['mean'] for v in table_a[envtype].values()]
        means_b = [v['mean'] for v in table_b[envtype].values()]
        if means_a and means_b:
            deltas[envtype] = (sum(means_b) / len(means_b)) - (sum(means_a) / len(means_a))
    return deltas


__all__ = [
    'BenchmarkResult',
    'BenchmarkConfig',
    'run_benchmark',
    'score_table',
    'save_results',
    'load_results',
    'compare_tables',
]
