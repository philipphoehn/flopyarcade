"""FloPyArcade autograder benchmark.

Loads a trained policy model, runs N games in a FloPyArcade environment,
and reports reward statistics (mean, std, min, max, median, % nonzero).

Usage:
    python -m flopyarcade.benchmark --model model_1s-d.p --envtype 1s-d --games 1000
    flopyarcade-benchmark --model model_3s-d.p --envtype 3s-d --games 1000 --seed 42
"""

import argparse
import json
import os
import time
from typing import Optional


def load_keras_agent(model_path: str):
    """Load a Keras model from .keras, .h5, or RLlib pickle .p file.

    For pickle .p files (RLlib policy weights), reconstructs a Keras Sequential
    model from the weight matrices: fc_1 -> fc_out -> dense (policy head).
    Works for both discrete (3 output) and continuous (N output) action spaces.
    """
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import Dense, Activation

    if model_path.endswith('.p') or model_path.endswith('.pickle'):
        import pickle
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and 'default_policy' in obj:
            policy = obj['default_policy']
            if hasattr(policy, 'predict_on_batch'):
                return policy

            # Reconstruct Keras model from RLlib weight dict
            prefix = 'default_policy/'
            w1 = policy.get(f'{prefix}fc_1/kernel')
            b1 = policy.get(f'{prefix}fc_1/bias')
            w2 = policy.get(f'{prefix}fc_out/kernel')
            b2 = policy.get(f'{prefix}fc_out/bias')
            w3 = policy.get(f'{prefix}dense/kernel')
            b3 = policy.get(f'{prefix}dense/bias')

            if None in (w1, b1, w2, b2, w3, b3):
                # Try alternative layer names
                w1 = policy.get(f'{prefix}fc_1_1/kernel') or policy.get(f'{prefix}hidden_0/kernel') or w1
                b1 = policy.get(f'{prefix}fc_1_1/bias') or policy.get(f'{prefix}hidden_0/bias') or b1
                w2 = policy.get(f'{prefix}fc_out_1/kernel') or w2
                b2 = policy.get(f'{prefix}fc_out_1/bias') or b2

            if w1 is not None and w2 is not None and w3 is not None:
                h1, h2, n_out = w1.shape[1], w2.shape[1], w3.shape[1]
                in_dim = w1.shape[0]
                model = Sequential([
                    Dense(h1, input_shape=(in_dim,)),
                    Activation('relu'),
                    Dense(h2),
                    Activation('relu'),
                    Dense(n_out, activation='linear'),
                ])
                model.layers[0].set_weights([w1, b1])
                model.layers[2].set_weights([w2, b2])
                model.layers[4].set_weights([w3, b3])
                return model

        raise ValueError(f'Cannot extract Keras model from pickle: {model_path}')

    return load_model(model_path, compile=False)


def run_benchmark(
    model_path: str,
    envtype: str = '1s-d',
    n_games: int = 1000,
    seed: int = 42,
    n_steps: int = 200,
    verbose: bool = True,
    output_path: Optional[str] = None,
) -> dict:
    """Run N games with a loaded model and return reward statistics.

    Args:
        model_path: Path to model file (.keras, .h5, .p).
        envtype: FloPyArcade environment type (e.g. 1s-d, 2s-d, 3s-d).
        n_games: Number of games to play.
        seed: Base random seed (game i uses seed+i).
        n_steps: Max steps per game.
        verbose: Print progress every 5%.
        output_path: Optional path to save JSON results.

    Returns:
        dict with statistics: mean, std, min, max, median, pct_nonzero,
        avg_runtime_s, total_runtime_s, per-game rewards.
    """
    import numpy as np
    from flopyarcade import FloPyArcade, FloPyAgent

    if verbose:
        FloPyAgent.suppressTensorFlowWarnings(FloPyAgent)

    if verbose:
        print(f'\n===== BENCHMARK {envtype} | model={os.path.basename(model_path)} | {n_games} games =====')

    model = load_keras_agent(model_path)
    arcade = FloPyArcade(agent=model, ENVTYPE=envtype, flagRender=False, NAGENTSTEPS=n_steps)

    rewards = []
    timestamps = []
    t_start = time.time()

    report_every = max(1, n_games // 20)  # report every 5%

    for i in range(n_games):
        t0 = time.time()
        r = arcade.play(ENVTYPE=envtype, seed=seed + i, returnReward=True, verbose=False)
        rewards.append(r if r is not None else 0.0)
        timestamps.append(time.time() - t0)

        if verbose and (i + 1) % report_every == 0:
            elapsed = time.time() - t_start
            avg = elapsed / (i + 1)
            eta = avg * (n_games - i - 1)
            print(f'  [{i+1:4d}/{n_games}] mean_so_far={np.mean(rewards):.1f} ' 
                  f'elapsed={elapsed:.0f}s ETA={eta:.0f}s', flush=True)

    total = time.time() - t_start
    arr = np.array(rewards, dtype=float)

    results = {
        'envtype': envtype,
        'model': os.path.basename(model_path),
        'n_games': n_games,
        'seed_base': seed,
        'n_steps': n_steps,
        'mean_reward': round(float(arr.mean()), 2),
        'std_reward': round(float(arr.std()), 2),
        'min_reward': round(float(arr.min()), 2),
        'max_reward': round(float(arr.max()), 2),
        'median_reward': round(float(np.median(arr)), 2),
        'pct_nonzero': round(float(np.mean(arr != 0)) * 100, 1),
        'avg_runtime_s': round(float(np.mean(timestamps)), 3),
        'total_runtime_s': round(total, 1),
    }

    if verbose:
        print(f'  DONE: mean={results["mean_reward"]} std={results["std_reward"]} ' 
              f'max={results["max_reward"]} time={results["total_runtime_s"]}s', flush=True)

    if output_path:
        out = dict(results)
        out['per_game_rewards'] = [round(float(r), 4) for r in rewards]
        with open(output_path, 'w') as f:
            json.dump(out, f, indent=2)
        if verbose:
            print(f'  Results saved to {output_path}')

    return results


def main():
    parser = argparse.ArgumentParser(description='FloPyArcade autograder benchmark')
    parser.add_argument('--model', required=True, type=str,
                        help='Path to model file (.keras, .h5, .p) or RLlib checkpoint')
    parser.add_argument('--envtype', default='1s-d', type=str,
                        help='Environment type (e.g. 1s-d, 2s-d, 3s-d, 6r-c)')
    parser.add_argument('--games', default=1000, type=int,
                        help='Number of games to run (default: 1000)')
    parser.add_argument('--seed', default=42, type=int,
                        help='Base random seed (default: 42)')
    parser.add_argument('--steps', default=200, type=int,
                        help='Max steps per game (default: 200)')
    parser.add_argument('--output', default=None, type=str,
                        help='Path to save JSON results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    args = parser.parse_args()

    results = run_benchmark(
        model_path=args.model,
        envtype=args.envtype,
        n_games=args.games,
        seed=args.seed,
        n_steps=args.steps,
        verbose=not args.quiet,
        output_path=args.output,
    )

    # Print final JSON summary to stdout (for CI capture)
    print('\n=== RESULTS JSON ===')
    print(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    main()
