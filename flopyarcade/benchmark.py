"""FloPyArcade autograder benchmark.

Loads a trained policy model, runs N games in a FloPyArcade environment,
and reports reward statistics (mean, std, min, max, median, % nonzero).

Supports:
  - Keras models (.keras, .h5, .model)
  - RLlib pickle policies (.p, .pickle)
  - RLlib checkpoint directories (containing .tune_metadata)
  - Discrete action spaces (1s-d, 2s-d, 3s-d)
  - Continuous action spaces (6r-c, etc.)

Usage:
    python -m flopyarcade.benchmark --model model_1s-d.p --envtype 1s-d --games 1000
    flopyarcade-benchmark --model model_3s-d.p --envtype 3s-d --games 1000 --seed 42
    flopyarcade-benchmark --model ./checkpoint_dir --envtype 6r-c --games 100
"""

import argparse
import json
import os
import time
from typing import Optional


def _try_layer_get(policy: dict, prefix: str, name: str) -> tuple:
    """Try multiple naming patterns for a layer's kernel and bias."""
    candidates = [
        f'{prefix}{name}/kernel',
        f'{prefix}{name}/bias',
        f'{prefix}{name}_1/kernel',
        f'{prefix}{name}_1/bias',
        f'{prefix}{name.replace("_", "")}/kernel',
        f'{prefix}{name.replace("_", "")}/bias',
    ]
    for i in range(0, len(candidates), 2):
        k = policy.get(candidates[i])
        b = policy.get(candidates[i + 1])
        if k is not None and b is not None:
            return k, b
    # Last resort: search all keys
    k = b = None
    for key in policy:
        if name in key and 'kernel' in key:
            k = policy[key]
        if name in key and 'bias' in key:
            b = policy[key]
    return k, b


def load_keras_agent(model_path: str):
    """Load a policy model from various formats.

    Supports:
      - .keras / .h5 / .model files (Keras native)
      - .p / .pickle files (RLlib policy weight dicts)
      - Directories containing .tune_metadata (RLlib checkpoint dirs)

    Returns a Keras model (or RLlib policy object with predict_on_batch).
    """
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import Dense, Activation
    import numpy as np

    # --- Case 1: RLlib checkpoint directory ---
    if os.path.isdir(model_path):
        tune_meta = os.path.join(model_path, '.tune_metadata')
        if os.path.exists(tune_meta):
            return _load_rllib_checkpoint(model_path, tune_meta)
        # Maybe a directory with a .keras file inside
        keras_files = [f for f in os.listdir(model_path)
                       if f.endswith(('.keras', '.h5', '.model'))]
        if keras_files:
            return load_model(os.path.join(model_path, keras_files[0]), compile=False)
        raise FileNotFoundError(
            f'No recognized model in directory {model_path}. '
            f'Expected .keras, .h5, .model, or .tune_metadata.')

    # --- Case 2: RLlib pickle .p / .pickle ---
    if model_path.endswith('.p') or model_path.endswith('.pickle'):
        return _load_rllib_pickle(model_path)

    # --- Case 3: Keras native ---
    return load_model(model_path, compile=False)


def _load_rllib_pickle(model_path: str):
    """Reconstruct a Keras model from an RLlib pickle policy weight dict."""
    import pickle
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation

    with open(model_path, 'rb') as f:
        obj = pickle.load(f)

    # If it's a live RLlib policy object, use it directly
    if hasattr(obj, 'predict_on_batch'):
        return obj

    if not isinstance(obj, dict):
        raise TypeError(f'Expected dict or policy object, got {type(obj)}')

    # Find the policy weights dict
    policy = None
    if 'default_policy' in obj:
        policy = obj['default_policy']
    elif 'policy' in obj:
        policy = obj['policy']
    else:
        # Maybe the dict IS the weight dict
        if any('kernel' in k for k in obj.keys()):
            policy = obj

    if policy is None or not isinstance(policy, dict):
        raise ValueError(f'Cannot find policy weights in {model_path}')

    # Determine prefix (common patterns)
    prefix = ''
    sample_key = next(iter(policy), '')
    if '/' in sample_key:
        prefix = sample_key.split('/')[0] + '/'
    elif 'default_policy' in sample_key:
        prefix = 'default_policy/'

    # Find layers: look for fc_1, fc_out, dense (or hidden_0, hidden_1, output)
    # Try standard RLlib DQN/PPO naming
    w1, b1 = _try_layer_get(policy, prefix, 'fc_1')
    w2, b2 = _try_layer_get(policy, prefix, 'fc_out')
    w3, b3 = _try_layer_get(policy, prefix, 'dense')

    # Alternative: hidden_0, hidden_1, output
    if w1 is None:
        w1, b1 = _try_layer_get(policy, prefix, 'hidden_0')
    if w2 is None:
        w2, b2 = _try_layer_get(policy, prefix, 'hidden_1')
    if w3 is None:
        w3, b3 = _try_layer_get(policy, prefix, 'output')

    # Alternative: single hidden layer
    if w1 is None:
        # Find any layer with 'kernel' in name
        kernel_keys = sorted([k for k in policy if 'kernel' in k.lower()])
        bias_keys = sorted([k for k in policy if 'bias' in k.lower() and 'kernel' not in k.lower()])
        if len(kernel_keys) >= 2:
            w1 = policy[kernel_keys[0]]
            b1 = policy[bias_keys[0]] if bias_keys else None
            w2 = policy[kernel_keys[1]]
            b2 = policy[bias_keys[1]] if len(bias_keys) > 1 else None
            if len(kernel_keys) >= 3:
                w3 = policy[kernel_keys[2]]
                b3 = policy[bias_keys[2]] if len(bias_keys) > 2 else None
            else:
                w3, b3 = w2, b2
                w2, b2 = w1, b1
                # Two-layer network: w1->w3
                w3, b3 = w2, b2

    if w1 is None or w3 is None:
        raise ValueError(
            f'Cannot reconstruct model from pickle. Keys found: '
            f'{list(policy.keys())[:20]}')

    # Build model based on number of layers
    in_dim = w1.shape[0]
    h1 = w1.shape[1]

    if w2 is not None and w2 is not w1 and w2.shape[0] == h1:
        h2 = w2.shape[1]
        n_out = w3.shape[1]
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
    else:
        # Two-layer: w1 -> w3
        n_out = w3.shape[1]
        model = Sequential([
            Dense(h1, input_shape=(in_dim,)),
            Activation('relu'),
            Dense(n_out, activation='linear'),
        ])
        model.layers[0].set_weights([w1, b1])
        model.layers[2].set_weights([w3, b3])

    return model


def _load_rllib_checkpoint(checkpoint_dir: str, tune_meta_path: str):
    """Load an RLlib checkpoint directory."""
    import pickle
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    import numpy as np

    with open(tune_meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Find the policy weights in the checkpoint
    # RLlib checkpoints store weights in various locations
    # Try to find .p or .npz files
    for fname in os.listdir(checkpoint_dir):
        fpath = os.path.join(checkpoint_dir, fname)
        if fname.endswith('.p') or fname.endswith('.pickle'):
            return _load_rllib_pickle(fpath)
        if fname.endswith('.npz'):
            return _load_npz_policy(fpath)

    # Try the meta object itself
    if hasattr(meta, 'predict_on_batch'):
        return meta

    raise ValueError(
        f'Could not find policy weights in checkpoint {checkpoint_dir}. '
        f'Contents: {os.listdir(checkpoint_dir)[:20]}')


def _load_npz_policy(npz_path: str):
    """Load a .npz policy file (alternative RLlib format)."""
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation

    data = np.load(npz_path, allow_pickle=True)
    # Try to reconstruct from arrays
    keys = list(data.keys())
    # Look for kernel/bias patterns
    kernels = {k: data[k] for k in keys if 'kernel' in k.lower()}
    biases = {k: data[k] for k in keys if 'bias' in k.lower() and 'kernel' not in k.lower()}

    if len(kernels) >= 2:
        w1 = list(kernels.values())[0]
        b1 = list(biases.values())[0] if biases else None
        w3 = list(kernels.values())[-1]
        b3 = list(biases.values())[-1] if biases else None
        in_dim, h1 = w1.shape[0], w1.shape[1]
        n_out = w3.shape[1]
        model = Sequential([
            Dense(h1, input_shape=(in_dim,)),
            Activation('relu'),
            Dense(n_out, activation='linear'),
        ])
        model.layers[0].set_weights([w1, b1])
        model.layers[2].set_weights([w3, b3])
        return model

    raise ValueError(f'Cannot reconstruct model from {npz_path}')


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
        model_path: Path to model file (.keras, .h5, .p) or checkpoint dir.
        envtype: FloPyArcade environment type (e.g. 1s-d, 2s-d, 3s-d, 6r-c).
        n_games: Number of games to play.
        seed: Base random seed (game i uses seed+i).
        n_steps: Max steps per game.
        verbose: Print progress every 5%.
        output_path: Optional path to save JSON results.

    Returns:
        dict with statistics: mean, std, min, max, median, pct_nonzero,
        avg_runtime_s, total_runtime_s.
    """
    import numpy as np
    from flopyarcade import FloPyArcade, FloPyAgent

    if verbose:
        FloPyAgent.suppressTensorFlowWarnings(FloPyAgent)

    if verbose:
        print(f'\n===== BENCHMARK {envtype} | model={os.path.basename(model_path)} | {n_games} games =====', flush=True)

    model = load_keras_agent(model_path)
    if verbose:
        print(f'  Model loaded: {type(model).__name__}', flush=True)

    arcade = FloPyArcade(agent=model, ENVTYPE=envtype, flagRender=False, NAGENTSTEPS=n_steps)

    rewards = []
    timestamps = []
    t_start = time.time()

    report_every = max(1, n_games // 20)  # report every 5%

    for i in range(n_games):
        t0 = time.time()
        try:
            r = arcade.play(ENVTYPE=envtype, seed=seed + i, returnReward=True, verbose=False)
        except Exception as e:
            if verbose and i < 5:
                print(f'  [WARN] Game {i} failed: {e}', flush=True)
            r = 0.0
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
        'model': os.path.basename(model_path) if not os.path.isdir(model_path) else model_path,
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
            print(f'  Results saved to {output_path}', flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description='FloPyArcade autograder benchmark')
    parser.add_argument('--model', required=True, type=str,
                        help='Path to model file (.keras, .h5, .p) or RLlib checkpoint dir')
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
