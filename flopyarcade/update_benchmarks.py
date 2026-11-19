"""Auto-update benchmark figures for the FloPyArcade website.

Reads benchmark_results_*.json and regenerates the two standard figures:
  - benchmarks_operatorScores.png  (bar chart of mean reward per environment)
  - benchmarks_averageEvolutions.png (box-plot style distribution)

Usage::

    python -m flopyarcade.update_benchmarks
    python -m flopyarcade.update_benchmarks --results benchmark_results_1000games.json --output-dir examples/images
"""

import argparse
import json
import os
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import numpy as np

# Default paths (relative to repo root)
DEFAULT_RESULTS = 'benchmark_results_1000games.json'
DEFAULT_OUTPUT_DIR = 'flopyarcade/examples/images'


def load_benchmark_results(path: str) -> List[Dict]:
    """Load benchmark results from a JSON file.

    Args:
        path: Path to the JSON results file.

    Returns:
        List of result dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark results not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")
    return data


def plot_operator_scores(results: List[Dict], output_path: str) -> str:
    """Generate the operator-scores bar chart.

    Shows mean reward per environment type with error bars (std).

    Args:
        results: List of benchmark result dicts.
        output_path: Where to save the PNG.

    Returns:
        The output file path.
    """
    envtypes = [r['envtype'] for r in results]
    means = [r['mean_reward'] for r in results]
    stds = [r.get('std_reward', 0) for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(envtypes))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='#42A5F5',
                  edgecolor='#1565C0', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(envtypes, fontsize=11)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('FloPyArcade Benchmark — Operator Scores', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    # Annotate bars with values
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(min(means) - max(stds) * 0.5, max(means) + max(stds) * 0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_average_evolutions(results: List[Dict], output_path: str) -> str:
    """Generate the average-evolutions distribution plot.

    Shows min/median/max reward spread per environment type.

    Args:
        results: List of benchmark result dicts.
        output_path: Where to save the PNG.

    Returns:
        The output file path.
    """
    envtypes = [r['envtype'] for r in results]
    mins = [r.get('min_reward', 0) for r in results]
    medians = [r.get('median_reward', r.get('mean_reward', 0)) for r in results]
    maxs = [r.get('max_reward', 0) for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(envtypes))
    width = 0.5

    # Draw range bars (min to max)
    for i in range(len(envtypes)):
        ax.plot([x[i], x[i]], [mins[i], maxs[i]], color='#90CAF9',
                linewidth=3, solid_capstyle='round', zorder=2)
        # Median marker
        ax.plot(x[i], medians[i], 'o', color='#1565C0', markersize=8, zorder=3)
        # Mean marker
        mean_val = results[i].get('mean_reward', 0)
        ax.plot(x[i], mean_val, 's', color='#E53935', markersize=7,
                markeredgecolor='white', markeredgewidth=1, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(envtypes, fontsize=11)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('FloPyArcade Benchmark — Reward Distribution per Environment',
                 fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#90CAF9', linewidth=3, label='Min–Max range'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0',
               markersize=8, label='Median'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E53935',
               markersize=7, label='Mean'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    all_vals = mins + maxs
    pad = (max(all_vals) - min(all_vals)) * 0.1 or 1
    ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def update_benchmarks(results_path: str = DEFAULT_RESULTS,
                      output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, str]:
    """Main entry: load results and regenerate both figures.

    Args:
        results_path: Path to the JSON results file.
        output_dir: Directory to write the PNG figures.

    Returns:
        Dict mapping figure name to output path.
    """
    results = load_benchmark_results(results_path)

    fig1_path = os.path.join(output_dir, 'benchmarks_operatorScores.png')
    fig2_path = os.path.join(output_dir, 'benchmarks_averageEvolutions.png')

    plot_operator_scores(results, fig1_path)
    plot_average_evolutions(results, fig2_path)

    return {
        'operator_scores': fig1_path,
        'average_evolutions': fig2_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate FloPyArcade benchmark figures from JSON results.')
    parser.add_argument('--results', default=DEFAULT_RESULTS,
                        help='Path to benchmark results JSON')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='Directory to write PNG figures')
    args = parser.parse_args()

    paths = update_benchmarks(args.results, args.output_dir)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("Done.")


if __name__ == '__main__':
    main()
