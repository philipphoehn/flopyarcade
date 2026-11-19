"""Generational trajectory ensemble plot for neuroevolution runs.

Overlays well trajectories from multiple generations in a single figure,
colored by generation index to show progress over evolution.

Usage::

    from flopyarcade.ensemble_plot import GenerationalEnsemblePlot

    plot = GenerationalEnsemblePlot()
    for gen in range(n_gens):
        plot.add_generation(trajectories[gen], reward=rewards[gen])
    plot.render()
    plot.save('ensemble.png')
"""

import numpy as np
from typing import List, Optional, Dict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class GenerationalEnsemblePlot:
    """Collect and plot well trajectories across neuroevolution generations.

    Args:
        figsize: Matplotlib figure size (default (10, 8)).
        colormap: Name of the matplotlib colormap (default 'viridis').
        alpha_range: (min_alpha, max_alpha) for line opacity (default (0.3, 1.0)).
        linewidth_range: (min_lw, max_lw) for line width (default (0.5, 2.0)).
        show_best: Highlight the generation with the best reward (default True).
    """

    def __init__(self, figsize=(10, 8), colormap='viridis',
                 alpha_range=(0.3, 1.0), linewidth_range=(0.5, 2.0),
                 show_best=True):
        self.figsize = figsize
        self.colormap = colormap
        self.alpha_range = alpha_range
        self.linewidth_range = linewidth_range
        self.show_best = show_best
        self.generations: List[Dict] = []
        self._fig = None
        self._ax = None

    @property
    def n_generations(self) -> int:
        """Number of generations stored."""
        return len(self.generations)

    def add_generation(self, trajectory: Dict, reward: float = 0.0) -> None:
        """Add one generation's trajectory data.

        Args:
            trajectory: Dict with keys 'x', 'y' (each a list of numpy arrays,
                        one per particle/segment). Optionally 'z'.
            reward: Reward score for this generation.
        """
        x_segs = np.asarray(trajectory.get('x', []), dtype=object)
        y_segs = np.asarray(trajectory.get('y', []), dtype=object)

        # Flatten to a list of (x, y) segment pairs
        segments = []
        n_segs = max(len(x_segs), len(y_segs)) if x_segs.size and y_segs.size else 0
        for i in range(n_segs):
            x_arr = np.asarray(x_segs[i], dtype=np.float64) if i < len(x_segs) else np.array([])
            y_arr = np.asarray(y_segs[i], dtype=np.float64) if i < len(y_segs) else np.array([])
            if len(x_arr) > 0 and len(y_arr) > 0:
                segments.append((x_arr, y_arr))

        self.generations.append({
            'segments': segments,
            'reward': reward,
            'n_segments': len(segments),
        })

    def add_from_env(self, env) -> None:
        """Add trajectory data directly from a FloPyEnv instance.

        Args:
            env: A FloPyEnv or compatible object with .trajectories and .reward attributes.
        """
        traj = {
            'x': env.trajectories.get('x', []),
            'y': env.trajectories.get('y', []),
        }
        reward = getattr(env, 'reward', 0.0)
        self.add_generation(traj, reward=reward)

    def render(self, ax=None, title: str = '') -> 'matplotlib.axes.Axes':
        """Render the ensemble plot.

        Args:
            ax: Optional existing axes to plot on.
            title: Plot title (default 'Generational Trajectory Ensemble').

        Returns:
            The matplotlib Axes object.
        """
        if not self.generations:
            raise ValueError("No generations added; call add_generation() first.")

        if ax is None:
            self._fig, ax = plt.subplots(figsize=self.figsize)
        self._ax = ax

        cmap = plt.get_cmap(self.colormap)
        norm = Normalize(vmin=0, vmax=max(1, len(self.generations) - 1))

        n_gen = len(self.generations)
        a_min, a_max = self.alpha_range
        lw_min, lw_max = self.linewidth_range

        # Find best generation index
        best_idx = int(np.argmax([g['reward'] for g in self.generations])) if self.show_best else -1

        for i, gen in enumerate(self.generations):
            # Scale alpha and linewidth by generation index (later = more prominent)
            if n_gen > 1:
                t = i / (n_gen - 1)
            else:
                t = 1.0
            alpha = a_min + t * (a_max - a_min)
            lw = lw_min + t * (lw_max - lw_min)
            color = cmap(norm(i))

            is_best = (i == best_idx)
            if is_best:
                alpha = 1.0
                lw = lw_max + 0.5

            for x_arr, y_arr in gen['segments']:
                ax.plot(x_arr, y_arr, color=color, alpha=alpha, linewidth=lw,
                        label=f'Gen {i+1} (r={gen["reward"]:.4f})' if is_best else None)

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(title or 'Generational Trajectory Ensemble')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Legend (only best generation to avoid clutter)
        if self.show_best and self._ax:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right', fontsize=8)

        plt.tight_layout()
        return ax

    def save(self, path: str, dpi: int = 150) -> str:
        """Save the rendered figure to a file.

        Args:
            path: Output file path (PNG, PDF, SVG, etc.).
            dpi: Resolution for raster formats.

        Returns:
            The output file path.
        """
        if self._fig is None:
            self.render()
        self._fig.savefig(path, dpi=dpi, bbox_inches='tight')
        return path

    def clear(self) -> None:
        """Remove all stored generations and the figure."""
        self.generations.clear()
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
        self._ax = None
