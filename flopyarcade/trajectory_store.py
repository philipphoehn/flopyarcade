"""Trajectory store for accumulating well trajectories across games.

Provides persistent storage of completed game trajectories and
ensemble visualization overlaying all stored trajectories.

Usage::

    from flopyarcade.trajectory_store import TrajectoryStore

    store = TrajectoryStore()
    # After each game completes:
    store.add(trajectories_dict, reward=game_reward, envtype='1s-d')

    # Render all stored trajectories as an ensemble:
    ax = store.render()
    store.save('all_trajectories.png')
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np


class TrajectoryStore:
    """Accumulate and visualize well trajectories across multiple games.

    Args:
        max_trajectories: Maximum number of completed trajectories to keep
            (0 = unlimited). Defaults to 0.
        figsize: Matplotlib figure size for ensemble plots.
        colormap: Matplotlib colormap name for coloring by reward.
    """

    def __init__(self, max_trajectories: int = 0,
                 figsize: tuple = (10, 8),
                 colormap: str = 'plasma'):
        self.max_trajectories = max_trajectories
        self.figsize = figsize
        self.colormap = colormap
        self._trajectories: List[Dict] = []

    @property
    def count(self) -> int:
        """Number of stored trajectories."""
        return len(self._trajectories)

    @property
    def trajectories(self) -> List[Dict]:
        """List of stored trajectory records."""
        return self._trajectories

    def add(self, trajectories: Dict, reward: float = 0.0,
            n_steps: int = 0, envtype: str = '',
            success: bool = False) -> None:
        """Store one completed game's trajectories.

        Args:
            trajectories: Dict with keys 'x', 'y', optionally 'z'.
                Each value is a list of numpy arrays (one per particle segment).
            reward: Total reward for this game.
            n_steps: Number of timesteps played.
            envtype: Environment type string (e.g. '1s-d').
            success: Whether the game was won.
        """
        record = {
            'x': [np.asarray(seg, dtype=np.float64) for seg in trajectories.get('x', [])],
            'y': [np.asarray(seg, dtype=np.float64) for seg in trajectories.get('y', [])],
            'z': [np.asarray(seg, dtype=np.float64) for seg in trajectories.get('z', [])],
            'reward': float(reward),
            'n_steps': int(n_steps),
            'envtype': envtype,
            'success': bool(success),
        }
        self._trajectories.append(record)

        # Enforce max_trajectories limit (drop oldest)
        if self.max_trajectories > 0 and len(self._trajectories) > self.max_trajectories:
            self._trajectories = self._trajectories[-self.max_trajectories:]

    def add_from_env(self, env) -> None:
        """Extract and store trajectory data from a FloPyEnv instance.

        Args:
            env: Object with .trajectories, .reward (or .rewardTotal),
                 .timeSteps, .envtype, .success attributes.
        """
        reward = getattr(env, 'rewardTotal', None)
        if reward is None:
            reward = getattr(env, 'reward', 0.0)
        n_steps = getattr(env, 'timeSteps', 0)
        envtype = getattr(env, 'envtype', '')
        success = bool(getattr(env, 'success', False))
        self.add(env.trajectories, reward=reward, n_steps=n_steps,
                 envtype=envtype, success=success)

    def render(self, title: str = 'All Well Trajectories (Ensemble)'):
        """Render all stored trajectories as an ensemble overlay.

        Color-coded by reward (higher reward = warmer color).

        Returns:
            matplotlib Axes object.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        if not self._trajectories:
            raise ValueError("No trajectories stored; call add() first.")

        fig, ax = plt.subplots(figsize=self.figsize)
        rewards = [t['reward'] for t in self._trajectories]
        norm = Normalize(vmin=min(rewards), vmax=max(rewards)) if len(set(rewards)) > 1             else Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap(self.colormap)

        for i, traj in enumerate(self._trajectories):
            color = cmap(norm(traj['reward']))
            alpha = 0.4 if not traj['success'] else 0.9
            lw = 1.0 if not traj['success'] else 2.0

            x_segs = traj['x']
            y_segs = traj['y']
            n_segs = min(len(x_segs), len(y_segs))
            for j in range(n_segs):
                ax.plot(x_segs[j], y_segs[j], color=color,
                        alpha=alpha, linewidth=lw)

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add summary text
        n_success = sum(1 for t in self._trajectories if t['success'])
        summary = (f"{self.count} games | {n_success} won | "
                   f"reward [{min(rewards):.0f}, {max(rewards):.0f}]")
        ax.text(0.02, 0.02, summary, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self._fig = fig
        self._ax = ax
        return ax

    def save(self, path: str, dpi: int = 150) -> str:
        """Save the ensemble plot to file.

        Args:
            path: Output file path.
            dpi: Resolution for raster formats.

        Returns:
            The output file path.
        """
        if not hasattr(self, '_fig') or self._fig is None:
            self.render()
        self._fig.savefig(path, dpi=dpi, bbox_inches='tight')
        return path

    def save_json(self, path: str) -> str:
        """Export all stored trajectories to a JSON file.

        Args:
            path: Output JSON file path.

        Returns:
            The output file path.
        """
        export = []
        for t in self._trajectories:
            export.append({
                'x': [seg.tolist() for seg in t['x']],
                'y': [seg.tolist() for seg in t['y']],
                'z': [seg.tolist() for seg in t['z']] if t['z'] else [],
                'reward': t['reward'],
                'n_steps': t['n_steps'],
                'envtype': t['envtype'],
                'success': t['success'],
            })
        with open(path, 'w') as f:
            json.dump(export, f, indent=2)
        return path

    def load_json(self, path: str) -> int:
        """Load trajectories from a JSON file into the store.

        Args:
            path: Path to a JSON file exported by save_json().

        Returns:
            Number of trajectories loaded.
        """
        with open(path) as f:
            data = json.load(f)
        for rec in data:
            self.add(
                {'x': rec.get('x', []), 'y': rec.get('y', []), 'z': rec.get('z', [])},
                reward=rec.get('reward', 0.0),
                n_steps=rec.get('n_steps', 0),
                envtype=rec.get('envtype', ''),
                success=rec.get('success', False),
            )
        return len(data)

    def clear(self) -> None:
        """Remove all stored trajectories."""
        self._trajectories.clear()
        self._fig = None
        self._ax = None

    def summary(self) -> Dict:
        """Return a summary dict of stored trajectories."""
        if not self._trajectories:
            return {'count': 0}
        rewards = [t['reward'] for t in self._trajectories]
        n_success = sum(1 for t in self._trajectories if t['success'])
        return {
            'count': self.count,
            'n_success': n_success,
            'n_failed': self.count - n_success,
            'reward_min': min(rewards),
            'reward_max': max(rewards),
            'reward_mean': float(np.mean(rewards)),
            'envtypes': sorted(set(t['envtype'] for t in self._trajectories if t['envtype'])),
        }
