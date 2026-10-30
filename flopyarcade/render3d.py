"""
FloPyArcade 3D Visualization Module.

Renders the groundwater flow environment in 3D using matplotlib 3D:
- Head surface as a colored terrain
- Particle position and trajectory
- Well and capture zone
- North/South constraint boundaries

Works headlessly with the Agg backend (no GPU/display required).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from typing import Optional, List, Tuple
import os


class Render3D:
    """3D renderer for FloPyArcade environments using matplotlib."""

    def __init__(self, env, headless=True, colormap='viridis',
                 surface_alpha=0.85, elevation_scale=1.0, fig_size=(12, 9)):
        """Initialize the 3D renderer.

        Args:
            env: A FloPyEnv-like instance with heads, particleCoords, wellCoords, etc.
            headless: If True, render to file only (no display window).
            colormap: Colormap name for head surface.
            surface_alpha: Transparency of the head surface (0-1).
            elevation_scale: Multiplier to exaggerate head values vertically.
            fig_size: Figure size tuple (width, height) in inches.
        """
        self.env = env
        self.headless = headless
        self.colormap = colormap
        self.surface_alpha = surface_alpha
        self.elevation_scale = elevation_scale
        self.fig_size = fig_size

        self.nrow = env.nrow
        self.ncol = env.ncol
        self.delr = getattr(env, 'delr', 100.0)
        self.delc = getattr(env, 'delc', 100.0)
        self.extentX = env.extentX
        self.extentY = env.extentY
        self.zTop = getattr(env, 'zTop', 50.0)
        self.zBot = getattr(env, 'zBot', 0.0)

        self.trajectory: List[Tuple[float, float, float]] = []
        self._fig = None
        self._ax = None

    def _build_grid(self):
        """Build 3D grid coordinates (X, Y, Z) for the head surface.

        Returns:
            tuple: (X, Y, Z, head_nodes) arrays of shape (nrow+1, ncol+1)
        """
        x = np.linspace(0, self.extentX, self.ncol + 1)
        y = np.linspace(0, self.extentY, self.nrow + 1)
        X, Y = np.meshgrid(x, y)

        heads = np.asarray(self.env.heads, dtype=float)
        if heads.shape != (self.nrow, self.ncol):
            from scipy.ndimage import zoom
            factor_r = self.nrow / heads.shape[0]
            factor_c = self.ncol / heads.shape[1]
            heads = zoom(heads, (factor_r, factor_c), order=1)

        # Interpolate cell-center heads to node grid
        head_nodes = np.zeros((self.nrow + 1, self.ncol + 1))
        for i in range(self.nrow + 1):
            i0 = max(0, i - 1)
            i1 = min(self.nrow - 1, i)
            for j in range(self.ncol + 1):
                j0 = max(0, j - 1)
                j1 = min(self.ncol - 1, j)
                head_nodes[i, j] = 0.25 * (heads[i0, j0] + heads[i0, j1] +
                                           heads[i1, j0] + heads[i1, j1])

        Z = head_nodes * self.elevation_scale
        return X, Y, Z, head_nodes

    def _interpolate_head(self, px, py):
        """Interpolate head value at (px, py) position."""
        heads = np.asarray(self.env.heads, dtype=float)
        col_idx = px / self.delr - 0.5
        row_idx = self.nrow - 1 - (py / self.delc - 0.5)

        col_idx = np.clip(col_idx, 0, self.ncol - 1)
        row_idx = np.clip(row_idx, 0, self.nrow - 1)

        c0 = int(np.floor(col_idx))
        r0 = int(np.floor(row_idx))
        c1 = min(c0 + 1, self.ncol - 1)
        r1 = min(r0 + 1, self.nrow - 1)
        fc = col_idx - c0
        fr = row_idx - r0

        head = ((1 - fc) * (1 - fr) * heads[r0, c0] +
                fc * (1 - fr) * heads[r0, c1] +
                (1 - fc) * fr * heads[r1, c0] +
                fc * fr * heads[r1, c1])
        return head

    def _get_particle_position(self) -> np.ndarray:
        """Get particle 3D position on the head surface."""
        coords = self.env.particleCoords
        px, py = coords[0], coords[1]
        head = self._interpolate_head(px, py)
        return np.array([px, py, head * self.elevation_scale])

    def _get_well_position(self) -> np.ndarray:
        """Get well 3D position on the head surface."""
        coords = self.env.wellCoords
        wx, wy = coords[0], coords[1]
        head = self._interpolate_head(wx, wy)
        return np.array([wx, wy, head * self.elevation_scale])

    def render(self, save_path: Optional[str] = None, title: str = None) -> Optional[str]:
        """Render a single 3D frame.

        Args:
            save_path: If provided, save the image to this path.
            title: Custom title. Auto-generated if None.

        Returns:
            The save path if saved, else None.
        """
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection='3d')

        # 1. Head surface
        X, Y, Z, head_nodes = self._build_grid()
        surf = ax.plot_surface(X, Y, Z, cmap=getattr(cm, self.colormap),
                               alpha=self.surface_alpha, rstride=1, cstride=1,
                               linewidth=0, antialiased=True)

        # 2. Contour lines on surface
        contour_levels = np.linspace(head_nodes.min(), head_nodes.max(), 8)
        cs = ax.contourf(X, Y, Z, levels=contour_levels,
                         cmap=getattr(cm, self.colormap), alpha=0.1, zdir='z', offset=Z.min())

        # 3. Particle
        ppos = self._get_particle_position()
        ax.scatter(*ppos, c='#FF4444', s=200, marker='o',
                   edgecolors='black', linewidths=0.5, depthshade=False)

        # 4. Particle trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color='#FFD700', linewidth=2.5, label='Trajectory')

        # 5. Well
        wpos = self._get_well_position()
        ax.scatter(*wpos, c='#4488FF', s=300, marker='s',
                   edgecolors='black', linewidths=0.8, depthshade=False)

        # Well capture zone (circle)
        if hasattr(self.env, 'wellRadius'):
            theta = np.linspace(0, 2 * np.pi, 64)
            r = self.env.wellRadius
            cx, cy, cz = wpos
            circ_x = cx + r * np.cos(theta)
            circ_y = cy + r * np.sin(theta)
            circ_z = np.full_like(theta, cz)
            ax.plot(circ_x, circ_y, circ_z, color='#4488FF', linewidth=1.5, alpha=0.7)

        # 6. North/South boundaries
        if hasattr(self.env, 'headSpecNorth') and self.env.headSpecNorth is not None:
            zline = self.zTop * self.elevation_scale * 0.3
            ax.plot([0, self.extentX], [self.extentY, self.extentY], [zline, zline],
                    color='#FF8800', linewidth=3, label='N boundary')
        if hasattr(self.env, 'headSpecSouth') and self.env.headSpecSouth is not None:
            zline = self.zTop * self.elevation_scale * 0.3
            ax.plot([0, self.extentX], [0, 0], [zline, zline],
                    color='#FF8800', linewidth=3, label='S boundary')

        # 7. West/East boundaries (subtle)
        zline2 = self.zTop * self.elevation_scale * 0.15
        ax.plot([0, 0], [0, self.extentY], [zline2, zline2], color='#888888', linewidth=1, alpha=0.5)
        ax.plot([self.extentX, self.extentX], [0, self.extentY], [zline2, zline2], color='#888888', linewidth=1, alpha=0.5)

        # 8. Labels and axes
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Head (m)', fontsize=9)

        if title is None:
            envtype = getattr(self.env, 'ENVTYPE', '?')
            title = (f'FloPyArcade 3D | ENV: {envtype} | '
                     f'Particle: ({ppos[0]:.0f}, {ppos[1]:.0f}) | '
                     f'Well: ({wpos[0]:.0f}, {wpos[1]:.0f})')
        ax.set_title(title, fontsize=11, pad=10)

        # Camera angle
        ax.view_init(elev=25, azim=-60)

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Hydraulic Head (m)', fontsize=9)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            return save_path

        if not self.headless:
            plt.show()
        else:
            plt.close(fig)
        return None

    def record_trajectory(self):
        """Record current particle position into trajectory."""
        ppos = self._get_particle_position()
        self.trajectory.append(tuple(ppos))


def render_environment_3d(env, output_path: str = None, show: bool = False,
                           elevation_scale: float = 1.0, colormap: str = 'viridis'):
    """Convenience function to render a single 3D frame of an environment.

    Args:
        env: FloPyEnv-like instance.
        output_path: Path to save PNG. If None and show=True, display interactively.
        show: Open interactive window.
        elevation_scale: Vertical exaggeration factor.
        colormap: Colormap for head surface.

    Returns:
        Saved file path or None.
    """
    renderer = Render3D(env, headless=output_path is not None,
                        elevation_scale=elevation_scale, colormap=colormap)
    return renderer.render(save_path=output_path)
