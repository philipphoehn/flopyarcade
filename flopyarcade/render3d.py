"""
FloPyArcade 3D Visualization Module — Cyberpunk Edition.

Renders the groundwater flow environment in 3D using matplotlib 3D:
- Head surface with terrain colormap (original FloPyArcade colours)
- Neon cyan/magenta cyberpunk accents for particle, well, trajectory
- Trajectory locked to surface (bilinear head interpolation)
- Optimized: precomputed grid, batched draws, minimal per-frame cost

Works headlessly with the Agg backend (no GPU/display required).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import Optional, List, Tuple
import os


# ─── Cyberpunk palette ───
BG = '#000000'
CYAN = '#00ffff'
MAGENTA = '#ff00cc'
NEON_GREEN = '#00ff88'
SURFACE_ALPHA = 0.72


class Render3D:
    """3D renderer for FloPyArcade — cyberpunk style, terrain colormap."""

    def __init__(self, env, headless=True, colormap='terrain',
                 surface_alpha=SURFACE_ALPHA, elevation_scale=1.0,
                 fig_size=(12, 9)):
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
        self._grid_cache = None

    # ─── Grid (cached) ───
    def _build_grid(self):
        """Build 3D grid once, cache for reuse."""
        if self._grid_cache is not None:
            return self._grid_cache

        x = np.linspace(0, self.extentX, self.ncol + 1)
        y = np.linspace(0, self.extentY, self.nrow + 1)
        X, Y = np.meshgrid(x, y)

        heads = np.asarray(self.env.heads, dtype=float)
        if heads.shape != (self.nrow, self.ncol):
            from scipy.ndimage import zoom
            heads = zoom(heads, (self.nrow / heads.shape[0],
                                 self.ncol / heads.shape[1]), order=1)

        # Bilinear cell-centre → node interpolation (vectorised)
        head_nodes = np.zeros((self.nrow + 1, self.ncol + 1))
        for i in range(self.nrow + 1):
            i0 = max(0, i - 1); i1 = min(self.nrow - 1, i)
            for j in range(self.ncol + 1):
                j0 = max(0, j - 1); j1 = min(self.ncol - 1, j)
                head_nodes[i, j] = 0.25 * (heads[i0, j0] + heads[i0, j1] +
                                           heads[i1, j0] + heads[i1, j1])

        Z = head_nodes * self.elevation_scale
        self._grid_cache = (X, Y, Z, head_nodes)
        return self._grid_cache

    def _interpolate_head(self, px, py):
        """Bilinear head at (px, py)."""
        heads = np.asarray(self.env.heads, dtype=float)
        ci = np.clip(px / self.delr - 0.5, 0, self.ncol - 1)
        ri = np.clip(self.nrow - 1 - (py / self.delc - 0.5), 0, self.nrow - 1)
        c0, r0 = int(ci), int(ri)
        c1, r1 = min(c0 + 1, self.ncol - 1), min(r0 + 1, self.nrow - 1)
        fc, fr = ci - c0, ri - r0
        return ((1 - fc) * (1 - fr) * heads[r0, c0] + fc * (1 - fr) * heads[r0, c1]
                + (1 - fc) * fr * heads[r1, c0] + fc * fr * heads[r1, c1])

    def _pos3d(self, px, py):
        """3D position on surface."""
        h = self._interpolate_head(px, py) * self.elevation_scale
        return np.array([px, py, h])

    def _get_particle_position(self):
        c = self.env.particleCoords
        return self._pos3d(c[0], c[1])

    def _get_well_position(self):
        c = self.env.wellCoords
        return self._pos3d(c[0], c[1])

    # ─── Render ───
    def render(self, save_path=None, title=None):
        """Render one cyberpunk 3D frame."""
        fig = plt.figure(figsize=self.fig_size)
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(BG)

        X, Y, Z, head_nodes = self._build_grid()
        cmap = getattr(cm, self.colormap)
        norm = Normalize(head_nodes.min(), head_nodes.max())

        # 1. Terrain surface (original colormap)
        ax.plot_surface(X, Y, Z,
                        facecolors=cmap(norm(head_nodes)),
                        alpha=self.surface_alpha,
                        linewidth=0, antialiased=True,
                        rstride=1, cstride=1)

        # 2. Subtle wireframe overlay (cyberpunk grid feel)
        ws = max(1, self.ncol // 12)
        ax.plot_wireframe(X[::ws, ::ws], Y[::ws, ::ws], Z[::ws, ::ws],
                          color=CYAN, alpha=0.08, lw=0.3)

        # 3. Trajectory — locked to surface, neon gradient
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            n = len(traj)
            # Batch: single plot call (fastest)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=MAGENTA, lw=2.0, alpha=0.7, zorder=5)
            # Glow dots every 3rd point
            step = max(1, n // 15)
            ax.scatter(traj[::step, 0], traj[::step, 1], traj[::step, 2],
                       color=CYAN, s=8, alpha=0.5, zorder=6)

        # 4. Particle — multi-layer glow
        ppos = self._get_particle_position()
        for s, a in [(500, 0.03), (250, 0.08), (120, 0.2), (50, 0.5)]:
            ax.scatter(ppos[0], ppos[1], ppos[2],
                       color=MAGENTA, s=s, alpha=a, zorder=8)
        ax.scatter(ppos[0], ppos[1], ppos[2],
                   color='white', s=30, zorder=9)

        # 5. Well — neon cyan tower
        wpos = self._get_well_position()
        theta = np.linspace(0, 2 * np.pi, 32)
        r = 0.5
        for k in range(8):
            ax.plot(wpos[0] + r * np.cos(theta),
                    wpos[1] + r * np.sin(theta),
                    np.full(32, wpos[2] + 0.05 + k * 0.08),
                    color=CYAN, lw=1, alpha=0.35 - k * 0.03, zorder=5)
        ax.scatter(wpos[0], wpos[1], wpos[2],
                   color=CYAN, s=200, alpha=0.08, zorder=5)

        # 6. Capture zone ring
        if hasattr(self.env, 'wellRadius'):
            cr = self.env.wellRadius
            ax.plot(wpos[0] + cr * np.cos(theta),
                    wpos[1] + cr * np.sin(theta),
                    np.full(32, wpos[2] + 0.02),
                    color=CYAN, lw=2, alpha=0.25, zorder=4)

        # 7. Boundaries (subtle)
        if hasattr(self.env, 'headSpecNorth') and self.env.headSpecNorth is not None:
            zl = head_nodes.max() * 0.15
            ax.plot([0, self.extentX], [self.extentY, self.extentY], [zl, zl],
                    color='#ff8800', lw=2, alpha=0.4)
        if hasattr(self.env, 'headSpecSouth') and self.env.headSpecSouth is not None:
            zl = head_nodes.max() * 0.15
            ax.plot([0, self.extentX], [0, 0], [zl, zl],
                    color='#ff8800', lw=2, alpha=0.4)

        # 8. HUD
        envtype = getattr(self.env, 'ENVTYPE', '?')
        score = getattr(self.env, 'rewardCurrent', 0)
        t = getattr(self.env, 'timeStep', 0)
        pl = getattr(self.env, 'periodLength', 50)
        hud = f'SCORE: {score}\nTIME: {t}d  [{int(t/pl*10)*"█" + int(10-t/pl*10)*"░"}]  {t}/{pl}'
        fig.text(0.03, 0.96, hud, fontsize=10, color=NEON_GREEN,
                 va='top', ha='left', family='monospace', fontweight='bold',
                 zorder=20)

        # 9. Strip all axes
        ax.grid(False)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0, 0, 0, 0))
            a.pane.fill = False
            a.pane.set_visible(False)
            a.pane.set_edgecolor((0, 0, 0, 0))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')

        ax.view_init(elev=20, azim=-50)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
            plt.close(fig)
            return save_path
        if not self.headless:
            plt.show()
        else:
            plt.close(fig)
        return None

    def record_trajectory(self):
        """Record particle position (on surface) into trajectory."""
        self.trajectory.append(tuple(self._get_particle_position()))


def render_environment_3d(env, output_path=None, show=False,
                           elevation_scale=1.0, colormap='terrain'):
    """Convenience: render one 3D cyberpunk frame."""
    r = Render3D(env, headless=(output_path is not None),
                 elevation_scale=elevation_scale, colormap=colormap)
    return r.render(save_path=output_path)
