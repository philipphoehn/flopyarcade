#!/usr/bin/env python3
"""
FloPyArcade 3D Play & Animation.

Renders the groundwater environment in 3D using matplotlib 3D.
Supports:
- Single-frame rendering from a live or synthetic environment
- Animated GIF of a particle crossing the domain

Usage:
    python -m flopyarcade.play3d --mode demo --output animation.gif
    python -m flopyarcade.play3d --mode live --envtype 1s-d --output frame.png
"""

import argparse
import os
import sys
import numpy as np
from typing import Optional, Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class MockEnv:
    """Synthetic environment for 3D demo without MODFLOW.

    Creates a realistic-looking head field with:
    - West-to-east gradient (headSpecWest > headSpecEast)
    - A localized depression (well effect)
    - North/south constraint bumps
    """

    def __init__(self, nrow=50, ncol=50, delr=100.0, delc=100.0,
                 headWest=60.0, headEast=56.0, wellRadius=5.0,
                 ENVTYPE='1s-d'):
        self.nrow = nrow
        self.ncol = ncol
        self.nlay = 1
        self.delr = delr
        self.delc = delc
        self.extentX = ncol * delr
        self.extentY = nrow * delc
        self.zBot = 0.0
        self.zTop = 50.0
        self.ENVTYPE = ENVTYPE
        self.wellRadius = wellRadius
        self.headSpecWest = headWest
        self.headSpecEast = headEast
        self.headSpecNorth = 58.0
        self.headSpecSouth = 57.0
        self.done = False

        # Well position (center-ish)
        self.wellX = self.extentX * 0.5
        self.wellY = self.extentY * 0.5
        self.wellZ = self.zTop
        self.wellCoords = [self.wellX, self.wellY, self.wellZ]
        self.wellQ = -100.0

        # Particle starts at west edge
        self.particleX = 1.1 * delc
        self.particleY = self.extentY * 0.5
        self.particleZ = self.zTop
        self.particleCoords = [self.particleX, self.particleY, self.particleZ]

        # Build synthetic head field
        self.heads = self._generate_heads()

    def _generate_heads(self) -> np.ndarray:
        """Generate a realistic head field."""
        x = np.linspace(0, self.extentX, self.ncol)
        y = np.linspace(0, self.extentY, self.nrow)
        X, Y = np.meshgrid(x, y)

        # Base gradient: west (high) to east (low)
        frac_x = X / self.extentX
        heads = self.headSpecWest + (self.headSpecEast - self.headSpecWest) * frac_x

        # Well depression (Gaussian)
        well_sigma = self.wellRadius * 5
        heads -= 3.0 * np.exp(-0.5 * ((X - self.wellX) ** 2 + (Y - self.wellY) ** 2) / (well_sigma ** 2))

        # North boundary constraint (slight bump)
        north_influence = np.exp(-Y / (self.extentY * 0.3))
        heads += 0.5 * north_influence

        # South boundary constraint
        south_influence = np.exp(-(self.extentY - Y) / (self.extentY * 0.3))
        heads += 0.3 * south_influence

        # Small random noise for realism
        np.random.seed(42)
        heads += np.random.normal(0, 0.1, heads.shape)

        return heads

    def reset(self):
        """Reset particle to starting position."""
        self.particleX = 1.1 * self.delc
        self.particleY = self.extentY * 0.5
        self.particleCoords = [self.particleX, self.particleY, self.particleZ]
        self.done = False
        return {'heads': self.heads}

    def step(self, action=None, teardownOnFinish=True):
        """Advance particle eastward (simple advection)."""
        dx = self.delr * 0.5
        self.particleX += dx
        if self.particleX >= self.extentX - self.delc:
            self.particleX = self.extentX - self.delc
            self.done = True
        self.particleCoords = [self.particleX, self.particleY, self.particleZ]
        return self.particleCoords, 0.0, self.done, False, {}

    def distance_to_well(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _interpolate_head(heads, nrow, ncol, delr, delc, px, py):
    """Bilinear interpolation of head at position (px, py)."""
    col_idx = px / delr - 0.5
    row_idx = nrow - 1 - (py / delc - 0.5)
    col_idx = np.clip(col_idx, 0, ncol - 1)
    row_idx = np.clip(row_idx, 0, nrow - 1)
    c0 = int(np.floor(col_idx))
    r0 = int(np.floor(row_idx))
    c1 = min(c0 + 1, ncol - 1)
    r1 = min(r0 + 1, nrow - 1)
    fc = col_idx - c0
    fr = row_idx - r0
    return ((1 - fc) * (1 - fr) * heads[r0, c0] +
            fc * (1 - fr) * heads[r0, c1] +
            (1 - fc) * fr * heads[r1, c0] +
            fc * fr * heads[r1, c1])


def _render_frame(env, trajectory, step_num, total_steps):
    """Render a single 3D frame. Returns fig."""
    from render3d import Render3D
    renderer = Render3D(env, headless=True)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Head surface
    X, Y, Z, head_nodes = renderer._build_grid()
    cmap = cm.viridis
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, rstride=1, cstride=1,
                           linewidth=0, antialiased=True)

    # Contour
    contour_levels = np.linspace(head_nodes.min(), head_nodes.max(), 8)
    ax.contourf(X, Y, Z, levels=contour_levels, cmap=cmap, alpha=0.1,
                zdir='z', offset=Z.min())

    # Trajectory
    if len(trajectory) > 1:
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color='#FFD700', linewidth=3, label='Trajectory')

    # Particle
    ppos = renderer._get_particle_position()
    ax.scatter(*ppos, c='#FF4444', s=250, marker='o',
               edgecolors='black', linewidths=0.8, depthshade=False,
               label='Particle')

    # Well
    wpos = renderer._get_well_position()
    ax.scatter(*wpos, c='#4488FF', s=350, marker='s',
               edgecolors='black', linewidths=1.0, depthshade=False,
               label='Well')

    # Well capture zone
    if hasattr(env, 'wellRadius'):
        theta = np.linspace(0, 2 * np.pi, 64)
        r = env.wellRadius
        cx, cy, cz = wpos
        circ_x = cx + r * np.cos(theta)
        circ_y = cy + r * np.sin(theta)
        circ_z = np.full_like(theta, cz)
        ax.plot(circ_x, circ_y, circ_z, color='#4488FF', linewidth=2, alpha=0.8)

    # Boundaries
    zline = env.zTop * 0.3
    if hasattr(env, 'headSpecNorth'):
        ax.plot([0, env.extentX], [env.extentY, env.extentY], [zline, zline],
                color='#FF8800', linewidth=3, label='N/S boundary')
    if hasattr(env, 'headSpecSouth'):
        ax.plot([0, env.extentX], [0, 0], [zline, zline],
                color='#FF8800', linewidth=3)

    # Labels
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Head (m)', fontsize=9)

    title = (f'FloPyArcade 3D | Step {step_num}/{total_steps} | '
             f'Particle: ({env.particleCoords[0]:.0f}m, {env.particleCoords[1]:.0f}m) | '
             f'Well: ({env.wellCoords[0]:.0f}m, {env.wellCoords[1]:.0f}m)')
    ax.set_title(title, fontsize=11, pad=10)
    ax.view_init(elev=25, azim=-60)

    return fig


def run_demo(output_path: str = 'animation_3d.gif', nsteps: int = 40,
             nrow: int = 50, ncol: int = 50):
    """Generate a 3D animation with a synthetic environment."""
    import imageio

    print(f"Creating MockEnv ({nrow}x{ncol})...")
    env = MockEnv(nrow=nrow, ncol=ncol)

    print(f"Rendering {nsteps} frames...")
    frames = []
    trajectory = []
    fig = None
    ax = None

    for step in range(nsteps):
        env.step()
        from render3d import Render3D
        renderer = Render3D(env, headless=True)
        renderer.record_trajectory()
        trajectory = renderer.trajectory

        fig = _render_frame(env, trajectory, step + 1, nsteps)

        # Save frame as uint8 RGB
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img = np.array(Image.open(buf).convert('RGB'))
        frames.append(img)
        buf.close()
        print(f"  Frame {step+1}/{nsteps} - particle at x={env.particleX:.0f}m")

        if env.done:
            print("  Particle reached eastern boundary!")
            break

    plt.close('all')

    # Save GIF
    if frames:
        imageio.mimsave(output_path, frames, duration=150)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nAnimation saved: {output_path} ({size_mb:.1f} MB, {len(frames)} frames)")

    # Also save a single high-res frame
    single_path = os.path.splitext(output_path)[0] + '_frame.png'
    if frames:
        # Re-render last state at high res
        from render3d import Render3D
        renderer = Render3D(env, headless=True)
        env2 = MockEnv(nrow=nrow, ncol=ncol)
        for s in range(nsteps):
            env2.step()
            renderer.record_trajectory()
            if env2.done:
                break
        fig2 = _render_frame(env2, renderer.trajectory, nsteps, nsteps)
        fig2.savefig(single_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"Single frame saved: {single_path}")

    return output_path


def run_live(output_path: str = 'frame3d.png', envtype: str = '1s-d'):
    """Render a live environment (requires MODFLOW)."""
    from flopyarcade import FloPyEnv
    from render3d import render_environment_3d

    print(f"Creating live environment (ENVTYPE={envtype})...")
    env = FloPyEnv(ENVTYPE=envtype, nRow=25, nCol=25, flagRender=False)
    env.reset()

    print("Rendering 3D frame...")
    path = render_environment_3d(env, output_path=output_path, elevation_scale=1.0)
    print(f"Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='FloPyArcade 3D Visualization')
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo',
                        help='demo=synthetic, live=MODFLOW')
    parser.add_argument('--output', default='animation_3d.gif',
                        help='Output file path')
    parser.add_argument('--nsteps', type=int, default=40,
                        help='Number of animation steps (demo mode)')
    parser.add_argument('--nrow', type=int, default=50,
                        help='Grid rows')
    parser.add_argument('--ncol', type=int, default=50,
                        help='Grid columns')
    parser.add_argument('--envtype', default='1s-d',
                        help='Environment type (live mode)')
    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo(output_path=args.output, nsteps=args.nsteps,
                 nrow=args.nrow, ncol=args.ncol)
    elif args.mode == 'live':
        run_live(output_path=args.output, envtype=args.envtype)


if __name__ == '__main__':
    main()
