#!/usr/bin/env python3
"""
FloPyArcade 3D Play & Animation — Cyberpunk Edition.

Renders the groundwater environment in 3D using matplotlib 3D:
- Terrain colormap (original FloPyArcade colours)
- Cyberpunk neon accents (magenta particle, cyan well, black bg)
- Trajectory locked to surface
- Optimized: grid cached, batched draws

Usage:
    python -m flopyarcade.play3d --mode demo --output animation.gif
    python -m flopyarcade.play3d --mode live --envtype 1s-d --output frame.png
"""

import argparse
import os
import sys
import io
import time
import numpy as np
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MockEnv:
    """Synthetic environment for 3D demo without MODFLOW."""

    def __init__(self, nrow=40, ncol=40, delr=100.0, delc=100.0,
                 headWest=60.0, headEast=56.0, wellRadius=800.0,
                 ENVTYPE='6r-c'):
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
        self.success = True
        self.rewardCurrent = 142
        self.timeStep = 0
        self.periodLength = 50

        self.wellX = self.extentX * 0.7
        self.wellY = self.extentY * 0.45
        self.wellCoords = [self.wellX, self.wellY]
        self.wellQ = -0.005
        self.minQ = -0.01

        self.particleX = self.extentX * 0.05
        self.particleY = self.extentY * 0.5
        self.particleCoords = [self.particleX, self.particleY]

        self.heads = self._generate_heads()

    def _generate_heads(self):
        x = np.linspace(0, 1, self.ncol)
        y = np.linspace(0, 1, self.nrow)
        X, Y = np.meshgrid(x, y)
        h = (55.0 + 3.0 * (1 - X) + 2.0 * np.sin(X * 4) + 1.5 * np.cos(Y * 3)
             + 0.8 * np.sin(X * 7 + Y * 5)
             - 3.5 * np.exp(-((X - 0.7) ** 2 + (Y - 0.45) ** 2) / 0.01))
        return h

    def reset(self):
        self.particleX = self.extentX * 0.05
        self.particleY = self.extentY * 0.5
        self.particleCoords = [self.particleX, self.particleY]
        self.done = False
        self.rewardCurrent = 142
        self.timeStep = 0
        return {'heads': self.heads}

    def step(self, action=None, teardownOnFinish=True):
        dx = self.extentX * 0.03
        self.particleX += dx
        self.timeStep += 1
        if self.particleX >= self.extentX * 0.95:
            self.particleX = self.extentX * 0.95
            self.done = True
        self.particleCoords = [self.particleX, self.particleY]
        return self.particleCoords, 0.0, self.done, False, {}

    def distance_to_well(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _render_frame(env, renderer, step_num, total_steps):
    """Render one frame using the shared renderer (grid cached)."""
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#000000')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#000000')

    X, Y, Z, head_nodes = renderer._build_grid()
    from matplotlib import cm
    from matplotlib.colors import Normalize
    cmap = cm.terrain
    norm = Normalize(head_nodes.min(), head_nodes.max())

    # Terrain surface
    ax.plot_surface(X, Y, Z,
                    facecolors=cmap(norm(head_nodes)),
                    alpha=0.72, linewidth=0, antialiased=True,
                    rstride=1, cstride=1)

    # Subtle wireframe
    ws = max(1, env.ncol // 12)
    ax.plot_wireframe(X[::ws, ::ws], Y[::ws, ::ws], Z[::ws, ::ws],
                      color='#00ffff', alpha=0.08, lw=0.3)

    # Trajectory (on surface)
    if len(renderer.trajectory) > 1:
        traj = np.array(renderer.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color='#ff00cc', lw=2.0, alpha=0.7, zorder=5)
        step = max(1, len(traj) // 15)
        ax.scatter(traj[::step, 0], traj[::step, 1], traj[::step, 2],
                   color='#00ffff', s=8, alpha=0.5, zorder=6)

    # Particle glow
    ppos = renderer._get_particle_position()
    for s, a in [(500, 0.03), (250, 0.08), (120, 0.2), (50, 0.5)]:
        ax.scatter(ppos[0], ppos[1], ppos[2],
                   color='#ff00cc', s=s, alpha=a, zorder=8)
    ax.scatter(ppos[0], ppos[1], ppos[2], color='white', s=30, zorder=9)

    # Well
    wpos = renderer._get_well_position()
    theta = np.linspace(0, 2 * np.pi, 32)
    for k in range(8):
        ax.plot(wpos[0] + 0.5 * np.cos(theta),
                wpos[1] + 0.5 * np.sin(theta),
                np.full(32, wpos[2] + 0.05 + k * 0.08),
                color='#00ffff', lw=1, alpha=0.35 - k * 0.03, zorder=5)

    # Capture zone
    if hasattr(env, 'wellRadius'):
        cr = env.wellRadius
        ax.plot(wpos[0] + cr * np.cos(theta),
                wpos[1] + cr * np.sin(theta),
                np.full(32, wpos[2] + 0.02),
                color='#00ffff', lw=2, alpha=0.25, zorder=4)

    # HUD
    score = getattr(env, 'rewardCurrent', 0)
    t = step_num
    pl = total_steps
    bar_len = 10
    filled = int(t / pl * bar_len)
    bar = '█' * filled + '░' * (bar_len - filled)
    hud = f'SCORE: {score}\nTIME: {t}/{pl}d  [{bar}]'
    fig.text(0.03, 0.96, hud, fontsize=10, color='#00ff88',
             va='top', ha='left', family='monospace', fontweight='bold',
             zorder=20)

    # Strip axes
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0))
        a.pane.fill = False
        a.pane.set_visible(False)
        a.pane.set_edgecolor((0, 0, 0, 0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')

    ax.view_init(elev=20, azim=-50)
    return fig


def run_demo(output_path='animation_3d.gif', nsteps=30,
             nrow=40, ncol=40, width=1200, height=800):
    """Generate a 3D animation with a synthetic environment."""
    import imageio
    from render3d import Render3D

    print(f"Creating MockEnv ({nrow}x{ncol})...")
    env = MockEnv(nrow=nrow, ncol=ncol)
    renderer = Render3D(env, headless=True)

    print(f"Rendering {nsteps} frames...")
    t0 = time.time()
    frames = []

    for step in range(nsteps):
        env.step()
        renderer.record_trajectory()

        fig = _render_frame(env, renderer, step + 1, nsteps)

        # Save frame as RGB array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='#000000')
        buf.seek(0)
        from PIL import Image
        img = np.array(Image.open(buf).convert('RGB'))
        frames.append(img)
        buf.close()
        plt.close(fig)

        if step % 10 == 0:
            print(f"  Frame {step+1}/{nsteps}")

        if env.done:
            print("  Particle reached destination!")
            break

    elapsed = time.time() - t0
    fps = len(frames) / elapsed if elapsed > 0 else 0
    print(f"  {len(frames)} frames in {elapsed:.2f}s ({fps:.1f} FPS)")

    # Save GIF
    if frames:
        imageio.mimsave(output_path, frames, duration=120)
        size_kb = os.path.getsize(output_path) // 1024
        print(f"Animation saved: {output_path} ({size_kb} KB, {len(frames)} frames)")

    # Single high-res frame
    single_path = os.path.splitext(output_path)[0] + '_frame.png'
    if frames:
        env2 = MockEnv(nrow=nrow, ncol=ncol)
        renderer2 = Render3D(env2, headless=True)
        for s in range(nsteps):
            env2.step()
            renderer2.record_trajectory()
            if env2.done:
                break
        fig2 = _render_frame(env2, renderer2, nsteps, nsteps)
        fig2.savefig(single_path, dpi=150, bbox_inches='tight',
                     facecolor='#000000')
        plt.close(fig2)
        size_kb = os.path.getsize(single_path) // 1024
        print(f"Single frame saved: {single_path} ({size_kb} KB)")

    return output_path


def run_live(output_path='frame3d.png', envtype='1s-d'):
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
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo')
    parser.add_argument('--output', default='animation_3d.gif')
    parser.add_argument('--nsteps', type=int, default=30)
    parser.add_argument('--nrow', type=int, default=40)
    parser.add_argument('--ncol', type=int, default=40)
    parser.add_argument('--envtype', default='1s-d')
    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo(output_path=args.output, nsteps=args.nsteps,
                 nrow=args.nrow, ncol=args.ncol)
    elif args.mode == 'live':
        run_live(output_path=args.output, envtype=args.envtype)


if __name__ == '__main__':
    main()
