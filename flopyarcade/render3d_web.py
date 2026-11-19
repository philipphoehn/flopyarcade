#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade - Browser 3D Viewer (Three.js / WebGL)
# V5: Self-contained HTML+Three.js 3D visualisation of the groundwater environment

"""Browser-based 3D visualisation for FloPyArcade.

Generates a single self-contained HTML file that renders:
- Head surface as a vertex-coloured mesh (terrain colormap)
- Wells as neon cylinders
- Agent trajectory as a glowing line
- OrbitControls for interactive rotation / zoom / pan

No server required — open the HTML file in any modern browser.
Three.js is loaded from CDN (unpkg.com).

Usage
-----
    from flopyarcade.render3d_web import Render3DWeb

    viewer = Render3DWeb(env, trajectory=trajectory_list, wells=wells_list)
    viewer.save('flopyarcade_3d.html')

CLI
---
    python -m flopyarcade.render3d_web --env 1s-d --out viewer.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# --- Colormap (terrain-like, 256 entries, normalised 0-1) ---

def _terrain_colormap(n: int = 256) -> list:
    """Generate a terrain-like colormap: deep blue → green → yellow → brown → white."""
    stops = [
        (0.0,   0.02, 0.05, 0.25),   # deep ocean
        (0.15,  0.0,  0.2,  0.5),    # deep water
        (0.3,   0.0,  0.5,  0.7),    # shallow water
        (0.45,  0.1,  0.6, 0.4),     # wet sand
        (0.55,  0.2,  0.8, 0.3),     # grass
        (0.65,  0.4, 0.85, 0.2),     # light grass
        (0.75,  0.7, 0.8, 0.2),      # yellow
        (0.85,  0.8, 0.5, 0.2),      # brown
        (0.95,  0.9, 0.8, 0.7),      # light brown
        (1.0,   1.0, 1.0, 1.0),      # white (snow)
    ]
    t = np.linspace(0, 1, n)
    colors = np.zeros((n, 3))
    for i in range(len(stops) - 1):
        t0, r0, g0, b0 = stops[i]
        t1, r1, g1, b1 = stops[i + 1]
        mask = (t >= t0) & (t <= t1)
        if t1 == t0:
            frac = np.zeros_like(t)
        else:
            frac = np.clip((t - t0) / (t1 - t0), 0, 1)
        colors[mask, 0] = r0 + (r1 - r0) * frac[mask]
        colors[mask, 1] = g0 + (g1 - g0) * frac[mask]
        colors[mask, 2] = b0 + (b1 - b0) * frac[mask]
    return colors.tolist()


# --- HTML Template ---

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FloPyArcade 3D Viewer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0f; overflow: hidden; font-family: 'Segoe UI', sans-serif; }
  #canvas { width: 100vw; height: 100vh; display: block; }
  #hud {
    position: fixed; top: 12px; left: 12px; color: #00ffcc;
    font-size: 13px; pointer-events: none; z-index: 10;
    text-shadow: 0 0 8px #00ffcc44;
  }
  #hud h1 { font-size: 16px; margin-bottom: 4px; }
  #hud span { opacity: 0.7; }
  #controls-hint {
    position: fixed; bottom: 12px; left: 50%; transform: translateX(-50%);
    color: #ffffff66; font-size: 11px; pointer-events: none; z-index: 10;
  }
</style>
</head>
<body>
<div id="hud">
  <h1>⬢ FloPyArcade 3D</h1>
  <span id="info"></span>
</div>
<div id="controls-hint">Left-drag: rotate · Right-drag: pan · Scroll: zoom</div>
<canvas id="canvas"></canvas>

<script type="importmap">
{ "imports": { "three": "https://unpkg.com/three@0.164.1/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.164.1/examples/jsm/" } }
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ─── Embedded data ───
const DATA = __DATA_JSON__;

const nrow = DATA.nrow, ncol = DATA.ncol;
const extentX = DATA.extentX, extentY = DATA.extentY;
const zTop = DATA.zTop, zBot = DATA.zBot;
const zScale = DATA.zScale;
const colormap = DATA.colormap; // array of [r,g,b] 0-1

// ─── Scene ───
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0a0a0f);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0a0f, 0.0008);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 50000);
camera.position.set(extentX * 0.7, Math.max(zTop - zBot, 50) * 1.5 + zTop, extentY * 0.7);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(extentX / 2, (zTop + zBot) / 2, extentY / 2);

// ─── Lighting ───
const ambient = new THREE.AmbientLight(0x404060, 0.6);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(extentX, (zTop - zBot) * 2 + zTop, extentY);
scene.add(dirLight);
const neonLight = new THREE.PointLight(0x00ffcc, 2, extentX * 2);
neonLight.position.set(extentX / 2, zTop + 20, extentY / 2);
scene.add(neonLight);

// ─── Head surface mesh ───
function buildSurface() {
  const heads = DATA.heads; // shape (nrow, ncol)
  const hMin = DATA.headMin, hMax = DATA.headMax;
  const hRange = Math.max(hMax - hMin, 1e-6);

  // Build geometry: (nrow+1) x (ncol+1) vertices
  const rows = nrow + 1, cols = ncol + 1;
  const positions = new Float32Array(rows * cols * 3);
  const colors = new Float32Array(rows * cols * 3);
  const indices = [];

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const x = (j / ncol) * extentX;
      const z = (i / nrow) * extentY;
      // Bilinear head interpolation
      const i0 = Math.max(0, i - 1), i1 = Math.min(nrow - 1, i);
      const j0 = Math.max(0, j - 1), j1 = Math.min(ncol - 1, j);
      const h = 0.25 * (heads[i0][j0] + heads[i0][j1] + heads[i1][j0] + heads[i1][j1]);
      const y = zBot + ((h - hMin) / hRange) * (zTop - zBot);

      const idx = (i * cols + j) * 3;
      positions[idx] = x;
      positions[idx + 1] = y;
      positions[idx + 2] = z;

      // Color from colormap
      const t = Math.max(0, Math.min(1, (h - hMin) / hRange));
      const ci = Math.min(colormap.length - 1, Math.floor(t * colormap.length));
      colors[idx] = colormap[ci][0];
      colors[idx + 1] = colormap[ci][1];
      colors[idx + 2] = colormap[ci][2];
    }
  }

  // Triangulate
  for (let i = 0; i < nrow; i++) {
    for (let j = 0; j < ncol; j++) {
      const a = i * cols + j;
      const b = a + 1;
      const c = a + cols;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    roughness: 0.7,
    metalness: 0.1,
    transparent: true,
    opacity: 0.92,
  });

  const mesh = new THREE.Mesh(geo, mat);
  scene.add(mesh);

  // Wireframe overlay (subtle)
  const wire = new THREE.LineSegments(
    new THREE.WireframeGeometry(geo),
    new THREE.LineBasicMaterial({ color: 0x00ffcc, transparent: true, opacity: 0.06 })
  );
  scene.add(wire);
}

buildSurface();

// ─── Wells ───
function buildWells() {
  const wells = DATA.wells; // [{x, z, q, name}]
  const wellRadius = Math.max(extentX, extentY) * 0.008;
  const wellHeight = (zTop - zBot) * 0.6;

  for (const w of wells) {
    const isPump = w.q > 0;
    const color = isPump ? 0xff00cc : 0x00ff88;
    const geo = new THREE.CylinderGeometry(wellRadius, wellRadius, wellHeight, 16);
    const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.5, roughness: 0.3, metalness: 0.6 });
    const mesh = new THREE.Mesh(geo, mat);
    const hNorm = (w.head - DATA.headMin) / Math.max(DATA.headMax - DATA.headMin, 1e-6);
    const y = zBot + hNorm * (zTop - zBot) + wellHeight / 2;
    mesh.position.set(w.x, y, w.z);
    scene.add(mesh);

    // Glow ring at base
    const ringGeo = new THREE.RingGeometry(wellRadius * 1.5, wellRadius * 2.5, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.4 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(w.x, zBot + 0.1, w.z);
    scene.add(ring);
  }
}

if (DATA.wells && DATA.wells.length > 0) buildWells();

// ─── Trajectory ───
function buildTrajectory() {
  const traj = DATA.trajectory; // [{x, z, y}]
  if (!traj || traj.length < 2) return;

  const points = traj.map(p => new THREE.Vector3(p.x, p.y, p.z));
  const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.3);
  const tubeGeo = new THREE.TubeGeometry(curve, Math.min(points.length * 3, 500), 0.8, 8, false);
  const tubeMat = new THREE.MeshStandardMaterial({
    color: 0x00ffff,
    emissive: 0x00ffff,
    emissiveIntensity: 0.8,
    roughness: 0.2,
    metalness: 0.8,
  });
  const tube = new THREE.Mesh(tubeGeo, tubeMat);
  scene.add(tube);

  // Start / end markers
  const startGeo = new THREE.SphereGeometry(2, 16, 16);
  const startMat = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
  const start = new THREE.Mesh(startGeo, startMat);
  start.position.copy(points[0]);
  scene.add(start);

  const endGeo = new THREE.SphereGeometry(2, 16, 16);
  const endMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
  const end = new THREE.Mesh(endGeo, endMat);
  end.position.copy(points[points.length - 1]);
  scene.add(end);
}

buildTrajectory();

// ─── Grid helper ───
const grid = new THREE.GridHelper(Math.max(extentX, extentY), 20, 0x222244, 0x111133);
grid.position.y = zBot - 0.5;
grid.position.x = extentX / 2;
grid.position.z = extentY / 2;
scene.add(grid);

// ─── HUD info ───
document.getElementById('info').textContent =
  `Grid: ${nrow}×${ncol} · Extent: ${extentX}×${extentY} · Heads: ${DATA.headMin.toFixed(2)}–${DATA.headMax.toFixed(2)}`;

// ─── Animation loop ───
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ─── Resize ───
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""


class Render3DWeb:
    """Generate a self-contained Three.js HTML 3D viewer.

    Parameters
    ----------
    env : object
        A FloPyArcade environment instance (must have ``nrow``, ``ncol``,
        ``extentX``, ``extentY``, ``heads``).
    trajectory : list of dict, optional
        Each dict has keys ``x``, ``z`` (or ``y``).  If ``y`` (height) is
        missing it is computed from the head surface.
    wells : list of dict, optional
        Each dict has keys ``x``, ``z``, ``q`` (positive = pump),
        optional ``name``, ``head``.
    z_top : float, optional
        Override top elevation (default: ``env.zTop`` or 50).
    z_bot : float, optional
        Override bottom elevation (default: ``env.zBot`` or 0).
    """

    def __init__(
        self,
        env,
        trajectory: Optional[Sequence[dict]] = None,
        wells: Optional[Sequence[dict]] = None,
        z_top: Optional[float] = None,
        z_bot: Optional[float] = None,
    ):
        self.env = env
        self.nrow = int(env.nrow)
        self.ncol = int(env.ncol)
        self.extentX = float(getattr(env, 'extentX', 100.0))
        self.extentY = float(getattr(env, 'extentY', 100.0))
        self.zTop = float(z_top if z_top is not None else getattr(env, 'zTop', 50.0))
        self.zBot = float(z_bot if z_bot is not None else getattr(env, 'zBot', 0.0))

        heads = np.asarray(env.heads, dtype=np.float64)
        # Ensure shape is (nrow, ncol)
        if heads.shape != (self.nrow, self.ncol):
            from scipy.ndimage import zoom
            factor = (self.nrow / heads.shape[0], self.ncol / heads.shape[1])
            heads = zoom(heads, factor, order=1)
        self.heads = heads
        self.headMin = float(heads.min())
        self.headMax = float(heads.max())

        self.trajectory = self._normalise_trajectory(trajectory)
        self.wells = self._normalise_wells(wells)

    def _normalise_trajectory(self, traj) -> list:
        """Ensure trajectory points have x, y, z."""
        if not traj:
            return []
        result = []
        for p in traj:
            x = float(p.get('x', p.get('col', 0)))
            z = float(p.get('z', p.get('row', 0)))
            if 'y' in p:
                y = float(p['y'])
            else:
                # Estimate from head at nearest cell
                i = min(max(int(z / self.extentY * self.nrow), 0), self.nrow - 1)
                j = min(max(int(x / self.extentX * self.ncol), 0), self.ncol - 1)
                h = self.heads[i, j]
                t = (h - self.headMin) / max(self.headMax - self.headMin, 1e-6)
                y = self.zBot + t * (self.zTop - self.zBot)
            result.append({'x': x, 'y': y, 'z': z})
        return result

    def _normalise_wells(self, wells) -> list:
        """Ensure well dicts have x, z, q, head."""
        if not wells:
            return []
        result = []
        for w in wells:
            result.append({
                'x': float(w.get('x', 0)),
                'z': float(w.get('z', 0)),
                'q': float(w.get('q', 0)),
                'head': float(w.get('head', (self.headMin + self.headMax) / 2)),
                'name': str(w.get('name', '')),
            })
        return result

    def to_dict(self) -> dict:
        """Serialise viewer data to a JSON-compatible dict."""
        return {
            'nrow': self.nrow,
            'ncol': self.ncol,
            'extentX': self.extentX,
            'extentY': self.extentY,
            'zTop': self.zTop,
            'zBot': self.zBot,
            'zScale': self.zTop - self.zBot,
            'heads': self.heads.tolist(),
            'headMin': self.headMin,
            'headMax': self.headMax,
            'trajectory': self.trajectory,
            'wells': self.wells,
            'colormap': _terrain_colormap(128),
        }

    def to_html(self) -> str:
        """Render the full HTML string."""
        data_json = json.dumps(self.to_dict(), separators=(',', ':'))
        return _HTML_TEMPLATE.replace('__DATA_JSON__', data_json)

    def save(self, path: str) -> Path:
        """Write the HTML file to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_html(), encoding='utf-8')
        return p


def make_viewer(
    env,
    trajectory: Optional[Sequence[dict]] = None,
    wells: Optional[Sequence[dict]] = None,
    out_path: str = 'flopyarcade_3d.html',
) -> Path:
    """Convenience: create viewer and save in one call."""
    viewer = Render3DWeb(env, trajectory=trajectory, wells=wells)
    return viewer.save(out_path)


# --- CLI ---

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FloPyArcade 3D Browser Viewer')
    parser.add_argument('--out', default='flopyarcade_3d.html', help='Output HTML path')
    parser.add_argument('--nrow', type=int, default=10)
    parser.add_argument('--ncol', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Generating 3D viewer: {args.nrow}x{args.ncol} grid")

    # Synthetic env for demo (no MODFLOW needed)
    class _DemoEnv:
        def __init__(self, nrow, ncol, seed):
            rng = np.random.default_rng(seed)
            self.nrow = nrow
            self.ncol = ncol
            self.extentX = 1000.0
            self.extentY = 1000.0
            self.zTop = 50.0
            self.zBot = 0.0
            # Synthetic head field: gradient + noise
            x = np.linspace(0, 1, ncol)
            y = np.linspace(0, 1, nrow)
            X, Y = np.meshgrid(x, y)
            self.heads = 30 + 20 * X + 5 * np.sin(3 * Y) + rng.normal(0, 2, (nrow, ncol))

    env = _DemoEnv(args.nrow, args.ncol, args.seed)

    # Demo trajectory
    rng = np.random.default_rng(7)
    traj = []
    x, z = 50.0, 500.0
    for step in range(30):
        traj.append({'x': x, 'z': z})
        x += rng.uniform(10, 40)
        z += rng.uniform(-30, 30)
        x = min(x, 950)
        z = max(min(z, 950), 50)

    # Demo wells
    wells = [
        {'x': 200, 'z': 300, 'q': 5.0, 'name': 'P1'},
        {'x': 700, 'z': 700, 'q': -3.0, 'name': 'W1'},
    ]

    path = make_viewer(env, trajectory=traj, wells=wells, out_path=args.out)
    print(f"Saved: {path}")
    print(f"Size: {path.stat().st_size / 1024:.1f} KB")
    print("Open in browser: file://" + str(path.resolve()))
