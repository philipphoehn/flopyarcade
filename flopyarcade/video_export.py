"""
FloPyArcade Video Export Module — V4 ROADMAP.

Renders 3D particle trajectory animations as MP4 (ffmpeg) or GIF (pillow fallback)
using matplotlib.animation.FuncAnimation.

Usage:
    from flopyarcade.video_export import VideoConfig, VideoExporter, generate_demo_frames

    config = VideoConfig(fps=30, duration_sec=5.0)
    exporter = VideoExporter(config)

    for frame in generate_demo_frames(n_frames=30):
        exporter.add_frame(
            t=frame.t,
            camera_angle=frame.camera_angle,
            particle_positions=frame.particle_positions,
            head_field=frame.head_field,
            active_wells=frame.active_wells,
        )

    path = exporter.render_matplotlib('/tmp/trajectory.mp4')
    print(format_video_report(exporter))
"""

from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any, ClassVar

import numpy as np


# ─── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class VideoConfig:
    """Configuration for video export."""

    fps: int = 30
    resolution: Tuple[int, int] = (1920, 1080)
    duration_sec: float = 5.0
    title: str = 'FloPyArcade'
    codec: str = 'libx264'

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['resolution'] = list(self.resolution)
        return d


@dataclass
class FrameData:
    """A single frame's data snapshot."""

    t: int
    camera_angle: Tuple[float, float] = (30.0, 45.0)
    particle_positions: Optional[List[List[float]]] = None
    head_field: Optional[List[List[float]]] = None
    active_wells: Optional[List[int]] = None


# ─── VideoExporter ─────────────────────────────────────────────────────────────


class VideoExporter:
    """Build and render 3D trajectory videos from particle simulation data."""

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self._frames: List[FrameData] = []

    # ─── Frame management ─────────────────────────────────────────────────────

    def add_frame(
        self,
        t: int,
        camera_angle: Optional[Tuple[float, float]] = None,
        particle_positions: Optional[List[List[float]]] = None,
        head_field: Optional[List[List[float]]] = None,
        active_wells: Optional[List[int]] = None,
    ) -> None:
        """Append a frame to the animation sequence."""
        frame = FrameData(
            t=t,
            camera_angle=camera_angle if camera_angle is not None else (30.0, 45.0),
            particle_positions=particle_positions,
            head_field=head_field,
            active_wells=active_wells,
        )
        self._frames.append(frame)

    def n_frames(self) -> int:
        """Return the number of frames added."""
        return len(self._frames)

    def total_duration(self) -> float:
        """Return total video duration in seconds (n_frames / fps)."""
        if self.config.fps <= 0:
            return 0.0
        return self.n_frames() / self.config.fps

    # ─── Animation frame building ─────────────────────────────────────────────

    def build_animation_frames(self) -> List[Dict[str, Any]]:
        """Convert stored frames into animation-ready dicts.

        Returns a list of dicts, each with keys:
            t, x, y, z (particle arrays), head_grid (2D array or None),
            camera_azimuth, camera_elevation
        """
        result: List[Dict[str, Any]] = []
        for fr in self._frames:
            x_arr, y_arr, z_arr = np.array([]), np.array([]), np.array([])
            if fr.particle_positions:
                pts = np.array(fr.particle_positions)
                if pts.ndim == 2 and pts.shape[1] >= 3:
                    x_arr = pts[:, 0]
                    y_arr = pts[:, 1]
                    z_arr = pts[:, 2]
            head_grid = None
            if fr.head_field:
                head_grid = np.array(fr.head_field)

            result.append({
                't': fr.t,
                'x': x_arr,
                'y': y_arr,
                'z': z_arr,
                'head_grid': head_grid,
                'camera_azimuth': fr.camera_angle[0],
                'camera_elevation': fr.camera_angle[1],
            })
        return result

    # ─── Rendering ────────────────────────────────────────────────────────────

    def render_matplotlib(self, path: str) -> str:
        """Render 3D trajectory animation using matplotlib.

        Uses FuncAnimation with a 3D scatter plot and particle trail.
        If a head grid is present, draws a contour surface at the bottom.
        Writes MP4 via ffmpeg writer if available, otherwise GIF via pillow.

        Args:
            path: Output file path (extension determines format if possible).

        Returns:
            The output file path actually written.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        frames = self.build_animation_frames()
        if not frames:
            raise ValueError("No frames to render. Call add_frame() first.")

        fig_w, fig_h = self.config.resolution
        fig = plt.figure(figsize=(fig_w / 100, fig_h / 100))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.config.title, fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Particle scatter + trail (init with single dummy point to avoid
        # matplotlib 3D backend bug with empty offsets in _scale_invalid_mask)
        _init_pts = frames[0] if len(frames) > 0 else None
        if _init_pts is not None and len(_init_pts['x']) > 0:
            scatter = ax.scatter(_init_pts['x'], _init_pts['y'], _init_pts['z'],
                                c='#00ffff', s=30, alpha=0.9, zorder=5)
            trail_line, = ax.plot(_init_pts['x'], _init_pts['y'], _init_pts['z'],
                                 c='#ff00cc', lw=1.5, alpha=0.6, zorder=4)
        else:
            scatter = ax.scatter([0.0], [0.0], [0.0],
                                c='#00ffff', s=30, alpha=0.9, zorder=5)
            trail_line, = ax.plot([0.0], [0.0], [0.0],
                                 c='#ff00cc', lw=1.5, alpha=0.6, zorder=4)

        # Head contour (optional, at z=0)
        contour = None
        if frames[0]['head_grid'] is not None:
            h = frames[0]['head_grid']
            ny, nx = h.shape
            xx, yy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
            contour = ax.contourf(xx, yy, h, levels=15, zdir='z', zoffset=0, cmap='terrain', alpha=0.5)

        # Determine max coordinate bounds for stable axis limits
        all_pts = []
        for f in frames:
            if len(f['x']) > 0:
                all_pts.append(np.column_stack([f['x'], f['y'], f['z']]))
        if all_pts:
            pts = np.vstack(all_pts)
            xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
            ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
            zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
            margin = max(1.0, 0.1 * (max(xmax - xmin, ymax - ymin, zmax - zmin)))
            ax.set_xlim(xmin - margin, xmax + margin)
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_zlim(zmin - margin, zmax + margin)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)

        def update(i: int):
            f = frames[i]
            ax.view_init(elev=f['camera_elevation'], azim=f['camera_azimuth'])

            x, y, z = f['x'], f['y'], f['z']
            if len(x) > 0:
                scatter._offsets3d = (np.asarray(x), np.asarray(y), np.asarray(z))
            else:
                scatter._offsets3d = (np.array([0.0]), np.array([0.0]), np.array([0.0]))

            # Trail: accumulate up to current frame
            trail_xs, trail_ys, trail_zs = [], [], []
            for j in range(i + 1):
                fj = frames[j]
                if len(fj['x']) > 0:
                    trail_xs.extend(fj['x'])
                    trail_ys.extend(fj['y'])
                    trail_zs.extend(fj['z'])
            if trail_xs:
                trail_line.set_data(trail_xs, trail_ys)
                trail_line.set_3d_properties(trail_zs)
            else:
                trail_line.set_data([], [])
                trail_line.set_3d_properties([])

            if contour is not None and f['head_grid'] is not None:
                # Update contour if grid changes shape (rare)
                pass

            ax.set_title(f"{self.config.title}  |  t={f['t']}", fontsize=12)
            return [scatter, trail_line]

        # Choose writer
        ext = os.path.splitext(path)[1].lower()
        want_gif = (ext == '.gif')

        if want_gif:
            writer = matplotlib.animation.PillowWriter(fps=self.config.fps)
            out_path = path if ext else path + '.gif'
        else:
            try:
                writer = matplotlib.animation.FFMpegWriter(
                    fps=self.config.fps,
                    codec=self.config.codec,
                )
                out_path = path if ext in ('.mp4', '.avi', '.mov') else path + '.mp4'
            except (IOError, OSError, RuntimeError):
                writer = matplotlib.animation.PillowWriter(fps=self.config.fps)
                out_path = os.path.splitext(path)[0] + '.gif'

        anim = FuncAnimation(
            fig, update, frames=range(len(frames)),
            interval=1000 // self.config.fps, blit=False,
        )
        anim.save(out_path, writer=writer, dpi=min(self.config.resolution[0] // 10, 150))
        plt.close(fig)
        return out_path

    # ─── File size estimation ─────────────────────────────────────────────────

    def estimate_file_size(self) -> float:
        """Rough MP4/GIF file size estimate in MB.

        Heuristic: ~1 MB per second per 1080p frame at 30fps with simple motion.
        Adjusted by resolution ratio and frame count.
        """
        n = self.n_frames()
        if n == 0:
            return 0.0
        w, h = self.config.resolution
        res_factor = (w * h) / (1920 * 1080)
        # Rough: 0.5 MB per frame at 1080p30 (H.264, moderate motion)
        base_per_frame_mb = 0.5 * res_factor
        return n * base_per_frame_mb

    # ─── Config persistence ───────────────────────────────────────────────────

    def save_config(self, path: str) -> None:
        """Save VideoConfig as JSON."""
        data = self.config.to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> VideoConfig:
        """Load VideoConfig from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        data['resolution'] = tuple(data['resolution'])
        return VideoConfig(**data)


# ─── Demo data generation ──────────────────────────────────────────────────────


def generate_demo_frames(
    n_frames: int = 30,
    n_particles: int = 5,
    grid_size: int = 5,
) -> List[FrameData]:
    """Generate synthetic orbital/linear particle data for testing.

    Particles follow simple parametric paths (circular + linear drift).
    A small grid of head values is generated for each frame.

    Args:
        n_frames: Number of frames to generate.
        n_particles: Number of simulated particles.
        grid_size: Size of the head field grid (grid_size × grid_size).

    Returns:
        List of FrameData with valid particle positions and head fields.
    """
    frames: List[FrameData] = []
    t_span = 2.0 * math.pi

    for i in range(n_frames):
        t = i
        phase = (i / max(n_frames - 1, 1)) * t_span

        # Particle positions: mix of circular orbit and linear drift
        positions: List[List[float]] = []
        for p in range(n_particles):
            # Circular orbit with different radii/speeds
            r = 1.0 + 0.3 * p
            speed = 1.0 + 0.2 * p
            x = r * math.cos(speed * phase + p * 0.7)
            y = r * math.sin(speed * phase + p * 0.7)
            z = 0.1 * math.sin(2.0 * phase + p)
            positions.append([x, y, z])

        # Head field: smooth gradient + slight time variation
        grid = np.zeros((grid_size, grid_size))
        for ri in range(grid_size):
            for ci in range(grid_size):
                grid[ri, ci] = (
                    10.0
                    + 5.0 * (ci / max(grid_size - 1, 1))
                    + 2.0 * math.sin(phase + ri * 0.5)
                )

        # Camera slowly orbits
        cam_az = 30.0 + 360.0 * (i / max(n_frames - 1, 1))
        cam_el = 25.0 + 10.0 * math.sin(phase)

        frames.append(FrameData(
            t=t,
            camera_angle=(cam_az % 360.0, cam_el),
            particle_positions=positions,
            head_field=grid.tolist(),
            active_wells=[0, 1] if i % 3 == 0 else [0],
        ))

    return frames


# ─── Reporting & validation ────────────────────────────────────────────────────


def format_video_report(exporter: VideoExporter) -> str:
    """Generate a human-readable text summary of the export configuration."""
    w, h = exporter.config.resolution
    lines = [
        "═══ FloPyArcade Video Export Report ═══",
        f"  Frames:        {exporter.n_frames()}",
        f"  FPS:           {exporter.config.fps}",
        f"  Duration:      {exporter.total_duration():.2f} s",
        f"  Resolution:    {w}×{h}",
        f"  Codec:         {exporter.config.codec}",
        f"  Title:         {exporter.config.title}",
        f"  Est. Size:     {exporter.estimate_file_size():.1f} MB",
        "═══════════════════════════════════════════",
    ]
    return '\n'.join(lines)


def validate_export(path: str) -> Dict[str, Any]:
    """Validate an exported video file.

    Checks:
        - File exists
        - File size > 0
        - Extension is .mp4 or .gif

    Returns:
        {valid: bool, size_bytes: int, extension: str, error: str|None}
    """
    ext = os.path.splitext(path)[1].lower()

    if not os.path.exists(path):
        return {
            'valid': False,
            'size_bytes': 0,
            'extension': ext,
            'error': f'File not found: {path}',
        }

    size = os.path.getsize(path)
    if size == 0:
        return {
            'valid': False,
            'size_bytes': 0,
            'extension': ext,
            'error': 'File is empty (0 bytes)',
        }

    if ext not in ('.mp4', '.gif', '.avi', '.mov'):
        return {
            'valid': False,
            'size_bytes': size,
            'extension': ext,
            'error': f'Unsupported extension: {ext} (expected .mp4 or .gif)',
        }

    return {
        'valid': True,
        'size_bytes': size,
        'extension': ext,
        'error': None,
    }


# ─── Module-level convenience ──────────────────────────────────────────────────

__all__ = [
    'VideoConfig',
    'FrameData',
    'VideoExporter',
    'generate_demo_frames',
    'format_video_report',
    'validate_export',
]
