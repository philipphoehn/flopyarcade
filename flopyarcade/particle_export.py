"""Particle trajectory export as .npz (ROADMAP P4).

Exports MODPATH particle paths as (T × P × 3) numpy arrays,
supports stats computation, resampling, comparison, and reporting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


@dataclass
class ParticleFrame:
    """A single timestep of particle positions."""

    t: int
    positions: list[list[float]]  # list of [x, y, z] per particle


@dataclass
class ParticleTrajectory:
    """Full particle trajectory across timesteps."""

    envtype: str
    seed: int
    n_particles: int
    n_timesteps: int
    frames: list[ParticleFrame] = field(default_factory=list)


def export_trajectory_npz(trajectory: ParticleTrajectory, path: str) -> str:
    """Write trajectory to an .npz file.

    Keys: positions (T×P×3 float64), timesteps (T, int64),
    n_particles (int), envtype (str), seed (int), timestamp (str)
    """
    # Build T×P×3 array
    T = trajectory.n_timesteps
    P = trajectory.n_particles
    positions = np.zeros((T, P, 3), dtype=np.float64)
    timesteps = np.zeros(T, dtype=np.int64)

    for i, frame in enumerate(trajectory.frames):
        timesteps[i] = frame.t
        for j, pos in enumerate(frame.positions):
            positions[i, j] = pos

    timestamp = datetime.now(timezone.utc).isoformat()

    # Ensure directory exists
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    np.savez(
        path,
        positions=positions,
        timesteps=timesteps,
        n_particles=np.int64(P),
        envtype=np.array(trajectory.envtype),
        seed=np.int64(trajectory.seed),
        timestamp=np.array(timestamp),
    )
    return path


def load_trajectory_npz(path: str) -> ParticleTrajectory:
    """Read an .npz file back into a ParticleTrajectory."""
    with np.load(path, allow_pickle=True) as data:
        positions = data["positions"]  # (T, P, 3)
        timesteps = data["timesteps"]  # (T,)
        n_particles = int(data["n_particles"])
        envtype = str(data["envtype"])
        seed = int(data["seed"])

    T = positions.shape[0]
    frames = []
    for i in range(T):
        pos_list = positions[i].tolist()
        frames.append(ParticleFrame(t=int(timesteps[i]), positions=pos_list))

    return ParticleTrajectory(
        envtype=envtype,
        seed=seed,
        n_particles=n_particles,
        n_timesteps=T,
        frames=frames,
    )


def compute_stats(trajectory: ParticleTrajectory) -> dict:
    """Compute summary statistics for a trajectory.

    Returns:
        n_particles, n_timesteps, total_path_length, max_displacement,
        min_displacement, mean_speed
    """
    n_particles = trajectory.n_particles
    n_timesteps = trajectory.n_timesteps

    if n_timesteps <= 1:
        return {
            "n_particles": n_particles,
            "n_timesteps": n_timesteps,
            "total_path_length": 0.0,
            "max_displacement": 0.0,
            "min_displacement": 0.0,
            "mean_speed": 0.0,
        }

    displacements = []
    for i in range(n_timesteps - 1):
        frame_curr = trajectory.frames[i]
        frame_next = trajectory.frames[i + 1]
        for p in range(min(len(frame_curr.positions), len(frame_next.positions))):
            a = np.array(frame_curr.positions[p], dtype=np.float64)
            b = np.array(frame_next.positions[p], dtype=np.float64)
            dist = float(np.linalg.norm(b - a))
            displacements.append(dist)

    total_path_length = sum(displacements) if displacements else 0.0
    max_displacement = max(displacements) if displacements else 0.0
    min_displacement = min(displacements) if displacements else 0.0
    mean_speed = total_path_length / n_timesteps

    return {
        "n_particles": n_particles,
        "n_timesteps": n_timesteps,
        "total_path_length": total_path_length,
        "max_displacement": max_displacement,
        "min_displacement": min_displacement,
        "mean_speed": mean_speed,
    }


def resample_trajectory(trajectory: ParticleTrajectory, n_points: int) -> ParticleTrajectory:
    """Linearly interpolate each particle's path to exactly n_points.

    Endpoints are preserved. Works per-particle: each particle's
    (T, 3) positions are resampled to (n_points, 3).
    """
    T = trajectory.n_timesteps
    P = trajectory.n_particles

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if n_points == 1:
        # Just keep first frame repeated
        positions_resampled = np.zeros((1, P, 3), dtype=np.float64)
        for p in range(P):
            positions_resampled[0, p] = trajectory.frames[0].positions[p]
        frames = [ParticleFrame(t=trajectory.frames[0].t, positions=positions_resampled[0].tolist())]
        return ParticleTrajectory(
            envtype=trajectory.envtype,
            seed=trajectory.seed,
            n_particles=P,
            n_timesteps=1,
            frames=frames,
        )

    # Build full array (T, P, 3)
    pos = np.zeros((T, P, 3), dtype=np.float64)
    for i in range(T):
        for p in range(P):
            pos[i, p] = trajectory.frames[i].positions[p]

    # Source indices: 0..T-1, target indices: 0..n_points-1
    src_indices = np.linspace(0, T - 1, num=n_points)

    # Resample each particle independently
    pos_resampled = np.zeros((n_points, P, 3), dtype=np.float64)
    for p in range(P):
        for dim in range(3):
            pos_resampled[:, p, dim] = np.interp(
                src_indices,
                np.arange(T),
                pos[:, p, dim],
            )

    # Timestep labels: linearly spaced from first to last
    t_start = trajectory.frames[0].t
    t_end = trajectory.frames[-1].t
    t_values = np.linspace(t_start, t_end, num=n_points).astype(int)

    frames = []
    for i in range(n_points):
        pos_list = pos_resampled[i].tolist()
        frames.append(ParticleFrame(t=int(t_values[i]), positions=pos_list))

    return ParticleTrajectory(
        envtype=trajectory.envtype,
        seed=trajectory.seed,
        n_particles=P,
        n_timesteps=n_points,
        frames=frames,
    )


def compare_trajectories(traj_a: ParticleTrajectory, traj_b: ParticleTrajectory) -> dict:
    """Compare two trajectories.

    Returns:
        n_a, n_b, mean_distance_per_timestep (if same length else None), shape_match
    """
    n_a = traj_a.n_timesteps
    n_b = traj_b.n_timesteps
    shape_match = (n_a == n_b) and (traj_a.n_particles == traj_b.n_particles)

    mean_distance = None
    if n_a == n_b:
        distances = []
        n_compare = min(n_a, n_b)
        for i in range(n_compare):
            for p in range(min(len(traj_a.frames[i].positions), len(traj_b.frames[i].positions))):
                a = np.array(traj_a.frames[i].positions[p], dtype=np.float64)
                b = np.array(traj_b.frames[i].positions[p], dtype=np.float64)
                distances.append(float(np.linalg.norm(a - b)))
        if distances:
            mean_distance = sum(distances) / len(distances)

    return {
        "n_a": n_a,
        "n_b": n_b,
        "mean_distance_per_timestep": mean_distance,
        "shape_match": shape_match,
    }


def format_export_report(trajectory: ParticleTrajectory, stats: dict) -> str:
    """Format a human-readable export report."""
    lines = [
        "=" * 60,
        "Particle Trajectory Export Report",
        "=" * 60,
        f"Environment : {trajectory.envtype}",
        f"Seed        : {trajectory.seed}",
        f"Particles   : {stats['n_particles']}",
        f"Timesteps   : {stats['n_timesteps']}",
        "-" * 60,
        f"Total path length : {stats['total_path_length']:.6f}",
        f"Max displacement  : {stats['max_displacement']:.6f}",
        f"Min displacement  : {stats['min_displacement']:.6f}",
        f"Mean speed        : {stats['mean_speed']:.6f}",
        "=" * 60,
    ]
    return "\n".join(lines)
