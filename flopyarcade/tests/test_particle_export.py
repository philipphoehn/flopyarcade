"""Tests for flopyarcade.particle_export (ROADMAP P4)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from flopyarcade.particle_export import (
    ParticleFrame,
    ParticleTrajectory,
    compare_trajectories,
    compute_stats,
    export_trajectory_npz,
    format_export_report,
    load_trajectory_npz,
    resample_trajectory,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_trajectory(n_particles=2, n_timesteps=3, offset=0.0) -> ParticleTrajectory:
    """Build a simple trajectory with deterministic positions."""
    frames = []
    for t in range(n_timesteps):
        positions = []
        for p in range(n_particles):
            x = float(t) + float(p) + offset
            y = float(p) * 2.0
            z = float(t * p)
            positions.append([x, y, z])
        frames.append(ParticleFrame(t=t, positions=positions))
    return ParticleTrajectory(
        envtype="1s-d",
        seed=42,
        n_particles=n_particles,
        n_timesteps=n_timesteps,
        frames=frames,
    )


# ------------------------------------------------------------------ #
# Dataclass tests
# ------------------------------------------------------------------ #

def test_particle_frame_creation():
    """Test ParticleFrame creation."""
    frame = ParticleFrame(t=5, positions=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert frame.t == 5
    assert len(frame.positions) == 2
    assert frame.positions[0] == [1.0, 2.0, 3.0]
    assert frame.positions[1] == [4.0, 5.0, 6.0]


def test_particle_trajectory_creation():
    """Test ParticleTrajectory creation."""
    traj = _make_trajectory(n_particles=3, n_timesteps=4)
    assert traj.envtype == "1s-d"
    assert traj.seed == 42
    assert traj.n_particles == 3
    assert traj.n_timesteps == 4
    assert len(traj.frames) == 4
    assert all(len(f.positions) == 3 for f in traj.frames)


# ------------------------------------------------------------------ #
# Export / Load tests
# ------------------------------------------------------------------ #

def test_export_trajectory_npz_creates_file():
    """Test export_trajectory_npz creates file."""
    traj = _make_trajectory()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_traj.npz")
        result = export_trajectory_npz(traj, path)
        assert result == path
        assert os.path.exists(path)
        # Verify npz keys
        with np.load(path, allow_pickle=True) as data:
            assert "positions" in data
            assert "timesteps" in data
            assert "n_particles" in data
            assert "envtype" in data
            assert "seed" in data
            assert "timestamp" in data
            assert data["positions"].shape == (3, 2, 3)


def test_load_trajectory_npz_roundtrip():
    """Test load_trajectory_npz roundtrip (positions preserved)."""
    traj = _make_trajectory(n_particles=2, n_timesteps=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "roundtrip.npz")
        export_trajectory_npz(traj, path)
        loaded = load_trajectory_npz(path)

        assert loaded.envtype == traj.envtype
        assert loaded.seed == traj.seed
        assert loaded.n_particles == traj.n_particles
        assert loaded.n_timesteps == traj.n_timesteps

        for i in range(traj.n_timesteps):
            for p in range(traj.n_particles):
                np.testing.assert_allclose(
                    loaded.frames[i].positions[p],
                    traj.frames[i].positions[p],
                    rtol=1e-10,
                )


def test_export_load_single_particle():
    """Test export/load with single particle."""
    frames = [
        ParticleFrame(t=0, positions=[[0.0, 0.0, 0.0]]),
        ParticleFrame(t=1, positions=[[1.0, 1.0, 1.0]]),
        ParticleFrame(t=2, positions=[[2.0, 2.0, 2.0]]),
    ]
    traj = ParticleTrajectory(
        envtype="2s-d",
        seed=7,
        n_particles=1,
        n_timesteps=3,
        frames=frames,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "single.npz")
        export_trajectory_npz(traj, path)
        loaded = load_trajectory_npz(path)

        assert loaded.n_particles == 1
        assert loaded.n_timesteps == 3
        np.testing.assert_allclose(loaded.frames[1].positions[0], [1.0, 1.0, 1.0])


# ------------------------------------------------------------------ #
# compute_stats tests
# ------------------------------------------------------------------ #

def test_compute_stats_basic():
    """Test compute_stats basic (2 particles, 3 timesteps)."""
    traj = _make_trajectory(n_particles=2, n_timesteps=3)
    stats = compute_stats(traj)

    assert stats["n_particles"] == 2
    assert stats["n_timesteps"] == 3
    assert stats["total_path_length"] > 0
    assert stats["max_displacement"] >= stats["min_displacement"]
    assert stats["mean_speed"] > 0
    assert stats["mean_speed"] == pytest.approx(stats["total_path_length"] / 3, rel=1e-10)


def test_compute_stats_total_path_length():
    """Test compute_stats total_path_length correct."""
    # 1 particle moving along x-axis: (0,0,0) -> (1,0,0) -> (3,0,0)
    frames = [
        ParticleFrame(t=0, positions=[[0.0, 0.0, 0.0]]),
        ParticleFrame(t=1, positions=[[1.0, 0.0, 0.0]]),
        ParticleFrame(t=2, positions=[[3.0, 0.0, 0.0]]),
    ]
    traj = ParticleTrajectory(
        envtype="test", seed=0, n_particles=1, n_timesteps=3, frames=frames
    )
    stats = compute_stats(traj)
    # distances: 1.0 + 2.0 = 3.0
    assert stats["total_path_length"] == pytest.approx(3.0)
    assert stats["max_displacement"] == pytest.approx(2.0)
    assert stats["min_displacement"] == pytest.approx(1.0)


def test_compute_stats_single_timestep_zero():
    """Test compute_stats single particle single timestep (zero displacement)."""
    frames = [ParticleFrame(t=0, positions=[[1.0, 2.0, 3.0]])]
    traj = ParticleTrajectory(
        envtype="test", seed=0, n_particles=1, n_timesteps=1, frames=frames
    )
    stats = compute_stats(traj)
    assert stats["total_path_length"] == 0.0
    assert stats["max_displacement"] == 0.0
    assert stats["min_displacement"] == 0.0
    assert stats["mean_speed"] == 0.0


# ------------------------------------------------------------------ #
# resample_trajectory tests
# ------------------------------------------------------------------ #

def test_resample_trajectory_reduces_points():
    """Test resample_trajectory reduces points."""
    traj = _make_trajectory(n_particles=2, n_timesteps=10)
    result = resample_trajectory(traj, 5)
    assert result.n_timesteps == 5
    assert result.n_particles == 2
    assert len(result.frames) == 5


def test_resample_trajectory_increases_points():
    """Test resample_trajectory increases points."""
    traj = _make_trajectory(n_particles=2, n_timesteps=3)
    result = resample_trajectory(traj, 10)
    assert result.n_timesteps == 10
    assert result.n_particles == 2
    assert len(result.frames) == 10


def test_resample_preserves_endpoints():
    """Test resample_trajectory preserves endpoints."""
    traj = _make_trajectory(n_particles=2, n_timesteps=5)
    result = resample_trajectory(traj, 10)

    # First frame positions should match
    np.testing.assert_allclose(
        result.frames[0].positions,
        traj.frames[0].positions,
        rtol=1e-10,
    )
    # Last frame positions should match
    np.testing.assert_allclose(
        result.frames[-1].positions,
        traj.frames[-1].positions,
        rtol=1e-10,
    )


# ------------------------------------------------------------------ #
# compare_trajectories tests
# ------------------------------------------------------------------ #

def test_compare_trajectories_same_length():
    """Test compare_trajectories same length."""
    traj_a = _make_trajectory(n_particles=2, n_timesteps=3)
    traj_b = _make_trajectory(n_particles=2, n_timesteps=3, offset=1.0)
    result = compare_trajectories(traj_a, traj_b)

    assert result["n_a"] == 3
    assert result["n_b"] == 3
    assert result["shape_match"] is True
    assert result["mean_distance_per_timestep"] is not None
    assert result["mean_distance_per_timestep"] > 0


def test_compare_trajectories_different_lengths():
    """Test compare_trajectories different lengths."""
    traj_a = _make_trajectory(n_particles=2, n_timesteps=3)
    traj_b = _make_trajectory(n_particles=2, n_timesteps=5)
    result = compare_trajectories(traj_a, traj_b)

    assert result["n_a"] == 3
    assert result["n_b"] == 5
    assert result["shape_match"] is False
    assert result["mean_distance_per_timestep"] is None


# ------------------------------------------------------------------ #
# format_export_report tests
# ------------------------------------------------------------------ #

def test_format_export_report_contains_text():
    """Test format_export_report contains text."""
    traj = _make_trajectory(n_particles=2, n_timesteps=3)
    stats = compute_stats(traj)
    report = format_export_report(traj, stats)

    assert "Particle Trajectory Export Report" in report
    assert "1s-d" in report
    assert "42" in report
    assert "Total path length" in report
    assert "Mean speed" in report
    assert isinstance(report, str)
    assert len(report) > 50


# ------------------------------------------------------------------ #
# Scale test
# ------------------------------------------------------------------ #

def test_5_particles_10_timesteps():
    """Test 5 particles 10 timesteps works end-to-end."""
    traj = _make_trajectory(n_particles=5, n_timesteps=10)
    stats = compute_stats(traj)
    assert stats["n_particles"] == 5
    assert stats["n_timesteps"] == 10
    assert stats["total_path_length"] > 0

    # Resample
    resampled = resample_trajectory(traj, 7)
    assert resampled.n_timesteps == 7

    # Export / load roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "scale.npz")
        export_trajectory_npz(traj, path)
        loaded = load_trajectory_npz(path)
        assert loaded.n_particles == 5
        assert loaded.n_timesteps == 10
        np.testing.assert_allclose(
            loaded.frames[5].positions[3],
            traj.frames[5].positions[3],
            rtol=1e-10,
        )

    # Compare with self
    cmp = compare_trajectories(traj, traj)
    assert cmp["shape_match"] is True
    assert cmp["mean_distance_per_timestep"] == pytest.approx(0.0)
