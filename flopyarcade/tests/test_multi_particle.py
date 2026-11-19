"""Tests for flopyarcade.multi_particle (A4: Multi-particle swarm per episode)."""

import pytest
import numpy as np

from flopyarcade.multi_particle import (
    ParticleState,
    spawn_particles,
    compute_reward,
    MultiParticleController,
)


# ---------------------------------------------------------------------------
# ParticleState dataclass
# ---------------------------------------------------------------------------

class TestParticleState:
    def test_defaults(self):
        p = ParticleState(index=0)
        assert p.index == 0
        assert p.position == (0.0, 0.0, 0.0)
        assert p.captured is False
        assert p.done is False
        assert p.steps == 0


# ---------------------------------------------------------------------------
# spawn_particles
# ---------------------------------------------------------------------------

class TestSpawnParticles:
    def test_count(self):
        positions = spawn_particles(n=5, extent_x=320, extent_y=640, d_col=10, z_top=20)
        assert len(positions) == 5

    def test_position_shape(self):
        positions = spawn_particles(n=3, extent_x=320, extent_y=640, d_col=10, z_top=20)
        for pos in positions:
            assert len(pos) == 3

    def test_x_position(self):
        x_expected = 320.0 - 1.1 * 10.0
        positions = spawn_particles(n=5, extent_x=320, extent_y=640, d_col=10, z_top=20)
        for pos in positions:
            assert pos[0] == pytest.approx(x_expected)

    def test_z_position(self):
        positions = spawn_particles(n=5, extent_x=320, extent_y=640, d_col=10, z_top=20)
        for pos in positions:
            assert pos[2] == pytest.approx(20.0)

    def test_y_within_buffer(self):
        positions = spawn_particles(n=20, extent_x=320, extent_y=640, d_col=10, z_top=20)
        for pos in positions:
            assert 20.0 <= pos[1] <= 620.0

    def test_seed_reproducibility(self):
        p1 = spawn_particles(n=5, extent_x=320, extent_y=640, d_col=10, z_top=20, seed=42)
        p2 = spawn_particles(n=5, extent_x=320, extent_y=640, d_col=10, z_top=20, seed=42)
        assert p1 == p2

    def test_different_seeds_differ(self):
        p1 = spawn_particles(n=5, extent_x=320, extent_y=640, d_col=10, z_top=20, seed=1)
        p2 = spawn_particles(n=5, extent_x=320, extent_y=640, d_col=10, z_top=20, seed=2)
        assert p1 != p2

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            spawn_particles(n=0, extent_x=320, extent_y=640, d_col=10, z_top=20)

    def test_extent_too_small(self):
        with pytest.raises(ValueError, match="extent_y"):
            spawn_particles(n=1, extent_x=320, extent_y=10, d_col=10, z_top=20)

    def test_single_particle(self):
        positions = spawn_particles(n=1, extent_x=320, extent_y=640, d_col=10, z_top=20)
        assert len(positions) == 1


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_all_captured(self):
        r = compute_reward(captured=5, total=5, base_reward=1.0)
        assert r == pytest.approx(1.0)

    def test_half_captured(self):
        r = compute_reward(captured=2, total=4, base_reward=1.0)
        assert r == pytest.approx(0.5)

    def test_none_captured(self):
        r = compute_reward(captured=0, total=5, base_reward=1.0)
        assert r == pytest.approx(0.0)

    def test_with_step_penalty(self):
        r = compute_reward(captured=3, total=5, base_reward=1.0, step_penalty=0.1)
        assert r == pytest.approx(0.6 - 0.1)

    def test_custom_base(self):
        r = compute_reward(captured=2, total=4, base_reward=10.0)
        assert r == pytest.approx(5.0)

    def test_total_zero_raises(self):
        with pytest.raises(ValueError, match="total must be >= 1"):
            compute_reward(captured=0, total=0)

    def test_captured_negative_raises(self):
        with pytest.raises(ValueError, match="captured must be >= 0"):
            compute_reward(captured=-1, total=5)

    def test_captured_exceeds_total(self):
        with pytest.raises(ValueError, match="cannot exceed total"):
            compute_reward(captured=6, total=5)

    def test_returns_float(self):
        r = compute_reward(captured=1, total=2)
        assert isinstance(r, float)


# ---------------------------------------------------------------------------
# MultiParticleController init
# ---------------------------------------------------------------------------

class TestControllerInit:
    def test_defaults(self):
        mc = MultiParticleController()
        assert mc.n_particles == 1
        assert mc.base_reward == 1.0
        assert mc.step_penalty == 0.0
        assert mc.max_steps == 0
        assert len(mc.particles) == 1
        assert mc.current_step == 0

    def test_n_particles(self):
        mc = MultiParticleController(n_particles=5)
        assert mc.n_particles == 5
        assert len(mc.particles) == 5
        for i, p in enumerate(mc.particles):
            assert p.index == i

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="n_particles must be >= 1"):
            MultiParticleController(n_particles=0)

    def test_custom_params(self):
        mc = MultiParticleController(
            n_particles=3, base_reward=5.0, step_penalty=0.01, max_steps=100
        )
        assert mc.base_reward == 5.0
        assert mc.step_penalty == 0.01
        assert mc.max_steps == 100


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

class TestPositions:
    def test_set_positions(self):
        mc = MultiParticleController(n_particles=3)
        mc.set_positions([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert mc.particles[0].position == (1, 2, 3)
        assert mc.particles[2].position == (7, 8, 9)

    def test_set_positions_wrong_length(self):
        mc = MultiParticleController(n_particles=3)
        with pytest.raises(ValueError, match="Expected 3 positions"):
            mc.set_positions([[1, 2, 3]])

    def test_update_position(self):
        mc = MultiParticleController(n_particles=2)
        mc.update_position(0, [10, 20, 30])
        assert mc.particles[0].position == (10, 20, 30)

    def test_update_position_out_of_range(self):
        mc = MultiParticleController(n_particles=2)
        with pytest.raises(IndexError):
            mc.update_position(5, [1, 2, 3])


# ---------------------------------------------------------------------------
# Capture tracking
# ---------------------------------------------------------------------------

class TestCapture:
    def test_mark_captured(self):
        mc = MultiParticleController(n_particles=3)
        mc.mark_captured([0, 2])
        assert mc.particles[0].captured is True
        assert mc.particles[0].done is True
        assert mc.particles[1].captured is False
        assert mc.particles[2].captured is True
        assert mc.captured_count == 2

    def test_mark_captured_out_of_range(self):
        mc = MultiParticleController(n_particles=2)
        with pytest.raises(IndexError):
            mc.mark_captured([5])

    def test_mark_done(self):
        mc = MultiParticleController(n_particles=3)
        mc.mark_done([1])
        assert mc.particles[1].done is True
        assert mc.particles[1].captured is False

    def test_mark_done_out_of_range(self):
        mc = MultiParticleController(n_particles=2)
        with pytest.raises(IndexError):
            mc.mark_done([9])


# ---------------------------------------------------------------------------
# Step management
# ---------------------------------------------------------------------------

class TestSteps:
    def test_advance_step(self):
        mc = MultiParticleController(n_particles=3)
        mc.advance_step()
        assert mc.current_step == 1
        assert mc.particles[0].steps == 1

    def test_advance_multiple(self):
        mc = MultiParticleController(n_particles=2)
        for _ in range(5):
            mc.advance_step()
        assert mc.current_step == 5
        assert mc.particles[0].steps == 5

    def test_done_particle_no_steps(self):
        mc = MultiParticleController(n_particles=2)
        mc.mark_done([0])
        mc.advance_step()
        assert mc.particles[0].steps == 0
        assert mc.particles[1].steps == 1


# ---------------------------------------------------------------------------
# Reward via controller
# ---------------------------------------------------------------------------

class TestControllerReward:
    def test_no_captures(self):
        mc = MultiParticleController(n_particles=4)
        assert mc.get_reward() == pytest.approx(0.0)

    def test_partial_captures(self):
        mc = MultiParticleController(n_particles=4)
        mc.mark_captured([0, 2])
        assert mc.get_reward() == pytest.approx(0.5)

    def test_all_captures(self):
        mc = MultiParticleController(n_particles=3)
        mc.mark_captured([0, 1, 2])
        assert mc.get_reward() == pytest.approx(1.0)

    def test_with_penalty(self):
        mc = MultiParticleController(n_particles=4, step_penalty=0.05)
        for _ in range(10):
            mc.advance_step()
        mc.mark_captured([0])
        # (1/4)*1.0 - 0.05*10 = 0.25 - 0.5 = -0.25
        assert mc.get_reward() == pytest.approx(-0.25)


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------

class TestEpisodeDone:
    def test_not_done_initially(self):
        mc = MultiParticleController(n_particles=3)
        assert mc.episode_done is False

    def test_all_done(self):
        mc = MultiParticleController(n_particles=2)
        mc.mark_captured([0])
        mc.mark_done([1])
        assert mc.episode_done is True

    def test_max_steps(self):
        mc = MultiParticleController(n_particles=2, max_steps=5)
        for _ in range(5):
            mc.advance_step()
        assert mc.episode_done is True

    def test_under_max_steps(self):
        mc = MultiParticleController(n_particles=2, max_steps=10)
        for _ in range(4):
            mc.advance_step()
        assert mc.episode_done is False

    def test_all_captured_property(self):
        mc = MultiParticleController(n_particles=3)
        assert mc.all_captured is False
        mc.mark_captured([0, 1, 2])
        assert mc.all_captured is True


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        mc = MultiParticleController(n_particles=3)
        mc.mark_captured([0])
        mc.advance_step()
        mc.advance_step()
        mc.reset()
        assert mc.current_step == 0
        assert mc.captured_count == 0
        assert mc.particles[0].captured is False
        assert mc.particles[0].steps == 0

    def test_reset_with_positions(self):
        mc = MultiParticleController(n_particles=2)
        mc.set_positions([[1, 1, 1], [2, 2, 2]])
        mc.mark_captured([0])
        mc.reset(positions=[[5, 5, 5], [6, 6, 6]])
        assert mc.particles[0].position == (5, 5, 5)
        assert mc.captured_count == 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_keys(self):
        mc = MultiParticleController(n_particles=2)
        d = mc.to_dict()
        assert "n_particles" in d
        assert "current_step" in d
        assert "captured_count" in d
        assert "active_count" in d
        assert "episode_done" in d
        assert "reward" in d
        assert "particles" in d
        assert len(d["particles"]) == 2

    def test_repr(self):
        mc = MultiParticleController(n_particles=3)
        r = repr(mc)
        assert "MultiParticleController" in r
        assert "n=3" in r


# ---------------------------------------------------------------------------
# Integration: full episode lifecycle
# ---------------------------------------------------------------------------

class TestEpisodeLifecycle:
    def test_full_episode(self):
        mc = MultiParticleController(n_particles=4, step_penalty=0.01, max_steps=50)
        positions = spawn_particles(n=4, extent_x=320, extent_y=640, d_col=10, z_top=20, seed=7)
        mc.set_positions(positions)

        # Simulate 10 steps
        for step in range(10):
            mc.advance_step()

        # 3 out of 4 captured
        mc.mark_captured([0, 1, 2])

        assert mc.captured_count == 3
        assert mc.active_count == 1
        assert mc.episode_done is False  # particle 3 still active

        # Particle 3 exits domain
        mc.mark_done([3])

        assert mc.episode_done is True
        reward = mc.get_reward()
        # (3/4)*1.0 - 0.01*10 = 0.75 - 0.10 = 0.65
        assert reward == pytest.approx(0.65)

    def test_backward_compat_single(self):
        """n_particles=1 should behave like single-particle."""
        mc = MultiParticleController(n_particles=1)
        mc.advance_step()
        mc.mark_captured([0])
        assert mc.episode_done is True
        assert mc.get_reward() == pytest.approx(1.0)
