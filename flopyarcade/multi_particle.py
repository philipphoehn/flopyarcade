"""A4: Multi-particle / particle swarm per episode.

Extends the single-particle FloPyArcade environment to support N
simultaneous particles spawned along the western boundary. The agent's
goal is to guide the flow field so that as many particles as possible
reach the capture well within the step limit.

Reward is proportional to the capture ratio:
    reward = (captured / total) * base_reward

**ENVTYPE Naming Convention**

Single-particle environments keep their existing names unchanged:
    ``1s-d``, ``2s-c``, ``3r-d``, etc.

Multi-particle environments append a ``-Np`` suffix:
    ``1s-d-2p``  (2 particles, 1 source, divergent)
    ``1s-d-5p``  (5 particles, 1 source, divergent)
    ``2s-c-3p``  (3 particles, 2 sources, convergent)
    ``3r-d-10p`` (10 particles, 3 recharge, divergent)

Use :func:`make_envtype` to build the string and :func:`parse_envtype`
to decompose it. Both are idempotent for single-particle names.

This module provides the state management, spawn logic, reward
calculation, and envtype naming helpers. It is designed to be plugged
into the existing ``modpath_runner.py`` and ``env_base.py`` pipeline
without modifying the single-particle path (backward compatible:
``n_particles=1`` behaves identically to the current implementation).

Usage::

    from flopyarcade.multi_particle import (
        MultiParticleController, spawn_particles, compute_reward,
        make_envtype, parse_envtype,
    )

    # Naming
    make_envtype('1s-d', 2)   # -> '1s-d-2p'
    make_envtype('1s-d', 1)   # -> '1s-d'  (no suffix for single)
    parse_envtype('1s-d-5p')  # -> ('1s-d', 5)
    parse_envtype('2s-c')     # -> ('2s-c', 1)

    controller = MultiParticleController(n_particles=5)
    positions = spawn_particles(
        n=5, extent_x=320.0, extent_y=640.0,
        d_col=10.0, z_top=20.0,
    )
    controller.set_positions(positions)

    # After each step, mark particles that reached the well:
    controller.mark_captured([0, 2])
    reward = controller.get_reward(base_reward=1.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from random import uniform


# ---------------------------------------------------------------------------
# Envtype naming helpers
# ---------------------------------------------------------------------------

_SINGLE_PARTICLE_RE = re.compile(r'^\d+[sr]-[dc]$')
_MULTI_PARTICLE_RE = re.compile(r'^(\d+[sr]-[dc])-(\d+)p$')


def make_envtype(base: str, n_particles: int) -> str:
    """Build an envtype string with optional particle-count suffix.

    Single-particle environments (``n_particles == 1``) return the
    base name unchanged.  Multi-particle environments get a ``-Np``
    suffix appended.

    Args:
        base: Base envtype, e.g. ``'1s-d'``, ``'2s-c'``, ``'3r-d'``.
        n_particles: Number of particles (>= 1).

    Returns:
        Envtype string.  ``'1s-d'`` for n=1, ``'1s-d-5p'`` for n=5.

    Raises:
        ValueError: If base is not a valid envtype or n_particles < 1.

    Examples::

        make_envtype('1s-d', 1)   # -> '1s-d'
        make_envtype('1s-d', 2)   # -> '1s-d-2p'
        make_envtype('3r-d', 10)  # -> '3r-d-10p'
    """
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")
    if not _SINGLE_PARTICLE_RE.match(base):
        raise ValueError(
            f"base envtype '{base}' is not a valid single-particle envtype "
            f"(expected pattern like '1s-d', '2s-c', '3r-d')"
        )
    if n_particles == 1:
        return base
    return f"{base}-{n_particles}p"


def parse_envtype(envtype: str) -> Tuple[str, int]:
    """Decompose an envtype string into (base, n_particles).

    Args:
        envtype: Envtype string, e.g. ``'1s-d'``, ``'1s-d-5p'``.

    Returns:
        Tuple of (base_envtype, n_particles).

    Raises:
        ValueError: If envtype is not recognised.

    Examples::

        parse_envtype('1s-d')      # -> ('1s-d', 1)
        parse_envtype('1s-d-5p')   # -> ('1s-d', 5)
        parse_envtype('3r-d-10p')  # -> ('3r-d', 10)
    """
    m = _SINGLE_PARTICLE_RE.match(envtype)
    if m:
        return (envtype, 1)
    m = _MULTI_PARTICLE_RE.match(envtype)
    if m:
        return (m.group(1), int(m.group(2)))
    raise ValueError(
        f"Unrecognised envtype '{envtype}'. "
        f"Expected single-particle (e.g. '1s-d') or "
        f"multi-particle suffix form (e.g. '1s-d-5p')."
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParticleState:
    """Per-particle tracking state.

    Attributes:
        index: Particle identifier (0-based).
        position: Current (x, y, z) position.
        captured: Whether the particle has reached the well.
        done: Whether the particle's episode is finished.
        steps: Number of steps this particle has been tracked.
    """
    index: int
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    captured: bool = False
    done: bool = False
    steps: int = 0


# ---------------------------------------------------------------------------
# Spawn logic
# ---------------------------------------------------------------------------

def spawn_particles(
    n: int,
    extent_x: float,
    extent_y: float,
    d_col: float,
    z_top: float,
    buffer_y: float = 20.0,
    seed: Optional[int] = None,
) -> List[List[float]]:
    """Spawn N particles along the western boundary.

    Each particle is placed at a random Y position within the buffer
    margins, at the same X offset used by the single-particle spawn,
    and at the top Z boundary.

    Args:
        n: Number of particles to spawn (>= 1).
        extent_x: Domain extent in X (model units).
        extent_y: Domain extent in Y (model units).
        d_col: Column width (grid spacing).
        z_top: Top boundary in Z.
        buffer_y: Vertical buffer from Y boundaries.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of [x, y, z] positions, one per particle.

    Raises:
        ValueError: If n < 1 or extent_y is too small for the buffer.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if extent_y < 2 * buffer_y:
        raise ValueError(
            f"extent_y ({extent_y}) must exceed 2*buffer_y ({2*buffer_y})"
        )

    rng = np.random.default_rng(seed)
    x = extent_x - 1.1 * d_col
    positions = []
    for _ in range(n):
        y = float(rng.uniform(buffer_y, extent_y - buffer_y))
        positions.append([x, y, z_top])
    return positions


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(
    captured: int,
    total: int,
    base_reward: float = 1.0,
    step_penalty: float = 0.0,
) -> float:
    """Compute the multi-particle reward.

    Reward = (captured / total) * base_reward - step_penalty

    Args:
        captured: Number of particles that reached the well.
        total: Total number of particles spawned.
        base_reward: Maximum reward when all particles are captured.
        step_penalty: Optional per-step penalty (e.g., 0.001 * steps).

    Returns:
        Float reward value.

    Raises:
        ValueError: If total < 1 or captured < 0 or captured > total.
    """
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    if captured < 0:
        raise ValueError(f"captured must be >= 0, got {captured}")
    if captured > total:
        raise ValueError(f"captured ({captured}) cannot exceed total ({total})")

    ratio = captured / total
    return float(ratio * base_reward - step_penalty)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MultiParticleController:
    """Manages N particle states for a single episode.

    Tracks per-particle position, capture status, and step count.
    Provides the aggregate reward and episode-termination logic.

    Args:
        n_particles: Number of particles in this episode.
        base_reward: Base reward scaled by capture ratio.
        step_penalty: Per-step penalty applied to reward.
        max_steps: Episode terminates after this many steps (0 = unlimited).

    Example::

        mc = MultiParticleController(n_particles=3)
        mc.set_positions([[10, 20, 5], [10, 50, 5], [10, 80, 5]])
        mc.advance_step()
        mc.mark_captured([1])
        print(mc.get_reward())  # (1/3) * 1.0 - penalty
        print(mc.episode_done)  # False (not all captured)
    """

    def __init__(
        self,
        n_particles: int = 1,
        base_reward: float = 1.0,
        step_penalty: float = 0.0,
        max_steps: int = 0,
    ) -> None:
        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")
        self.n_particles = n_particles
        self.base_reward = base_reward
        self.step_penalty = step_penalty
        self.max_steps = max_steps
        self.particles: List[ParticleState] = [
            ParticleState(index=i) for i in range(n_particles)
        ]
        self.current_step: int = 0
        self._captured_indices: set = set()

    # -- Position management --

    def set_positions(self, positions: List[List[float]]) -> None:
        """Set initial or updated positions for all particles.

        Args:
            positions: List of [x, y, z], length must equal n_particles.

        Raises:
            ValueError: If length mismatch.
        """
        if len(positions) != self.n_particles:
            raise ValueError(
                f"Expected {self.n_particles} positions, got {len(positions)}"
            )
        for i, pos in enumerate(positions):
            self.particles[i].position = tuple(pos)  # type: ignore[arg-type]

    def update_position(self, index: int, position: List[float]) -> None:
        """Update a single particle's position.

        Args:
            index: Particle index (0-based).
            position: New [x, y, z].

        Raises:
            IndexError: If index out of range.
        """
        if index < 0 or index >= self.n_particles:
            raise IndexError(f"Particle index {index} out of range [0, {self.n_particles})")
        self.particles[index].position = tuple(position)  # type: ignore[arg-type]

    # -- Capture tracking --

    def mark_captured(self, indices: List[int]) -> None:
        """Mark one or more particles as captured.

        Args:
            indices: List of particle indices that reached the well.
        """
        for i in indices:
            if i < 0 or i >= self.n_particles:
                raise IndexError(f"Particle index {i} out of range")
            self.particles[i].captured = True
            self.particles[i].done = True
            self._captured_indices.add(i)

    def mark_done(self, indices: List[int]) -> None:
        """Mark particles as done (exited domain, etc.) without capture.

        Args:
            indices: List of particle indices.
        """
        for i in indices:
            if i < 0 or i >= self.n_particles:
                raise IndexError(f"Particle index {i} out of range")
            self.particles[i].done = True

    # -- Step management --

    def advance_step(self) -> None:
        """Increment the step counter for all active particles."""
        self.current_step += 1
        for p in self.particles:
            if not p.done:
                p.steps += 1

    # -- Reward --

    @property
    def captured_count(self) -> int:
        """Number of particles that reached the well."""
        return len(self._captured_indices)

    @property
    def active_count(self) -> int:
        """Number of particles still active (not done)."""
        return sum(1 for p in self.particles if not p.done)

    def get_reward(self) -> float:
        """Compute the current reward.

        Returns:
            Float reward = (captured/total) * base - step_penalty * steps.
        """
        penalty = self.step_penalty * self.current_step
        return compute_reward(
            captured=self.captured_count,
            total=self.n_particles,
            base_reward=self.base_reward,
            step_penalty=penalty,
        )

    # -- Episode termination --

    @property
    def episode_done(self) -> bool:
        """Episode is done when all particles are done OR max_steps reached."""
        if all(p.done for p in self.particles):
            return True
        if self.max_steps > 0 and self.current_step >= self.max_steps:
            return True
        return False

    @property
    def all_captured(self) -> bool:
        """True if every particle reached the well."""
        return self.captured_count == self.n_particles

    # -- Reset --

    def reset(self, positions: Optional[List[List[float]]] = None) -> None:
        """Reset the controller for a new episode.

        Args:
            positions: Optional new positions; if None, keeps current.
        """
        self.current_step = 0
        self._captured_indices.clear()
        for p in self.particles:
            p.captured = False
            p.done = False
            p.steps = 0
        if positions is not None:
            self.set_positions(positions)

    # -- Serialization --

    def to_dict(self) -> dict:
        """Serialize controller state to a plain dict."""
        return {
            "n_particles": self.n_particles,
            "current_step": self.current_step,
            "captured_count": self.captured_count,
            "active_count": self.active_count,
            "episode_done": self.episode_done,
            "reward": self.get_reward(),
            "particles": [
                {
                    "index": p.index,
                    "position": list(p.position),
                    "captured": p.captured,
                    "done": p.done,
                    "steps": p.steps,
                }
                for p in self.particles
            ],
        }

    def __repr__(self) -> str:
        return (
            f"MultiParticleController(n={self.n_particles}, "
            f"step={self.current_step}, "
            f"captured={self.captured_count}, "
            f"done={self.episode_done})"
        )
