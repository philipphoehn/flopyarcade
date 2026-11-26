#!/usr/bin/env python3
"""Q1: Property-based testing with Hypothesis.

Generates random valid configurations and asserts invariants
across the multi-particle module and reward computation.
"""

import pytest
try:
    from hypothesis import given, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

if not HAS_HYPOTHESIS:
    pytest.skip('hypothesis not installed', allow_module_level=True)
from hypothesis import strategies as st

from flopyarcade.multi_particle import (
    spawn_particles,
    compute_reward,
    MultiParticleController,
    make_envtype,
    parse_envtype,
)

# ---------------------------------------------------------------------------
# Valid envtype base strings (single-particle naming)
# ---------------------------------------------------------------------------
VALID_ENVTYPES = [
    '1s-d', '1s-c', '1r-d', '1r-c',
    '2s-d', '2s-c', '2r-d', '2r-c',
    '3s-d', '3s-c', '3r-d', '3r-c',
    '4s-c', '4r-c',
    '5s-c', '5r-c',
    '6s-c', '6r-c',
    '0s-c',
]

envtype_st = st.sampled_from(VALID_ENVTYPES)
n_particles_st = st.integers(min_value=1, max_value=20)
seed_st = st.integers(min_value=0, max_value=99999)
episode_count_st = st.integers(min_value=1, max_value=100)

# Domain parameters (realistic FloPyArcade values)
EXTENT_X = 320.0
EXTENT_Y = 640.0
D_COL = 10.0
Z_TOP = 20.0
BUFFER_Y = 20.0


# ---------------------------------------------------------------------------
# Invariant 1: spawn_particles returns exactly n positions
# ---------------------------------------------------------------------------
@given(n=n_particles_st, seed=seed_st)
@settings(max_examples=200)
def test_spawn_returns_exactly_n_positions(n, seed):
    """spawn_particles(n, ...) always returns exactly n positions."""
    positions = spawn_particles(
        n=n, extent_x=EXTENT_X, extent_y=EXTENT_Y,
        d_col=D_COL, z_top=Z_TOP, buffer_y=BUFFER_Y, seed=seed,
    )
    assert len(positions) == n
    # Each position is a 3-element list
    for pos in positions:
        assert len(pos) == 3


# ---------------------------------------------------------------------------
# Invariant 2: All positions are within domain bounds
# ---------------------------------------------------------------------------
@given(n=n_particles_st, seed=seed_st)
@settings(max_examples=200)
def test_positions_within_domain_bounds(n, seed):
    """All spawned positions lie within the domain extents."""
    positions = spawn_particles(
        n=n, extent_x=EXTENT_X, extent_y=EXTENT_Y,
        d_col=D_COL, z_top=Z_TOP, buffer_y=BUFFER_Y, seed=seed,
    )
    for x, y, z in positions:
        assert 0 <= x <= EXTENT_X, f"x={x} out of [0, {EXTENT_X}]"
        assert BUFFER_Y <= y <= EXTENT_Y - BUFFER_Y, (
            f"y={y} out of [{BUFFER_Y}, {EXTENT_Y - BUFFER_Y}]"
        )
        assert 0 <= z <= Z_TOP, f"z={z} out of [0, {Z_TOP}]"


# ---------------------------------------------------------------------------
# Invariant 3: compute_reward in [-1, 1] when base_reward=1, penalty=0
# ---------------------------------------------------------------------------
@given(
    total=n_particles_st,
    captured_frac=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200)
def test_reward_bounded_unit_interval(total, captured_frac):
    """compute_reward(captured, total) ∈ [-1, 1] for base_reward=1, penalty=0."""
    captured = int(captured_frac * total)
    assume(0 <= captured <= total)
    reward = compute_reward(
        captured=captured, total=total,
        base_reward=1.0, step_penalty=0.0,
    )
    assert -1.0 <= reward <= 1.0, f"reward={reward} out of [-1, 1]"


# ---------------------------------------------------------------------------
# Invariant 4: Controller starts with episode_done=False
# ---------------------------------------------------------------------------
@given(n=n_particles_st)
@settings(max_examples=100)
def test_controller_starts_not_done(n):
    """MultiParticleController with any valid n starts with episode_done=False."""
    ctrl = MultiParticleController(n_particles=n, max_steps=0)
    positions = spawn_particles(
        n=n, extent_x=EXTENT_X, extent_y=EXTENT_Y,
        d_col=D_COL, z_top=Z_TOP, buffer_y=BUFFER_Y, seed=42,
    )
    ctrl.set_positions(positions)
    assert ctrl.episode_done is False
    assert ctrl.captured_count == 0
    assert ctrl.current_step == 0


# ---------------------------------------------------------------------------
# Invariant 5: After marking all captured, episode_done becomes True
# ---------------------------------------------------------------------------
@given(n=n_particles_st)
@settings(max_examples=100)
def test_all_captured_marks_done(n):
    """After marking all particles captured, episode_done is True (no max_steps)."""
    ctrl = MultiParticleController(n_particles=n, max_steps=0)
    positions = spawn_particles(
        n=n, extent_x=EXTENT_X, extent_y=EXTENT_Y,
        d_col=D_COL, z_top=Z_TOP, buffer_y=BUFFER_Y, seed=7,
    )
    ctrl.set_positions(positions)
    assert ctrl.episode_done is False

    # Mark all particles as captured
    ctrl.mark_captured(list(range(n)))
    assert ctrl.episode_done is True
    assert ctrl.captured_count == n
    assert ctrl.all_captured is True


# ---------------------------------------------------------------------------
# Invariant 6: make_envtype + parse_envtype roundtrip
# ---------------------------------------------------------------------------
@given(base=envtype_st, n=n_particles_st)
@settings(max_examples=200)
def test_envtype_roundtrip(base, n):
    """make_envtype(base, n) → parse_envtype(...) → (base, n) roundtrip."""
    constructed = make_envtype(base, n)
    parsed_base, parsed_n = parse_envtype(constructed)
    assert parsed_base == base
    assert parsed_n == n


# ---------------------------------------------------------------------------
# Invariant 7: Reward monotonically non-decreasing in captured count
# ---------------------------------------------------------------------------
@given(total=n_particles_st)
@settings(max_examples=100)
def test_reward_monotonic_in_captured(total):
    """Reward is monotonically non-decreasing as captured increases (fixed total)."""
    prev_reward = -float('inf')
    for captured in range(0, total + 1):
        reward = compute_reward(
            captured=captured, total=total,
            base_reward=1.0, step_penalty=0.0,
        )
        assert reward >= prev_reward, (
            f"Reward decreased: captured={captured} → {reward} < prev {prev_reward}"
        )
        prev_reward = reward


# ---------------------------------------------------------------------------
# Additional: seed determinism
# ---------------------------------------------------------------------------
@given(n=n_particles_st, seed=seed_st)
@settings(max_examples=50)
def test_spawn_deterministic_with_seed(n, seed):
    """Same seed produces identical positions."""
    kw = dict(
        n=n, extent_x=EXTENT_X, extent_y=EXTENT_Y,
        d_col=D_COL, z_top=Z_TOP, buffer_y=BUFFER_Y, seed=seed,
    )
    pos1 = spawn_particles(**kw)
    pos2 = spawn_particles(**kw)
    assert pos1 == pos2


# ---------------------------------------------------------------------------
# Additional: reward at extremes
# ---------------------------------------------------------------------------
@given(total=n_particles_st)
@settings(max_examples=50)
def test_reward_extremes(total):
    """capture=0 → reward 0; capture=total → reward = base_reward (penalty=0)."""
    r_zero = compute_reward(captured=0, total=total, base_reward=1.0, step_penalty=0.0)
    assert r_zero == 0.0

    r_full = compute_reward(captured=total, total=total, base_reward=1.0, step_penalty=0.0)
    assert r_full == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
