"""B13: Tests for keep-winner distance cap penalty."""

import pytest

from flopyarcade.keep_penalty import KeepPenaltyTracker, compute_distance_cap_reward


class TestKeepPenaltyTracker:
    def test_initial_count_zero(self):
        t = KeepPenaltyTracker()
        assert t.count == 0
        assert t.penalty == 0.0

    def test_single_keep(self):
        t = KeepPenaltyTracker(base=1.0, max_penalty=10.0)
        p = t.on_action('keep')
        assert p == 1.0
        assert t.count == 1

    def test_consecutive_keeps_escalate(self):
        t = KeepPenaltyTracker(base=1.0, max_penalty=100.0)
        assert t.on_action('keep') == 1.0
        assert t.on_action('keep') == 2.0
        assert t.on_action('keep') == 3.0
        assert t.count == 3

    def test_non_keep_resets(self):
        t = KeepPenaltyTracker(base=1.0)
        t.on_action('keep')
        t.on_action('keep')
        assert t.on_action('up') == 0.0
        assert t.count == 0

    def test_keepq_also_counts(self):
        t = KeepPenaltyTracker(base=1.0)
        assert t.on_action('keepQ') == 1.0
        assert t.on_action('keep') == 2.0

    def test_max_penalty_cap(self):
        t = KeepPenaltyTracker(base=1.0, max_penalty=5.0)
        for _ in range(10):
            p = t.on_action('keep')
        assert p == 5.0
        # Further keeps stay capped
        assert t.on_action('keep') == 5.0

    def test_zero_base(self):
        t = KeepPenaltyTracker(base=0.0)
        t.on_action('keep')
        assert t.penalty == 0.0

    def test_reset_method(self):
        t = KeepPenaltyTracker(base=1.0)
        t.on_action('keep')
        t.on_action('keep')
        t.reset()
        assert t.count == 0
        assert t.penalty == 0.0

    def test_custom_keep_actions(self):
        t = KeepPenaltyTracker(base=1.0, keep_actions={'stay', 'wait'})
        assert t.on_action('stay') == 1.0
        assert t.on_action('wait') == 2.0
        assert t.on_action('keep') == 0.0  # not in custom set
        assert t.count == 0

    def test_negative_base_clamped(self):
        t = KeepPenaltyTracker(base=-5.0)
        assert t.base == 0.0

    def test_negative_max_clamped(self):
        t = KeepPenaltyTracker(base=1.0, max_penalty=-1.0)
        assert t.max_penalty == 0.0

    def test_repr(self):
        t = KeepPenaltyTracker(base=0.5, max_penalty=10.0)
        t.on_action('keep')
        r = repr(t)
        assert 'base=0.5' in r
        assert 'count=1' in r

    def test_penalty_property(self):
        t = KeepPenaltyTracker(base=2.0, max_penalty=10.0)
        t.on_action('keep')
        t.on_action('keep')
        assert t.penalty == 4.0


class TestComputeDistanceCapReward:
    def test_forward_perfect_path(self):
        """Shortest == actual gives full reward."""
        r = compute_distance_cap_reward(
            length_shortest=50.0, length_actual=50.0,
            distance_max=100.0, reward_max=1000.0, deviation_factor=10.0
        )
        # dist_frac = 0.5, path_ratio = 1.0, reward = 500 * 1.0 = 500
        assert r == pytest.approx(500.0)

    def test_forward_deviant_path(self):
        """Actual > shortest reduces reward."""
        r = compute_distance_cap_reward(
            length_shortest=50.0, length_actual=100.0,
            distance_max=100.0, reward_max=1000.0, deviation_factor=10.0
        )
        # path_ratio = 0.5, reward = 500 * 0.5^10 = very small
        assert r < 1.0
        assert r > 0.0

    def test_no_movement_zero(self):
        r = compute_distance_cap_reward(
            length_shortest=0.0, length_actual=0.0,
            distance_max=100.0, reward_max=1000.0, deviation_factor=10.0
        )
        assert r == 0.0

    def test_backward_movement_penalty(self):
        r = compute_distance_cap_reward(
            length_shortest=-20.0, length_actual=30.0,
            distance_max=100.0, reward_max=1000.0, deviation_factor=10.0
        )
        # dist_frac = -0.2, rewardMaxSegment = -200, reward = -5 * 200 = -1000
        assert r == pytest.approx(-1000.0)

    def test_keep_penalty_applied(self):
        r = compute_distance_cap_reward(
            length_shortest=0.0, length_actual=0.0,
            distance_max=100.0, reward_max=1000.0, deviation_factor=10.0,
            keep_count=5, keep_base=1.0, keep_max=10.0
        )
        # base reward 0.0, penalty = min(5, 10) = 5
        assert r == pytest.approx(-5.0)

    def test_keep_penalty_capped(self):
        r = compute_distance_cap_reward(
            length_shortest=0.0, length_actual=0.0,
            distance_max=100.0, reward_max=1000.0, deviation_factor=10.0,
            keep_count=100, keep_base=1.0, keep_max=5.0
        )
        # penalty capped at 5
        assert r == pytest.approx(-5.0)

    def test_keep_penalty_with_forward_reward(self):
        r = compute_distance_cap_reward(
            length_shortest=50.0, length_actual=50.0,
            distance_max=100.0, reward_max=1000.0, deviation_factor=10.0,
            keep_count=3, keep_base=1.0, keep_max=10.0
        )
        # base = 500, penalty = 3
        assert r == pytest.approx(497.0)

    def test_zero_actual_length_forward(self):
        """length_actual=0 with positive shortest avoids division by zero."""
        r = compute_distance_cap_reward(
            length_shortest=50.0, length_actual=0.0,
            distance_max=100.0, reward_max=1000.0, deviation_factor=10.0
        )
        # path_ratio defaults to 1.0
        assert r == pytest.approx(500.0)

    def test_no_keep_count(self):
        """keep_count=0 means no penalty."""
        r = compute_distance_cap_reward(
            length_shortest=50.0, length_actual=50.0,
            distance_max=100.0, reward_max=1000.0, deviation_factor=10.0,
            keep_count=0
        )
        assert r == pytest.approx(500.0)
