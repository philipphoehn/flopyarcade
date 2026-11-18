"""A8: Verify reward scaling 0→1 across env types."""

import math
import pytest

from flopyarcade.reward import RewardMixin


class _RewardTestBase(RewardMixin):
    """Minimal concrete class to test RewardMixin methods."""
    pass


def test_normalize_reward_zero_gives_half():
    """normalize_reward_to_unit(0) == 0.5 (sigmoid midpoint)."""
    result = RewardMixin.normalize_reward_to_unit(0.0)
    assert result == pytest.approx(0.5, abs=1e-9)


def test_normalize_reward_positive_in_range():
    """Positive rewards map to (0.5, 1]."""
    for val in [0.1, 1.0, 10.0, 100.0]:
        result = RewardMixin.normalize_reward_to_unit(val)
        assert 0.5 < result <= 1.0, f"reward={val} -> {result} not in (0.5, 1]"


def test_normalize_reward_negative_in_range():
    """Negative rewards map to [0, 0.5)."""
    for val in [-0.1, -1.0, -10.0, -100.0]:
        result = RewardMixin.normalize_reward_to_unit(val)
        assert 0.0 <= result < 0.5, f"reward={val} -> {result} not in [0, 0.5)"


def test_normalize_reward_extremes():
    """Very large positive → ~1, very large negative → ~0."""
    assert RewardMixin.normalize_reward_to_unit(1000.0) > 0.999
    assert RewardMixin.normalize_reward_to_unit(-1000.0) < 0.001


def test_normalize_reward_scale_param():
    """Higher scale makes the sigmoid steeper (faster saturation)."""
    val = 1.0
    low_scale = RewardMixin.normalize_reward_to_unit(val, scale=0.1)
    high_scale = RewardMixin.normalize_reward_to_unit(val, scale=10.0)
    assert high_scale > low_scale


def test_normalize_reward_symmetry():
    """f(x) + f(-x) == 1 for sigmoid."""
    for val in [0.5, 1.0, 3.0, 7.0]:
        pair_sum = (RewardMixin.normalize_reward_to_unit(val) +
                    RewardMixin.normalize_reward_to_unit(-val))
        assert pair_sum == pytest.approx(1.0, abs=1e-9)


def test_get_reward_normalized_uses_instance_reward():
    """get_reward_normalized reads self.reward and applies sigmoid."""
    obj = _RewardTestBase()
    obj.reward = 2.0
    expected = 1.0 / (1.0 + math.exp(-1.0 * 2.0))
    assert obj.get_reward_normalized() == pytest.approx(expected, abs=1e-9)


def test_get_reward_normalized_zero():
    """When reward is 0, normalized is 0.5."""
    obj = _RewardTestBase()
    obj.reward = 0.0
    assert obj.get_reward_normalized() == pytest.approx(0.5, abs=1e-9)


def test_normalize_is_monotonic():
    """Larger raw reward → larger normalized value."""
    values = [-5.0, -1.0, 0.0, 1.0, 5.0]
    normalized = [RewardMixin.normalize_reward_to_unit(v) for v in values]
    for i in range(len(normalized) - 1):
        assert normalized[i] < normalized[i + 1]
