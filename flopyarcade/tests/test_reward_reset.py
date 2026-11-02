#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Regression tests for C6: reward reset on fail."""

import pytest
from unittest.mock import MagicMock, patch
from flopyarcade.reward import RewardMixin


class TestRewardReset:
    """Verify update_reward_on_fail resets rewardCurrent correctly."""

    def _make_env(self, envtype='3s-d', reward_current=5.0, cost_q_fail=-10.0):
        env = MagicMock(spec=RewardMixin)
        env.ENVTYPE = envtype
        env.rewardCurrent = reward_current
        env.reward = 0.0
        env.costQFail = cost_q_fail
        return env

    def test_non_cost_mode_positive_reward(self):
        env = self._make_env('3s-d', reward_current=5.0)
        RewardMixin.update_reward_on_fail(env)
        assert env.reward == -5.0
        assert env.rewardCurrent == 0.0

    def test_non_cost_mode_negative_reward_edge_case(self):
        env = self._make_env('3s-d', reward_current=-3.0)
        RewardMixin.update_reward_on_fail(env)
        assert env.reward == 0.0
        assert env.rewardCurrent == 0.0

    def test_cost_mode_resets_reward_current(self):
        env = self._make_env('5s-c-cost', reward_current=2.0, cost_q_fail=-10.0)
        RewardMixin.update_reward_on_fail(env)
        assert env.reward == -10.0 + 2.0
        assert env.rewardCurrent == 0.0

    def test_non_cost_mode_zero_reward(self):
        env = self._make_env('3r-d', reward_current=0.0)
        RewardMixin.update_reward_on_fail(env)
        assert env.reward == -0.0
        assert env.rewardCurrent == 0.0

    def test_all_envtype3_variants(self):
        for et in ['3s-d', '3r-d', '3s-c', '3r-c']:
            env = self._make_env(et, reward_current=7.5)
            RewardMixin.update_reward_on_fail(env)
            assert env.reward == -7.5, f"{et}: expected -7.5"
            assert env.rewardCurrent == 0.0, f"{et}: rewardCurrent not reset"
