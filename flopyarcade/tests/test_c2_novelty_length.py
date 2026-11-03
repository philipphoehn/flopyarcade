#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Tests for C2: novelty score is length-invariant (normalized by episode count)."""

import pytest
from numpy import array
from flopyarcade.agent import FloPyAgent


class TestNoveltyLengthInvariance:
    """Verify calculateNoveltyPerPair normalizes by episode count."""

    def _make_agent(self, action_type='discrete'):
        agent = FloPyAgent.__new__(FloPyAgent)
        agent.actionType = action_type
        return agent

    def test_discrete_same_score_different_lengths(self):
        agent = self._make_agent('discrete')
        # 5 episodes vs 10 episodes of same per-episode novelty
        a5 = [[0], [1]] * 5
        b5 = [[1], [0]] * 5
        a10 = [[0], [1]] * 10
        b10 = [[1], [0]] * 10

        n5 = agent.calculateNoveltyPerPair([a5, b5])
        n10 = agent.calculateNoveltyPerPair([a10, b10])
        assert n5 == pytest.approx(n10, rel=1e-6)

    def test_discrete_known_value(self):
        agent = self._make_agent('discrete')
        # All different: novelty per pair = 1.0, normalized by 1 episode = 1.0
        a = [[0], [1]]
        b = [[1], [0]]
        n = agent.calculateNoveltyPerPair([a, b])
        assert n == pytest.approx(1.0)

    def test_discrete_identical_zero(self):
        agent = self._make_agent('discrete')
        a = [[0], [1], [2]]
        n5 = agent.calculateNoveltyPerPair([[a] * 5, [a] * 5])
        n20 = agent.calculateNoveltyPerPair([[a] * 20, [a] * 20])
        assert n5 == pytest.approx(0.0)
        assert n20 == pytest.approx(0.0)

    def test_partial_novelty_length_invariant(self):
        agent = self._make_agent('discrete')
        sa = [0, 1, 2]
        sb = [0, 1, 0]  # 1 diff out of 3 = 1/3
        n10 = agent.calculateNoveltyPerPair([[sa] * 10, [sb] * 10])
        n30 = agent.calculateNoveltyPerPair([[sa] * 30, [sb] * 30])
        assert n10 == pytest.approx(n30, rel=1e-6)
        assert n10 == pytest.approx(1.0 / 3.0)

    def test_empty_actions_returns_zero(self):
        agent = self._make_agent('discrete')
        n = agent.calculateNoveltyPerPair([[], []])
        assert n == 0.0
