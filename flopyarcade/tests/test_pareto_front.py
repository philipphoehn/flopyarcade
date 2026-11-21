"""Tests for flopyarcade.pareto_front (ROADMAP B22)."""

import os
import tempfile

import numpy as np
import pytest

from flopyarcade.pareto_front import (
    ParetoPoint,
    find_pareto_front,
    compute_ci_envelope,
    front_efficiency,
    rank_algorithms,
    format_pareto_report,
    save_front,
    load_front,
)


# ─── find_pareto_front ───────────────────────────────────────────────────────

class TestFindParetoFront:
    def test_basic_three_points_two_nondominated(self):
        """3 points where one is dominated → 2 remain on front."""
        a = ParetoPoint("algoA", "env1", cost=1.0, reward=5.0)
        b = ParetoPoint("algoB", "env1", cost=3.0, reward=7.0)
        c = ParetoPoint("algoC", "env1", cost=2.0, reward=4.0)  # dominated by a

        front = find_pareto_front([a, b, c])
        assert len(front) == 2
        front_costs = {p.cost for p in front}
        assert front_costs == {1.0, 3.0}

    def test_single_point(self):
        p = ParetoPoint("solo", "env", cost=1.0, reward=1.0)
        front = find_pareto_front([p])
        assert len(front) == 1
        assert front[0] is p

    def test_all_dominated_one_remains(self):
        """One best point dominates all others → only 1 on front."""
        best = ParetoPoint("best", "env", cost=0.5, reward=10.0)
        p1 = ParetoPoint("p1", "env", cost=2.0, reward=3.0)
        p2 = ParetoPoint("p2", "env", cost=4.0, reward=2.0)
        p3 = ParetoPoint("p3", "env", cost=1.5, reward=5.0)

        front = find_pareto_front([best, p1, p2, p3])
        assert len(front) == 1
        assert front[0].algorithm == "best"

    def test_empty(self):
        assert find_pareto_front([]) == []

    def test_random_10_points_front_smaller(self):
        """With 10 random points the front should be strictly smaller."""
        rng = np.random.default_rng(123)
        points = [
            ParetoPoint(f"algo{i}", "env", cost=float(rng.uniform(0, 10)),
                        reward=float(rng.uniform(0, 10)))
            for i in range(10)
        ]
        front = find_pareto_front(points)
        assert len(front) < len(points)
        assert len(front) >= 1


# ─── compute_ci_envelope ─────────────────────────────────────────────────────

class TestComputeCIEnvelope:
    def test_structure(self):
        points = [
            ParetoPoint("A", "env", cost=1.0, reward=5.0),
            ParetoPoint("A", "env", cost=2.0, reward=7.0),
            ParetoPoint("B", "env", cost=3.0, reward=9.0),
        ]
        result = compute_ci_envelope(points, n_bootstraps=100, ci_level=0.95)
        assert isinstance(result, list)
        assert len(result) == 3
        for entry in result:
            assert set(entry.keys()) == {"cost", "reward_mean", "ci_low", "ci_high"}
        # sorted by cost
        costs = [e["cost"] for e in result]
        assert costs == sorted(costs)

    def test_ci_contains_mean(self):
        rng = np.random.default_rng(99)
        points = [
            ParetoPoint("A", "env", cost=1.0, reward=float(rng.normal(5, 1))),
            ParetoPoint("A", "env", cost=1.0, reward=float(rng.normal(5, 1))),
            ParetoPoint("A", "env", cost=1.0, reward=float(rng.normal(5, 1))),
            ParetoPoint("A", "env", cost=2.0, reward=float(rng.normal(8, 1))),
            ParetoPoint("A", "env", cost=2.0, reward=float(rng.normal(8, 1))),
        ]
        result = compute_ci_envelope(points, n_bootstraps=500, ci_level=0.95)
        for entry in result:
            assert entry["ci_low"] <= entry["reward_mean"] <= entry["ci_high"], (
                f"CI [{entry['ci_low']}, {entry['ci_high']}] does not contain "
                f"mean {entry['reward_mean']}"
            )

    def test_empty_points(self):
        assert compute_ci_envelope([]) == []


# ─── front_efficiency ────────────────────────────────────────────────────────

class TestFrontEfficiency:
    def test_known_value(self):
        all_pts = [
            ParetoPoint("A", "env", cost=1.0, reward=5.0),
            ParetoPoint("B", "env", cost=2.0, reward=6.0),
            ParetoPoint("C", "env", cost=3.0, reward=3.0),
            ParetoPoint("D", "env", cost=4.0, reward=2.0),
        ]
        front = [all_pts[0], all_pts[1]]
        eff = front_efficiency(front, all_pts)
        assert eff == pytest.approx(0.5)

    def test_all_on_front(self):
        pts = [
            ParetoPoint("A", "env", cost=1.0, reward=5.0),
            ParetoPoint("B", "env", cost=2.0, reward=6.0),
        ]
        assert front_efficiency(pts, pts) == 1.0

    def test_empty_all(self):
        assert front_efficiency([], []) == 1.0


# ─── rank_algorithms ─────────────────────────────────────────────────────────

class TestRankAlgorithms:
    def test_ordering(self):
        fronts = {
            "low":  [ParetoPoint("low", "env", cost=1.0, reward=1.0)],
            "mid":  [ParetoPoint("mid", "env", cost=2.0, reward=5.0)],
            "high": [ParetoPoint("high", "env", cost=3.0, reward=10.0)],
        }
        ranked = rank_algorithms(fronts)
        assert ranked == ["high", "mid", "low"]

    def test_empty_fronts(self):
        fronts = {"empty": []}
        ranked = rank_algorithms(fronts)
        assert ranked == ["empty"]


# ─── format_pareto_report ────────────────────────────────────────────────────

class TestFormatParetoReport:
    def test_contains_expected_text(self):
        pts = [
            ParetoPoint("PPO", "water", cost=1.5, reward=7.2),
            ParetoPoint("SAC", "water", cost=3.1, reward=9.8),
        ]
        all_pts = pts + [ParetoPoint("DQN", "water", cost=5.0, reward=2.0)]
        report = format_pareto_report(pts, all_pts)
        assert "Pareto Front Report" in report
        assert "PPO" in report
        assert "SAC" in report
        assert "Total points:" in report
        assert "Efficiency:" in report
        assert "Front size:" in report

    def test_empty_front(self):
        report = format_pareto_report([], [])
        assert "Pareto Front Report" in report
        assert "empty front" in report


# ─── save / load roundtrip ──────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self):
        pts = [
            ParetoPoint("A", "env1", cost=0.5, reward=3.3),
            ParetoPoint("B", "env2", cost=1.2, reward=7.7),
            ParetoPoint("C", "env3", cost=9.9, reward=1.1),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_front(pts, path)
            loaded = load_front(path)
            assert len(loaded) == 3
            for orig, ld in zip(pts, loaded):
                assert ld.algorithm == orig.algorithm
                assert ld.envtype == orig.envtype
                assert ld.cost == orig.cost
                assert ld.reward == orig.reward
        finally:
            os.unlink(path)
