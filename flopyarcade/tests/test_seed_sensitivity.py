"""Tests for B19: seed-sensitivity table module."""

import math

from flopyarcade.seed_sensitivity import (
    SeedSensitivityCell,
    compute_seed_sensitivity,
    cross_algorithm_comparison,
    find_unstable_cells,
    format_sensitivity_report,
)


class TestComputeSeedSensitivity:
    def test_basic_single_env_algo(self):
        """1 env, 1 algo, 5 seeds → one cell with correct stats."""
        results = {"envA": {"dqn": [10.0, 20.0, 30.0, 40.0, 50.0]}}
        cells = compute_seed_sensitivity(results)
        assert len(cells) == 1
        c = cells[0]
        assert c.envtype == "envA"
        assert c.algorithm == "dqn"
        assert c.n_seeds == 5
        assert c.mean == 30.0
        assert math.isclose(c.std, 14.142135623730951, rel_tol=1e-9)
        assert c.min == 10.0
        assert c.max == 50.0
        assert c.scores == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_multiple_envs_algorithms(self):
        """2 envs × 2 algos → 4 cells."""
        results = {
            "envA": {"dqn": [1.0, 2.0, 3.0], "ppo": [4.0, 5.0, 6.0]},
            "envB": {"dqn": [7.0, 8.0, 9.0], "sac": [10.0, 11.0, 12.0]},
        }
        cells = compute_seed_sensitivity(results)
        assert len(cells) == 4
        envs = {c.envtype for c in cells}
        algos = {c.algorithm for c in cells}
        assert envs == {"envA", "envB"}
        assert algos == {"dqn", "ppo", "sac"}

    def test_iqr_correctness(self):
        """IQR = Q3 - Q1 for known values."""
        # scores 1..10 → Q1=3.25, Q3=7.75, IQR=4.5
        results = {"envX": {"algo": [float(i) for i in range(1, 11)]}}
        cells = compute_seed_sensitivity(results)
        c = cells[0]
        assert c.iqr == 4.5

    def test_cv_calculation(self):
        """CV = std / mean."""
        scores = [10.0, 20.0, 30.0]
        results = {"env": {"a": scores}}
        cells = compute_seed_sensitivity(results)
        c = cells[0]
        expected_cv = c.std / c.mean
        assert math.isclose(c.cv, expected_cv, rel_tol=1e-9)

    def test_cv_zero_when_all_identical(self):
        """All scores same → std=0 → CV=0."""
        results = {"env": {"a": [5.0, 5.0, 5.0]}}
        cells = compute_seed_sensitivity(results)
        assert cells[0].cv == 0.0
        assert cells[0].std == 0.0

    def test_cv_zero_when_mean_is_zero(self):
        """Scores sum to 0 → mean=0 → CV=0 (division guard)."""
        results = {"env": {"a": [-5.0, 5.0]}}
        cells = compute_seed_sensitivity(results)
        assert cells[0].cv == 0.0


class TestFindUnstableCells:
    def test_high_variance_flagged(self):
        """CV > 0.5 → flagged."""
        # scores [1, 100] → mean=50.5, std≈49.5, cv≈0.98
        results = {"env": {"a": [1.0, 100.0]}}
        cells = compute_seed_sensitivity(results)
        unstable = find_unstable_cells(cells)
        assert len(unstable) == 1
        assert unstable[0].algorithm == "a"

    def test_low_variance_none_flagged(self):
        """CV < 0.5 → not flagged."""
        results = {"env": {"a": [10.0, 11.0, 12.0]}}
        cells = compute_seed_sensitivity(results)
        unstable = find_unstable_cells(cells)
        assert len(unstable) == 0


class TestFormatSensitivityReport:
    def test_contains_headers(self):
        """Report has expected column headers."""
        results = {"envA": {"dqn": [1.0, 2.0, 3.0]}}
        cells = compute_seed_sensitivity(results)
        report = format_sensitivity_report(cells)
        assert "EnvType" in report
        assert "Algorithm" in report
        assert "Mean" in report
        assert "Std" in report
        assert "IQR" in report
        assert "CV" in report
        assert "Flagged" in report


class TestCrossAlgorithmComparison:
    def test_structure(self):
        """Returns {envtype: {algorithm: cv}}."""
        results = {
            "envA": {"dqn": [1.0, 2.0, 3.0], "ppo": [10.0, 20.0, 30.0]},
            "envB": {"sac": [5.0, 6.0, 7.0]},
        }
        cells = compute_seed_sensitivity(results)
        comp = cross_algorithm_comparison(cells)
        assert set(comp.keys()) == {"envA", "envB"}
        assert set(comp["envA"].keys()) == {"dqn", "ppo"}
        assert set(comp["envB"].keys()) == {"sac"}
        for env in comp:
            for algo in comp[env]:
                assert isinstance(comp[env][algo], float)


class TestEdgeCases:
    def test_empty_input(self):
        """Empty results → empty list."""
        assert compute_seed_sensitivity({}) == []

    def test_single_seed(self):
        """One seed → std=0, IQR=0, CV=0."""
        results = {"env": {"a": [42.0]}}
        cells = compute_seed_sensitivity(results)
        c = cells[0]
        assert c.n_seeds == 1
        assert c.std == 0.0
        assert c.iqr == 0.0
        assert c.cv == 0.0
        assert c.min == 42.0
        assert c.max == 42.0

    def test_scores_preserved(self):
        """Original scores list is preserved in the cell."""
        scores = [3.14, 2.71, 1.41]
        results = {"env": {"a": scores}}
        cells = compute_seed_sensitivity(results)
        assert cells[0].scores == scores
