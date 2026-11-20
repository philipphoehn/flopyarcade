"""Tests for flopyarcade.bootstrap_ci (ROADMAP B18)."""

import numpy as np
import pytest

from flopyarcade.bootstrap_ci import bootstrap_ci, bootstrap_test, seed_sensitivity_table


# ─── bootstrap_ci ─────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_returns_correct_keys(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(scores, n_resamples=200, seed=42)
        assert set(result.keys()) == {"mean", "ci_low", "ci_high", "n", "ci_level"}

    def test_ci_contains_sample_mean(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(loc=50, scale=10, size=100)
        result = bootstrap_ci(scores, n_resamples=500, seed=42)
        assert result["ci_low"] <= result["mean"] <= result["ci_high"]

    def test_ci_width_decreases_with_more_samples(self):
        rng = np.random.default_rng(1)
        scores_small = rng.normal(loc=10, scale=2, size=10)
        scores_large = rng.normal(loc=10, scale=2, size=500)

        res_small = bootstrap_ci(scores_small, n_resamples=500, seed=42)
        res_large = bootstrap_ci(scores_large, n_resamples=500, seed=42)

        width_small = res_small["ci_high"] - res_small["ci_low"]
        width_large = res_large["ci_high"] - res_large["ci_low"]
        assert width_large < width_small

    def test_n1_degenerate_case(self):
        result = bootstrap_ci([42.0], n_resamples=100, seed=42)
        assert result["mean"] == 42.0
        assert result["ci_low"] == 42.0
        assert result["ci_high"] == 42.0
        assert result["n"] == 1

    def test_ci_symmetric_around_mean(self):
        rng = np.random.default_rng(2)
        scores = rng.normal(loc=0, scale=1, size=200)  # symmetric distribution
        result = bootstrap_ci(scores, n_resamples=2000, seed=42)
        # For a symmetric distribution the CI should be approximately symmetric
        left_span = result["mean"] - result["ci_low"]
        right_span = result["ci_high"] - result["mean"]
        ratio = max(left_span, right_span) / max(min(left_span, right_span), 1e-10)
        assert ratio < 1.15  # within 15 % of symmetric

    def test_reproducibility_same_seed(self):
        scores = [1.0, 2.5, 3.7, 4.1, 5.3, 6.0]
        r1 = bootstrap_ci(scores, n_resamples=500, seed=123)
        r2 = bootstrap_ci(scores, n_resamples=500, seed=123)
        assert r1 == r2

    def test_different_seeds_differ(self):
        scores = [1.0, 2.5, 3.7, 4.1, 5.3, 6.0]
        r1 = bootstrap_ci(scores, n_resamples=500, seed=1)
        r2 = bootstrap_ci(scores, n_resamples=500, seed=2)
        # Extremely unlikely to be identical
        assert r1["ci_low"] != r2["ci_low"] or r1["ci_high"] != r2["ci_high"]


# ─── bootstrap_test ───────────────────────────────────────────────────────────

class TestBootstrapTest:
    def test_identical_samples_p_near_1(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_test(a, a, n_resamples=500, seed=42)
        assert result["mean_diff"] == 0.0
        assert result["p_value"] > 0.9  # essentially 1
        assert result["significant"] is False

    def test_very_different_samples_p_near_0(self):
        a = [100.0, 101.0, 102.0, 103.0, 104.0]
        b = [0.0, 0.1, 0.2, 0.3, 0.4]
        result = bootstrap_test(a, b, n_resamples=500, seed=42)
        assert result["mean_diff"] > 50
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_overlapping_samples_0_lt_p_lt_1(self):
        rng = np.random.default_rng(7)
        a = rng.normal(loc=5, scale=2, size=50)
        b = rng.normal(loc=5.5, scale=2, size=50)
        result = bootstrap_test(a, b, n_resamples=500, seed=42)
        assert 0.0 < result["p_value"] < 1.0
        assert result["mean_diff"] != 0.0

    def test_returns_correct_keys(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = bootstrap_test(a, b, n_resamples=200, seed=42)
        assert set(result.keys()) == {"mean_diff", "ci_low", "ci_high", "p_value", "significant"}

    def test_reproducibility_same_seed(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 3.0, 4.0, 5.0]
        r1 = bootstrap_test(a, b, n_resamples=300, seed=99)
        r2 = bootstrap_test(a, b, n_resamples=300, seed=99)
        assert r1 == r2


# ─── seed_sensitivity_table ───────────────────────────────────────────────────

class TestSeedSensitivityTable:
    def test_structure(self):
        data = {
            "env_a": [10.0, 20.0, 30.0, 40.0],
            "env_b": [5.0, 15.0, 25.0],
        }
        table = seed_sensitivity_table(data, n_seeds=4)
        assert set(table.keys()) == {"env_a", "env_b"}
        for envtype in ("env_a", "env_b"):
            assert set(table[envtype].keys()) == {"mean", "std", "iqr", "cv", "n"}

    def test_single_envtype(self):
        data = {"only_env": [1.0, 2.0, 3.0, 4.0, 5.0]}
        table = seed_sensitivity_table(data)
        assert "only_env" in table
        assert table["only_env"]["mean"] == 3.0
        assert table["only_env"]["n"] == 5
        assert table["only_env"]["std"] > 0
        assert table["only_env"]["iqr"] > 0
        assert table["only_env"]["cv"] > 0

    def test_known_statistics(self):
        data = {"d": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]}
        table = seed_sensitivity_table(data)
        t = table["d"]
        assert t["n"] == 8
        assert t["mean"] == pytest.approx(5.0)
        # numpy linear interpolation: Q1=4.0, Q3=5.5, IQR=1.5
        assert t["iqr"] == pytest.approx(1.5)

    def test_empty_scores(self):
        data = {"empty": []}
        table = seed_sensitivity_table(data)
        assert table["empty"]["n"] == 0
        assert table["empty"]["mean"] == 0.0


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            bootstrap_ci([])

    def test_empty_sample_in_test_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            bootstrap_test([], [1.0, 2.0])

    def test_ci_level_custom(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        r90 = bootstrap_ci(scores, n_resamples=500, ci_level=0.90, seed=42)
        r99 = bootstrap_ci(scores, n_resamples=500, ci_level=0.99, seed=42)
        width90 = r90["ci_high"] - r90["ci_low"]
        width99 = r99["ci_high"] - r99["ci_low"]
        assert width99 > width90
        assert r99["ci_level"] == 0.99
