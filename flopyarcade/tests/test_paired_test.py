"""Tests for flopyarcade.paired_test module (B25)."""

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from flopyarcade.paired_test import (
    PairedResult,
    paired_t_test,
    cohens_d_benchmark,
    format_paired_report,
    batch_paired_test,
)


class TestPairedTTest:
    """Tests for paired_t_test function."""

    def test_identical_samples(self):
        """Identical samples should give t=0, p≈1, not significant."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = paired_t_test(a, b, name="identical")
        assert result.t_stat == 0.0
        assert result.p_value == 1.0
        assert result.mean_diff == 0.0
        assert result.cohens_d == 0.0
        assert result.significant is False
        assert result.n == 10

    def test_very_different_samples(self):
        """Very different samples should give small p-value."""
        a = [10.0, 12.0, 11.0, 13.0, 12.5, 11.5, 10.5, 13.5, 12.0, 11.0]
        b = [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.2, 2.8, 1.9, 2.1]
        result = paired_t_test(a, b, name="different")
        assert result.p_value < 0.001
        assert result.significant is True
        assert result.mean_diff > 5.0
        assert abs(result.t_stat) > 5.0

    def test_length_mismatch_raises(self):
        """Different length samples should raise ValueError."""
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0]
        with pytest.raises(ValueError, match="lengths must be equal"):
            paired_t_test(a, b)

    def test_n_equals_2_minimum(self):
        """n=2 should work (minimum valid sample size)."""
        a = [1.0, 3.0]
        b = [0.5, 2.5]
        result = paired_t_test(a, b, name="n2")
        assert result.n == 2
        assert result.mean_diff == 0.5
        # With n=2, df=1, t should be finite
        assert math.isfinite(result.t_stat)
        assert 0.0 <= result.p_value <= 1.0

    def test_p_value_in_range(self):
        """p_value must always be in [0, 1]."""
        a = [3.0, 5.0, 2.0, 7.0, 4.0, 6.0, 1.0, 8.0, 5.5, 3.5]
        b = [2.0, 4.0, 3.0, 6.0, 3.0, 5.0, 2.0, 7.0, 4.5, 2.5]
        result = paired_t_test(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_ci_contains_mean_diff(self):
        """The 95% CI should contain the mean_diff."""
        a = [3.0, 5.0, 2.0, 7.0, 4.0, 6.0, 1.0, 8.0, 5.5, 3.5]
        b = [2.0, 4.0, 3.0, 6.0, 3.0, 5.0, 2.0, 7.0, 4.5, 2.5]
        result = paired_t_test(a, b)
        assert result.ci_low <= result.mean_diff <= result.ci_high

    def test_significant_flag_correct(self):
        """significant flag should match p < 0.05."""
        # Significant case
        a = [10.0, 12.0, 11.0, 13.0, 12.5, 11.5, 10.5, 13.5, 12.0, 11.0]
        b = [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.2, 2.8, 1.9, 2.1]
        result = paired_t_test(a, b)
        assert result.significant == (result.p_value < 0.05)

        # Non-significant case (identical)
        a2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        b2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        result2 = paired_t_test(a2, b2)
        assert result2.significant == (result2.p_value < 0.05)
        assert result2.significant is False

    def test_empty_list_raises(self):
        """Empty lists should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            paired_t_test([], [])

    def test_single_element_raises(self):
        """n=1 should raise ValueError (need at least 2)."""
        with pytest.raises(ValueError, match="at least 2"):
            paired_t_test([1.0], [0.5])


class TestCohensD:
    """Tests for Cohen's d computation."""

    def test_cohens_d_known_values(self):
        """Verify Cohen's d against manually computed value."""
        a = [5.0, 6.0, 7.0, 8.0, 9.0]
        b = [3.0, 4.0, 5.0, 6.0, 7.0]
        # differences: [2, 2, 2, 2, 2], std=0
        # When std is 0, we return 0.0
        d = cohens_d_benchmark(a, b)
        assert d == 0.0

    def test_cohens_d_with_variability(self):
        """Cohen's d with non-zero std."""
        a = [1.0, 3.0, 2.0, 5.0, 4.0]
        b = [0.5, 1.0, 1.5, 2.0, 3.0]
        # diffs: [0.5, 2.0, 0.5, 3.0, 1.0]
        # mean_diff = 1.4, std_d = sqrt(((0.5-1.4)^2 + (2-1.4)^2 + (0.5-1.4)^2 + (3-1.4)^2 + (1-1.4)^2)/4)
        # = sqrt((0.81+0.36+0.81+2.56+0.16)/4) = sqrt(4.7/4) = sqrt(1.175) ≈ 1.0840
        import numpy as np
        d_arr = np.array(a) - np.array(b)
        expected = float(np.mean(d_arr)) / float(np.std(d_arr, ddof=1))
        d = cohens_d_benchmark(a, b)
        assert abs(d - expected) < 1e-10

    def test_cohens_d_benchmark_standalone(self):
        """cohens_d_benchmark works independently."""
        a = [10.0, 15.0, 20.0, 25.0, 30.0]
        b = [8.0, 13.0, 18.0, 23.0, 28.0]
        d = cohens_d_benchmark(a, b)
        # diffs all 2.0, std=0 -> return 0.0
        assert d == 0.0

    def test_cohens_d_mismatch_raises(self):
        """Length mismatch should raise."""
        with pytest.raises(ValueError):
            cohens_d_benchmark([1.0, 2.0], [1.0])


class TestFormatReport:
    """Tests for format_paired_report."""

    def test_report_contains_tstat_and_pvalue(self):
        """Report should contain t-statistic and p-value text."""
        a = [3.0, 5.0, 2.0, 7.0, 4.0, 6.0, 1.0, 8.0, 5.5, 3.5]
        b = [2.0, 4.0, 3.0, 6.0, 3.0, 5.0, 2.0, 7.0, 4.5, 2.5]
        result = paired_t_test(a, b, name="report_test")
        report = format_paired_report(result)
        assert "t(" in report
        assert "p =" in report
        assert "Cohen" in report
        assert "95% CI" in report
        assert "report_test" in report

    def test_report_significant_marker(self):
        """Report should indicate significance status."""
        a = [10.0, 12.0, 11.0, 13.0, 12.5, 11.5, 10.5, 13.5, 12.0, 11.0]
        b = [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.2, 2.8, 1.9, 2.1]
        result = paired_t_test(a, b, name="sig_test")
        report = format_paired_report(result)
        assert "SIGNIFICANT" in report


class TestBatchPairedTest:
    """Tests for batch_paired_test."""

    def test_batch_two_envs(self):
        """Batch test with 2 common environments."""
        results_a = {
            "env_alpha": [1.0, 2.0, 3.0, 4.0, 5.0],
            "env_beta": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        results_b = {
            "env_alpha": [0.5, 1.5, 2.5, 3.5, 4.5],
            "env_beta": [9.0, 19.0, 29.0, 39.0, 49.0],
        }
        out = batch_paired_test(results_a, results_b)
        assert set(out.keys()) == {"env_alpha", "env_beta"}
        assert out["env_alpha"].name == "env_alpha"
        assert out["env_beta"].name == "env_beta"
        assert out["env_alpha"].mean_diff == 0.5
        assert out["env_beta"].mean_diff == 1.0

    def test_batch_skips_non_common_keys(self):
        """Keys not in both dicts should be skipped."""
        results_a = {
            "shared": [1.0, 2.0, 3.0],
            "only_a": [4.0, 5.0, 6.0],
        }
        results_b = {
            "shared": [0.5, 1.5, 2.5],
            "only_b": [7.0, 8.0, 9.0],
        }
        out = batch_paired_test(results_a, results_b)
        assert set(out.keys()) == {"shared"}
        assert "only_a" not in out
        assert "only_b" not in out

    def test_batch_empty_intersection(self):
        """No common keys should return empty dict."""
        results_a = {"x": [1.0, 2.0]}
        results_b = {"y": [3.0, 4.0]}
        out = batch_paired_test(results_a, results_b)
        assert out == {}


class TestPairedResultDataclass:
    """Verify PairedResult dataclass structure."""

    def test_fields(self):
        """All expected fields exist with correct types."""
        r = PairedResult(
            name="test", n=10, mean_diff=1.5, t_stat=3.2,
            p_value=0.008, cohens_d=1.0, significant=True,
            ci_low=0.8, ci_high=2.2,
        )
        assert isinstance(r.name, str)
        assert isinstance(r.n, int)
        assert isinstance(r.mean_diff, float)
        assert isinstance(r.t_stat, float)
        assert isinstance(r.p_value, float)
        assert isinstance(r.cohens_d, float)
        assert isinstance(r.significant, bool)
        assert isinstance(r.ci_low, float)
        assert isinstance(r.ci_high, float)
