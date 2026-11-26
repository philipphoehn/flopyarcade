"""Tests for C11: averaging_test module."""

import json
import math
import os
import tempfile

import pytest

from flopyarcade.averaging_test import (
    AveragingValidator,
    RunResult,
    assert_consistent,
    format_averaging_report,
    simulate_identical_runs,
    simulate_noisy_runs,
)


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

class TestRunResult:
    def test_creation(self):
        r = RunResult(run_id=0, seed=42, score=0.8, n_episodes=100)
        assert r.run_id == 0
        assert r.seed == 42
        assert r.score == 0.8
        assert r.n_episodes == 100
        assert r.final_epsilon is None
        assert r.model_hash is None

    def test_creation_with_optional(self):
        r = RunResult(run_id=1, seed=7, score=0.5, n_episodes=50,
                      final_epsilon=0.1, model_hash='abc')
        assert r.final_epsilon == 0.1
        assert r.model_hash == 'abc'


# ---------------------------------------------------------------------------
# AveragingValidator basics
# ---------------------------------------------------------------------------

class TestAveragingValidator:
    def test_init_defaults(self):
        v = AveragingValidator()
        assert v.model_hash == 'default'
        assert v.tolerance == 0.0
        assert v.n_runs() == 0
        assert v.runs() == []

    def test_init_custom(self):
        v = AveragingValidator(model_hash='xyz', tolerance=0.05)
        assert v.model_hash == 'xyz'
        assert v.tolerance == 0.05

    def test_add_run_appends(self):
        v = AveragingValidator()
        v.add_run(0, 42, 0.7, 100)
        v.add_run(1, 42, 0.8, 100)
        assert v.n_runs() == 2
        assert v.runs()[0].score == 0.7
        assert v.runs()[1].score == 0.8

    def test_mean_score_known(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.0, 10)
        v.add_run(1, 1, 1.0, 10)
        v.add_run(2, 1, 2.0, 10)
        assert v.mean_score() == pytest.approx(1.0)

    def test_std_score_zero_identical(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.75, 10)
        v.add_run(1, 1, 0.75, 10)
        v.add_run(2, 1, 0.75, 10)
        assert v.std_score() == pytest.approx(0.0)

    def test_std_score_nonzero_varied(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.0, 10)
        v.add_run(1, 1, 2.0, 10)
        # population std of [0, 2] = 1.0
        assert v.std_score() == pytest.approx(1.0)

    def test_cv_correct(self):
        v = AveragingValidator()
        v.add_run(0, 1, 10.0, 10)
        v.add_run(1, 1, 20.0, 10)
        # mean=15, std=5, cv=5/15
        assert v.cv() == pytest.approx(5.0 / 15.0)

    def test_deterministic_true_identical(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.7, 10)
        v.add_run(1, 1, 0.7, 10)
        v.add_run(2, 1, 0.7, 10)
        assert v.deterministic() is True

    def test_deterministic_false_varied(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.5, 10)
        v.add_run(1, 1, 0.9, 10)
        assert v.deterministic() is False

    def test_max_deviation_correct(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.0, 10)
        v.add_run(1, 1, 4.0, 10)
        # mean = 2.0, max_dev = max(|0-2|, |4-2|) = 2.0
        assert v.max_deviation() == pytest.approx(2.0)

    def test_consistency_check_passed_identical(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.75, 10)
        v.add_run(1, 1, 0.75, 10)
        check = v.consistency_check(threshold=0.01)
        assert check['passed'] is True
        assert check['deterministic'] is True
        assert check['std'] == pytest.approx(0.0)

    def test_consistency_check_failed_noisy(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.50, 10)
        v.add_run(1, 1, 0.95, 10)
        # mean=0.725, max_dev=0.225, threshold*mean=0.00725 -> fail
        check = v.consistency_check(threshold=0.01)
        assert check['passed'] is False

    def test_pairwise_diffs_count(self):
        v = AveragingValidator()
        for i in range(5):
            v.add_run(i, 1, 0.1 * i, 10)
        diffs = v.pairwise_diffs()
        assert len(diffs) == 5 * 4 // 2  # 10

    def test_pairwise_diffs_values(self):
        v = AveragingValidator()
        v.add_run(0, 1, 1.0, 10)
        v.add_run(1, 1, 3.0, 10)
        v.add_run(2, 1, 6.0, 10)
        diffs = v.pairwise_diffs()
        assert diffs[0] == (0, 1, 2.0)
        assert diffs[1] == (0, 2, 5.0)
        assert diffs[2] == (1, 2, 3.0)

    def test_same_seed_scores_filters(self):
        v = AveragingValidator()
        v.add_run(0, 42, 0.7, 10)
        v.add_run(1, 42, 0.8, 10)
        v.add_run(2, 99, 0.5, 10)
        assert v.same_seed_scores(42) == [0.7, 0.8]
        assert v.same_seed_scores(99) == [0.5]
        assert v.same_seed_scores(0) == []

    def test_different_seed_scores_groups(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.7, 10)
        v.add_run(1, 2, 0.8, 10)
        v.add_run(2, 1, 0.75, 10)
        result = v.different_seed_scores()
        assert set(result.keys()) == {1, 2}
        # last run per seed: seed 1 -> 0.75, seed 2 -> 0.8
        assert result[1] == 0.75
        assert result[2] == 0.8

    def test_variance_by_seed_structure(self):
        v = AveragingValidator()
        v.add_run(0, 1, 0.0, 10)
        v.add_run(1, 1, 2.0, 10)
        v.add_run(2, 2, 5.0, 10)
        result = v.variance_by_seed()
        assert set(result.keys()) == {1, 2}
        # seed 1: [0, 2] -> std = 1.0
        assert result[1] == pytest.approx(1.0)
        # seed 2: single run -> 0.0
        assert result[2] == 0.0

    def test_save_load_roundtrip(self):
        v = AveragingValidator(model_hash='test123', tolerance=0.03)
        v.add_run(0, 42, 0.7, 100, final_epsilon=0.05)
        v.add_run(1, 42, 0.8, 200, final_epsilon=0.01)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            v.save_json(path)
            v2 = AveragingValidator.load_json(path)
            assert v2.model_hash == 'test123'
            assert v2.tolerance == 0.03
            assert v2.n_runs() == 2
            assert v2.runs()[0].score == 0.7
            assert v2.runs()[1].final_epsilon == 0.01
            assert v2.mean_score() == pytest.approx(v.mean_score())
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_simulate_identical_runs_same_score(self):
        results = simulate_identical_runs(n_runs=5, seed=42, base_score=0.75)
        assert len(results) == 5
        scores = [r.score for r in results]
        assert all(s == 0.75 for s in scores)

    def test_simulate_noisy_runs_has_variation(self):
        results = simulate_noisy_runs(n_runs=20, seed=123, base_score=0.75, noise_std=0.05)
        scores = [r.score for r in results]
        # with noise_std=0.05 and 20 samples, there should be variation
        assert max(scores) - min(scores) > 0.0


# ---------------------------------------------------------------------------
# assert_consistent
# ---------------------------------------------------------------------------

class TestAssertConsistent:
    def test_passes_for_identical(self):
        results = simulate_identical_runs(n_runs=5, seed=42, base_score=0.8)
        check = assert_consistent(results, tolerance=0.01)
        assert check['passed'] is True
        assert check['deterministic'] is True

    def test_raises_for_noisy(self):
        results = simulate_noisy_runs(n_runs=50, seed=99, base_score=0.75, noise_std=0.2)
        with pytest.raises(AssertionError, match="Inconsistency"):
            assert_consistent(results, tolerance=0.01)


# ---------------------------------------------------------------------------
# format_averaging_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_contains_key_fields(self):
        v = AveragingValidator(model_hash='report_test')
        v.add_run(0, 42, 0.7, 100)
        v.add_run(1, 42, 0.8, 100)
        report = format_averaging_report(v)
        assert 'Averaging Validation Report' in report
        assert 'report_test' in report
        assert 'Mean score' in report
        assert 'Std deviation' in report
        assert 'Deterministic' in report
        assert 'Consistency' in report
        assert 'run' in report

    def test_empty_validator(self):
        v = AveragingValidator()
        report = format_averaging_report(v)
        assert 'Averaging Validation Report' in report
        assert 'Number of runs  : 0' in report
