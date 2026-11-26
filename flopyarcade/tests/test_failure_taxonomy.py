"""Tests for flopyarcade.failure_taxonomy (ROADMAP P6)."""

import json
import os
import tempfile

import pytest

from flopyarcade.failure_taxonomy import (
    FailureClassifier,
    FailureRecord,
    FailureType,
    classify_failure,
    format_failure_report,
)


# ─── classify_failure ─────────────────────────────────────────────────────────

class TestClassifyFailure:
    def test_nan(self):
        """NaN detection takes top priority."""
        result = classify_failure(
            reward_history=[1.0, 2.0, 3.0],
            obs_history=[0.5, 0.6],
            step=5,
            max_steps=100,
            nan_detected=True,
        )
        assert result == FailureType.NAN

    def test_no_progress_flat_reward(self):
        """Reaching max_steps with flat reward → NO_PROGRESS."""
        result = classify_failure(
            reward_history=[0.5, 0.5, 0.5, 0.5],
            obs_history=[0.3, 0.4],
            step=100,
            max_steps=100,
        )
        assert result == FailureType.NO_PROGRESS

    def test_convergence_fail(self):
        """High convergence_iters → CONVERGENCE_FAIL."""
        result = classify_failure(
            reward_history=[0.1, 0.5, 0.9],
            obs_history=[0.3, 0.4],
            step=50,
            max_steps=100,
            convergence_iters=600,
        )
        assert result == FailureType.CONVERGENCE_FAIL

    def test_boundary_hit(self):
        """High obs_norm (> 0.99) → BOUNDARY_HIT."""
        result = classify_failure(
            reward_history=[0.1, 0.5, 0.9],
            obs_history=[0.5, 0.995],
            step=50,
            max_steps=100,
        )
        assert result == FailureType.BOUNDARY_HIT

    def test_timeout_with_progress(self):
        """Reaching max_steps with reward progress → TIMEOUT."""
        result = classify_failure(
            reward_history=[0.1, 0.5, 0.9, 1.0],
            obs_history=[0.3, 0.4],
            step=100,
            max_steps=100,
        )
        assert result == FailureType.TIMEOUT

    def test_success(self):
        """Before max_steps, improving reward, no issues → SUCCESS."""
        result = classify_failure(
            reward_history=[0.1, 0.5, 0.9],
            obs_history=[0.3, 0.4],
            step=30,
            max_steps=100,
        )
        assert result == FailureType.SUCCESS

    def test_priority_nan_over_all(self):
        """NaN beats everything else."""
        result = classify_failure(
            reward_history=[0.5, 0.5],
            obs_history=[0.999],
            step=100,
            max_steps=100,
            convergence_iters=999,
            nan_detected=True,
        )
        assert result == FailureType.NAN

    def test_priority_no_progress_over_convergence(self):
        """No progress (flat reward at max_steps) beats convergence_fail."""
        result = classify_failure(
            reward_history=[0.5, 0.5, 0.5],
            obs_history=[0.3],
            step=100,
            max_steps=100,
            convergence_iters=600,
        )
        assert result == FailureType.NO_PROGRESS

    def test_priority_convergence_over_boundary(self):
        """Convergence fail beats boundary hit."""
        result = classify_failure(
            reward_history=[0.1, 0.5, 0.9],
            obs_history=[0.995],
            step=50,
            max_steps=100,
            convergence_iters=700,
        )
        assert result == FailureType.CONVERGENCE_FAIL


# ─── FailureClassifier ────────────────────────────────────────────────────────

class TestFailureClassifier:
    def _make_classifier(self):
        return FailureClassifier()

    def test_classify_returns_failure_record(self):
        """classify() returns a FailureRecord with correct fields."""
        fc = self._make_classifier()
        rec = fc.classify(
            episode=1,
            step=5,
            reward_history=[0.1, 0.2],
            obs_history=[0.3],
            max_steps=100,
        )
        assert isinstance(rec, FailureRecord)
        assert rec.episode == 1
        assert rec.step == 5
        assert rec.failure_type == FailureType.SUCCESS
        assert rec.reward_at_fail == 0.2
        assert isinstance(rec.details, str)

    def test_summary_counts_correct(self):
        """summary() returns correct per-type counts."""
        fc = self._make_classifier()
        # Add one of each type
        fc.classify(0, 100, [0.5, 0.5], [0.3], 100)          # NO_PROGRESS
        fc.classify(1, 50, [0.1, 0.9], [0.3], 100, 600)       # CONVERGENCE_FAIL
        fc.classify(2, 50, [0.1, 0.9], [0.999], 100)          # BOUNDARY_HIT
        fc.classify(3, 100, [0.1, 1.0], [0.3], 100)           # TIMEOUT
        fc.classify(4, 5, [0.1, 0.2], [0.3], 100, nan_detected=True)  # NAN
        fc.classify(5, 10, [0.1, 0.5], [0.3], 100)            # SUCCESS

        s = fc.summary()
        assert s["total_episodes"] == 6
        assert s["counts"]["no_progress"] == 1
        assert s["counts"]["convergence_fail"] == 1
        assert s["counts"]["boundary_hit"] == 1
        assert s["counts"]["timeout"] == 1
        assert s["counts"]["nan"] == 1
        assert s["counts"]["success"] == 1
        # failure_rate = 5/6
        assert abs(s["failure_rate"] - 5/6) < 1e-9

    def test_most_common_failure(self):
        """most_common_failure returns the most frequent non-success type."""
        fc = self._make_classifier()
        # 3 timeout, 1 success, 1 nan
        fc.classify(0, 100, [0.1, 1.0], [0.3], 100)
        fc.classify(1, 100, [0.1, 1.0], [0.3], 100)
        fc.classify(2, 100, [0.1, 1.0], [0.3], 100)
        fc.classify(3, 10, [0.1, 0.5], [0.3], 100)
        fc.classify(4, 10, [0.1], [0.3], 100, nan_detected=True)

        assert fc.most_common_failure() == FailureType.TIMEOUT

    def test_failure_rate_calculation(self):
        """failure_rate returns correct fraction for a given type."""
        fc = self._make_classifier()
        fc.classify(0, 100, [0.1, 1.0], [0.3], 100)  # TIMEOUT
        fc.classify(1, 10, [0.1, 0.5], [0.3], 100)    # SUCCESS

        assert fc.failure_rate(FailureType.TIMEOUT) == 0.5
        assert fc.failure_rate(FailureType.SUCCESS) == 0.5
        assert fc.failure_rate(FailureType.NAN) == 0.0

    def test_save_load_roundtrip(self):
        """save then load preserves all records and fields."""
        fc = self._make_classifier()
        fc.classify(0, 100, [0.5, 0.5], [0.3], 100)
        fc.classify(1, 50, [0.1, 0.9], [0.999], 100)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name

        fc.save(path)
        fc2 = FailureClassifier.load(path)

        assert len(fc2.records) == 2
        assert fc2.records[0].episode == 0
        assert fc2.records[0].failure_type == FailureType.NO_PROGRESS
        assert fc2.records[1].episode == 1
        assert fc2.records[1].failure_type == FailureType.BOUNDARY_HIT
        assert fc2.records[1].obs_norm == pytest.approx(0.999)

        os.unlink(path)

    def test_format_failure_report_contains_type_names(self):
        """Report text contains all FailureType value strings."""
        fc = self._make_classifier()
        fc.classify(0, 10, [0.1, 0.5], [0.3], 100)

        report = format_failure_report(fc)
        for ft in FailureType:
            assert ft.value in report, f"Missing {ft.value} in report"
        assert "FAILURE TAXONOMY REPORT" in report

    def test_empty_classifier_summary(self):
        """Empty classifier: total=0, failure_rate=0."""
        fc = self._make_classifier()
        s = fc.summary()
        assert s["total_episodes"] == 0
        assert s["failure_rate"] == 0.0
        for ft in FailureType:
            assert s["counts"][ft.value] == 0

    def test_all_failures_scenario(self):
        """All episodes are failures → failure_rate == 1.0."""
        fc = self._make_classifier()
        fc.classify(0, 100, [0.5, 0.5], [0.3], 100)  # NO_PROGRESS
        fc.classify(1, 50, [0.1, 0.9], [0.3], 100, 900)  # CONVERGENCE_FAIL
        fc.classify(2, 10, [0.1], [0.3], 100, nan_detected=True)  # NAN

        s = fc.summary()
        assert s["total_episodes"] == 3
        assert s["failure_rate"] == 1.0
        assert s["counts"]["success"] == 0

    def test_most_common_failure_all_success(self):
        """All records are success → returns SUCCESS."""
        fc = self._make_classifier()
        fc.classify(0, 10, [0.1, 0.5], [0.3], 100)
        fc.classify(1, 20, [0.2, 0.6], [0.3], 100)

        assert fc.most_common_failure() == FailureType.SUCCESS

    def test_most_common_failure_empty_raises(self):
        """Empty classifier raises ValueError."""
        fc = self._make_classifier()
        with pytest.raises(ValueError, match="No records"):
            fc.most_common_failure()
