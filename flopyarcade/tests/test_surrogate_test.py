"""Tests for flopyarcade.surrogate_test (C1: surrogate recall + boost)."""

import pytest

from flopyarcade.surrogate_test import (
    SurrogateMetrics,
    BoostReport,
    compute_boost_factor,
    compute_recall,
    compute_precision,
    compute_f1,
    compute_accuracy,
    SurrogateValidator,
)


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestSurrogateMetrics:
    def test_defaults(self):
        m = SurrogateMetrics()
        assert m.n_total == 0
        assert m.n_tp == 0
        assert m.recall == 0.0
        assert m.precision == 0.0
        assert m.f1 == 0.0
        assert m.accuracy == 0.0


class TestBoostReport:
    def test_defaults(self):
        b = BoostReport()
        assert b.boost_factor == 0.0
        assert b.pass_boost is False


# ---------------------------------------------------------------------------
# compute_boost_factor
# ---------------------------------------------------------------------------

class TestComputeBoostFactor:
    def test_basic(self):
        r = compute_boost_factor(100.0, 10.0)
        assert r.boost_factor == 10.0
        assert r.speedup_pct == pytest.approx(90.0)
        assert r.pass_boost is True

    def test_min_boost_pass(self):
        r = compute_boost_factor(100.0, 10.0, min_boost=5.0)
        assert r.pass_boost is True

    def test_min_boost_fail(self):
        r = compute_boost_factor(100.0, 10.0, min_boost=20.0)
        assert r.pass_boost is False

    def test_equal_times(self):
        r = compute_boost_factor(50.0, 50.0)
        assert r.boost_factor == 1.0
        assert r.speedup_pct == 0.0

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            compute_boost_factor(-1.0, 1.0)

    def test_zero_surrogate_raises(self):
        with pytest.raises(ValueError):
            compute_boost_factor(10.0, 0.0)

    def test_zero_full(self):
        r = compute_boost_factor(0.0, 1.0)
        assert r.boost_factor == 0.0
        assert r.speedup_pct == 0.0


# ---------------------------------------------------------------------------
# compute_recall / precision / f1 / accuracy
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_recall_perfect(self):
        assert compute_recall([True, True], [True, True]) == 1.0

    def test_recall_zero(self):
        assert compute_recall([False, False], [True, True]) == 0.0

    def test_recall_no_positives(self):
        assert compute_recall([False, True], [False, False]) == 0.0

    def test_recall_mixed(self):
        # TP=1, FN=1 => recall = 0.5
        assert compute_recall([True, False], [True, True]) == 0.5

    def test_precision_perfect(self):
        assert compute_precision([True, True], [True, True]) == 1.0

    def test_precision_zero(self):
        assert compute_precision([True, True], [False, False]) == 0.0

    def test_precision_no_positive_preds(self):
        assert compute_precision([False, False], [True, False]) == 0.0

    def test_f1_perfect(self):
        assert compute_f1([True, True], [True, True]) == 1.0

    def test_f1_zero(self):
        assert compute_f1([False, False], [True, True]) == 0.0

    def test_f1_half(self):
        # P=0.5, R=1.0 => F1 = 2*0.5*1/(0.5+1) = 0.667
        f1 = compute_f1([True, True], [True, False])
        assert f1 == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))

    def test_accuracy_perfect(self):
        assert compute_accuracy([True, False], [True, False]) == 1.0

    def test_accuracy_zero(self):
        assert compute_accuracy([True, True], [False, False]) == 0.0

    def test_accuracy_empty(self):
        assert compute_accuracy([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_recall([True], [True, True])


# ---------------------------------------------------------------------------
# SurrogateValidator
# ---------------------------------------------------------------------------

class TestSurrogateValidator:
    def test_init_defaults(self):
        v = SurrogateValidator()
        assert v.min_recall == 0.85
        assert v.min_boost == 5.0
        assert v.min_f1 == 0.75
        assert v.n_samples == 0

    def test_add_prediction(self):
        v = SurrogateValidator()
        v.add_prediction(True, True)
        assert v.n_samples == 1

    def test_add_batch(self):
        v = SurrogateValidator()
        v.add_batch([True, True, False], [True, False, False])
        assert v.n_samples == 3

    def test_add_batch_mismatch_raises(self):
        v = SurrogateValidator()
        with pytest.raises(ValueError):
            v.add_batch([True, True], [True])

    def test_metrics_all_tp(self):
        v = SurrogateValidator()
        v.add_batch([True, True, True], [True, True, True])
        m = v.metrics()
        assert m.n_tp == 3
        assert m.recall == 1.0
        assert m.precision == 1.0
        assert m.f1 == 1.0
        assert m.accuracy == 1.0

    def test_metrics_mixed(self):
        v = SurrogateValidator()
        # TP=2, FP=1, FN=1, TN=1
        v.add_batch([True, True, True, False, False],
                    [True, True, False, True, False])
        m = v.metrics()
        assert m.n_tp == 2
        assert m.n_fp == 1
        assert m.n_fn == 1
        assert m.n_tn == 1
        assert m.recall == pytest.approx(2/3)
        assert m.precision == pytest.approx(2/3)

    def test_report_pass(self):
        v = SurrogateValidator(min_recall=0.5, min_f1=0.5, min_boost=2.0)
        v.add_batch([True, True], [True, True])
        r = v.report(full_sim_time_s=10.0, surrogate_time_s=1.0)
        assert r['pass'] is True
        assert r['boost'].boost_factor == 10.0

    def test_report_fail_recall(self):
        v = SurrogateValidator(min_recall=0.9)
        v.add_batch([True, False], [True, True])  # recall = 0.5
        r = v.report()
        assert r['pass'] is False
        assert any('recall' in i for i in r['issues'])

    def test_report_fail_boost(self):
        v = SurrogateValidator(min_boost=100.0)
        v.add_batch([True], [True])
        r = v.report(full_sim_time_s=10.0, surrogate_time_s=1.0)  # 10x < 100x
        assert r['pass'] is False
        assert any('boost' in i for i in r['issues'])

    def test_report_no_boost_times(self):
        v = SurrogateValidator(min_recall=0.5, min_f1=0.5)
        v.add_batch([True], [True])
        r = v.report()  # no times
        assert 'boost' not in r

    def test_reset(self):
        v = SurrogateValidator()
        v.add_batch([True, True], [True, True])
        v.reset()
        assert v.n_samples == 0
        assert v.metrics().n_total == 0

    def test_empty_metrics(self):
        v = SurrogateValidator()
        m = v.metrics()
        assert m.n_total == 0
        assert m.recall == 0.0
        assert m.f1 == 0.0
