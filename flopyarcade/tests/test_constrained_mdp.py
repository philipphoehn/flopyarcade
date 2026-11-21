"""Tests for constrained_mdp (ROADMAP R8)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from flopyarcade.constrained_mdp import (
    ConstrainedReward,
    ConstraintSpec,
    LagrangianDual,
    ViolationTracker,
    format_constraint_report,
)


# ---------------------------------------------------------------------------
# ConstraintSpec
# ---------------------------------------------------------------------------

class TestConstraintSpec:
    def test_defaults(self):
        spec = ConstraintSpec()
        assert spec.min_head == 0.5
        assert spec.penalty_weight == 1.0
        assert spec.violation_log == []


# ---------------------------------------------------------------------------
# LagrangianDual
# ---------------------------------------------------------------------------

class TestLagrangianDual:
    def test_update_increases_lambda(self):
        dual = LagrangianDual(initial_lambda=1.0, learning_rate=0.1, max_lambda=10.0)
        before = dual.get_lambda()
        dual.update(violation=2.0)
        assert dual.get_lambda() > before
        assert dual.get_lambda() == pytest.approx(1.0 + 0.1 * 2.0)

    def test_clamped_at_max(self):
        dual = LagrangianDual(initial_lambda=9.5, learning_rate=1.0, max_lambda=10.0)
        dual.update(violation=100.0)
        assert dual.get_lambda() == 10.0

    def test_get_penalty_zero_when_satisfied(self):
        dual = LagrangianDual(initial_lambda=2.0)
        dual.min_head = 0.5
        # constraint_value >= min_head => no penalty
        assert dual.get_penalty(constraint_value=0.7) == 0.0
        assert dual.get_penalty(constraint_value=0.5) == 0.0

    def test_get_penalty_positive_when_violated(self):
        dual = LagrangianDual(initial_lambda=2.0)
        dual.min_head = 0.5
        # constraint_value 0.3, shortfall = 0.2, penalty = 2.0 * 0.2 = 0.4
        assert dual.get_penalty(constraint_value=0.3) == pytest.approx(0.4)

    def test_history_and_reset(self):
        dual = LagrangianDual(initial_lambda=1.0, learning_rate=0.1)
        dual.update(1.0)
        h = dual.history()
        assert len(h) == 2
        assert h[0] == 1.0
        dual.reset()
        assert dual.get_lambda() == 1.0
        assert dual.history() == [1.0]

    def test_dual_adapts_with_repeated_violations(self):
        dual = LagrangianDual(initial_lambda=1.0, learning_rate=0.1, max_lambda=10.0)
        lam_0 = dual.get_lambda()
        for _ in range(5):
            dual.update(violation=1.0)
        lam_5 = dual.get_lambda()
        assert lam_5 > lam_0
        # should be 1.0 + 5*0.1*1.0 = 1.5
        assert lam_5 == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# ConstrainedReward
# ---------------------------------------------------------------------------

class TestConstrainedReward:
    def _make(self, min_head=0.5):
        spec = ConstraintSpec(min_head=min_head)
        dual = LagrangianDual(initial_lambda=1.0, learning_rate=0.1, max_lambda=10.0)
        base_fn = lambda state: state.get("base", 1.0)
        cr = ConstrainedReward(base_fn, spec, dual)
        return cr, spec, dual

    def test_compute_returns_correct_keys(self):
        cr, _, _ = self._make()
        result = cr.compute({"constraint_value": 0.6, "base": 2.0})
        for key in ("total_reward", "base_reward", "penalty", "constraint_value", "violated"):
            assert key in result

    def test_compute_with_violation(self):
        cr, spec, dual = self._make(min_head=0.5)
        result = cr.compute({"constraint_value": 0.3, "base": 2.0})
        assert result["violated"] is True
        assert result["penalty"] > 0.0
        assert result["total_reward"] < result["base_reward"]
        assert len(spec.violation_log) == 1

    def test_compute_without_violation(self):
        cr, spec, dual = self._make(min_head=0.5)
        result = cr.compute({"constraint_value": 0.7, "base": 2.0})
        assert result["violated"] is False
        assert result["penalty"] == 0.0
        assert result["total_reward"] == pytest.approx(2.0)
        assert len(spec.violation_log) == 0

    def test_accumulate_stats(self):
        cr, spec, dual = self._make(min_head=0.5)
        steps = [
            {"total_reward": 1.0, "base_reward": 2.0, "penalty": 1.0, "constraint_value": 0.4, "violated": True},
            {"total_reward": 2.0, "base_reward": 2.0, "penalty": 0.0, "constraint_value": 0.6, "violated": False},
            {"total_reward": 1.5, "base_reward": 2.0, "penalty": 0.5, "constraint_value": 0.45, "violated": True},
        ]
        agg = cr.accumulate(steps)
        assert agg["total"] == pytest.approx(4.5)
        assert agg["mean_base"] == pytest.approx(2.0)
        assert agg["mean_penalty"] == pytest.approx(0.5)
        assert agg["violation_rate"] == pytest.approx(2.0 / 3.0)
        assert agg["final_lambda"] >= 1.0


# ---------------------------------------------------------------------------
# ViolationTracker
# ---------------------------------------------------------------------------

class TestViolationTracker:
    def test_record_and_rate(self):
        t = ViolationTracker(threshold=0.5)
        t.record(0.6)  # safe
        t.record(0.3)  # violation
        t.record(0.7)  # safe
        t.record(0.4)  # violation
        assert t.n_violations() == 2
        assert t.violation_rate() == pytest.approx(0.5)

    def test_max_violation(self):
        t = ViolationTracker(threshold=0.5)
        t.record(0.6)
        t.record(0.3)  # violation magnitude 0.2
        t.record(0.1)  # violation magnitude 0.4
        assert t.max_violation() == pytest.approx(0.4)

    def test_rolling_rate(self):
        t = ViolationTracker(threshold=0.5)
        for v in [0.6, 0.7, 0.8, 0.9, 0.95]:  # 5 safe
            t.record(v)
        for v in [0.4, 0.3]:  # 2 violations
            t.record(v)
        # window=7 => last 7 values: 0.8,0.9,0.95,0.4,0.3 => wait, 7 values
        # last 7: 0.7,0.8,0.9,0.95,0.4,0.3 => no, 7 total recorded, all
        # Actually 7 recorded total, window=7 => all 7: 5 safe + 2 viol => 2/7
        rate = t.rolling_rate(window=7)
        assert rate == pytest.approx(2.0 / 7.0)

    def test_is_safe_true(self):
        t = ViolationTracker(threshold=0.5)
        for _ in range(200):
            t.record(0.6)
        # 0 violations out of 200 => rate = 0 < 0.01
        assert t.is_safe() is True

    def test_is_safe_false(self):
        t = ViolationTracker(threshold=0.5)
        for _ in range(10):
            t.record(0.3)  # all violations => rate = 1.0 >= 0.01
        assert t.is_safe() is False

    def test_save_load_roundtrip(self):
        t = ViolationTracker(threshold=0.5)
        for v in [0.6, 0.3, 0.7, 0.4, 0.8]:
            t.record(v)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            t.save(path)
            t2 = ViolationTracker.load(path)
            assert t2.threshold == 0.5
            assert t2.n_violations() == t.n_violations()
            assert t2.violation_rate() == pytest.approx(t.violation_rate())
            assert t2.max_violation() == pytest.approx(t.max_violation())
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# format_constraint_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_contains_violation_rate(self):
        results = [
            {"total": 5.0, "mean_base": 2.0, "mean_penalty": 0.5, "violation_rate": 0.25, "final_lambda": 1.5},
            {"total": 4.0, "mean_base": 2.0, "mean_penalty": 0.25, "violation_rate": 0.10, "final_lambda": 1.6},
        ]
        report = format_constraint_report(results)
        assert "violation rate" in report.lower()
        assert "lambda" in report.lower()
        assert "penalty" in report.lower()

    def test_empty_results(self):
        report = format_constraint_report([])
        assert "No episodes" in report


# ---------------------------------------------------------------------------
# Zero-violation episode
# ---------------------------------------------------------------------------

class TestZeroViolation:
    def test_all_penalties_zero(self):
        spec = ConstraintSpec(min_head=0.5)
        dual = LagrangianDual(initial_lambda=1.0, learning_rate=0.1, max_lambda=10.0)
        base_fn = lambda s: s["base"]
        cr = ConstrainedReward(base_fn, spec, dual)

        steps = []
        for _ in range(10):
            r = cr.compute({"constraint_value": 0.8, "base": 1.0})
            assert r["penalty"] == 0.0
            assert r["violated"] is False
            steps.append(r)

        agg = cr.accumulate(steps)
        assert agg["mean_penalty"] == 0.0
        assert agg["violation_rate"] == 0.0
        assert agg["final_lambda"] == pytest.approx(1.0)  # unchanged
