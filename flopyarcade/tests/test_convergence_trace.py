"""Tests for flopyarcade.convergence_trace (ROADMAP P2)."""

import json
import tempfile
from pathlib import Path

import pytest

from flopyarcade.convergence_trace import (
    ConvergenceAnalyzer,
    ConvergenceStep,
    format_convergence_report,
    parse_convergence_log,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SAMPLE_STRUCTURED = """\
STEP 1
  Iterations: 12
  Condition Number: 1.5e10
  Residual: 3.2e-8
  Converged: Yes
  Elapsed: 45.2 ms
STEP 2
  Iterations: 8
  Condition Number: 2.1e10
  Residual: 1.1e-9
  Converged: Yes
  Elapsed: 30.1 ms
STEP 3
  Iterations: 15
  Condition Number: 0.9e10
  Residual: 5.5e-8
  Converged: No
  Elapsed: 55.0 ms
"""

SAMPLE_NUMERIC = """\
  12  1.5e+10  3.2e-08  YES  45.2
  8   2.1e+10  1.1e-09  YES  30.1
  15  0.9e+10  5.5e-08  NO   55.0
"""


def make_steps(iterations: list[int], converged: list[bool] | None = None) -> list[ConvergenceStep]:
    """Quick helper to build steps from iteration counts."""
    if converged is None:
        converged = [True] * len(iterations)
    return [
        ConvergenceStep(
            step=i + 1,
            n_iterations=it,
            condition_number=1.0e10,
            residual=1.0e-8,
            converged=conv,
            elapsed_ms=10.0,
        )
        for i, (it, conv) in enumerate(zip(iterations, converged))
    ]


# ---------------------------------------------------------------------------
# parse_convergence_log
# ---------------------------------------------------------------------------

class TestParseConvergenceLog:
    def test_structured_basic_three_steps(self):
        """Parse structured 3-step log correctly."""
        steps = parse_convergence_log(SAMPLE_STRUCTURED)
        assert len(steps) == 3
        assert steps[0].step == 1
        assert steps[0].n_iterations == 12
        assert steps[0].condition_number == pytest.approx(1.5e10)
        assert steps[0].residual == pytest.approx(3.2e-8)
        assert steps[0].converged is True
        assert steps[0].elapsed_ms == pytest.approx(45.2)

        assert steps[1].step == 2
        assert steps[1].n_iterations == 8
        assert steps[1].converged is True

        assert steps[2].step == 3
        assert steps[2].n_iterations == 15
        assert steps[2].converged is False

    def test_numeric_format(self):
        """Parse numeric space-separated lines."""
        steps = parse_convergence_log(SAMPLE_NUMERIC)
        assert len(steps) == 3
        assert steps[0].n_iterations == 12
        assert steps[0].condition_number == pytest.approx(1.5e10)
        assert steps[0].residual == pytest.approx(3.2e-08)
        assert steps[0].converged is True
        assert steps[0].elapsed_ms == pytest.approx(45.2)
        assert steps[2].converged is False

    def test_empty_string_returns_empty_list(self):
        """Empty input returns []."""
        assert parse_convergence_log("") == []
        assert parse_convergence_log("   \n  \n  ") == []

    def test_malformed_line_skipped(self):
        """Lines that don't match any pattern are ignored."""
        text = """\
This is a random line
STEP 1
  Iterations: 5
  Condition Number: 1.0e9
  Residual: 1.0e-7
  Converged: Yes
  Elapsed: 10.0 ms
garbage data here 12345
"""
        steps = parse_convergence_log(text)
        # Only the valid step should be parsed
        assert len(steps) == 1
        assert steps[0].n_iterations == 5


# ---------------------------------------------------------------------------
# ConvergenceAnalyzer
# ---------------------------------------------------------------------------

class TestConvergenceAnalyzer:
    def test_max_iterations(self):
        steps = make_steps([5, 20, 10])
        analyzer = ConvergenceAnalyzer(steps)
        assert analyzer.max_iterations() == 20

    def test_avg_iterations(self):
        steps = make_steps([10, 20, 30])
        analyzer = ConvergenceAnalyzer(steps)
        assert analyzer.avg_iterations() == pytest.approx(20.0)

    def test_worst_step(self):
        steps = make_steps([5, 50, 10])
        analyzer = ConvergenceAnalyzer(steps)
        worst = analyzer.worst_step()
        assert worst.n_iterations == 50
        assert worst.step == 2

    def test_convergence_rate(self):
        steps = make_steps([10, 20, 30, 40], converged=[True, False, True, False])
        analyzer = ConvergenceAnalyzer(steps)
        assert analyzer.convergence_rate() == pytest.approx(0.5)

    def test_is_stable_yes(self):
        """All steps under threshold → stable."""
        steps = make_steps([10, 20, 50])
        analyzer = ConvergenceAnalyzer(steps)
        assert analyzer.is_stable(threshold=100) is True

    def test_is_stable_no(self):
        """One step over threshold → not stable."""
        steps = make_steps([10, 200, 50])
        analyzer = ConvergenceAnalyzer(steps)
        assert analyzer.is_stable(threshold=100) is False

    def test_trend_improving(self):
        """Iterations decreasing → improving."""
        # 20 steps: first 10 avg high, last 10 avg low
        iters = [50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12]
        steps = make_steps(iters)
        analyzer = ConvergenceAnalyzer(steps)
        trend = analyzer.trend()
        assert trend["improving"] is True
        assert trend["first_avg_10"] > trend["last_avg_10"]

    def test_trend_worsening(self):
        """Iterations increasing → worsening."""
        iters = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
        steps = make_steps(iters)
        analyzer = ConvergenceAnalyzer(steps)
        trend = analyzer.trend()
        assert trend["improving"] is False
        assert trend["last_avg_10"] > trend["first_avg_10"]

    def test_summary_structure(self):
        steps = make_steps([10, 20, 30])
        analyzer = ConvergenceAnalyzer(steps)
        sm = analyzer.summary()
        assert sm["n_steps"] == 3
        assert sm["max_iterations"] == 30
        assert sm["avg_iterations"] == pytest.approx(20.0)
        assert sm["max_condition_number"] == pytest.approx(1.0e10)
        assert sm["convergence_rate"] == 1.0
        assert sm["is_stable"] is True
        assert "trend" in sm
        assert "worst_step" in sm

    def test_save_load_roundtrip(self, tmp_path):
        steps = make_steps([10, 20, 30])
        analyzer = ConvergenceAnalyzer(steps)
        out = tmp_path / "conv.json"
        analyzer.save(out)

        loaded = ConvergenceAnalyzer.load(out)
        assert len(loaded.steps) == 3
        for orig, new in zip(analyzer.steps, loaded.steps):
            assert orig.step == new.step
            assert orig.n_iterations == new.n_iterations
            assert orig.condition_number == new.condition_number
            assert orig.residual == new.residual
            assert orig.converged == new.converged
            assert orig.elapsed_ms == new.elapsed_ms

    def test_single_step_analyzer(self):
        steps = [ConvergenceStep(step=1, n_iterations=42, condition_number=9.9e9,
                                 residual=1e-7, converged=True, elapsed_ms=5.0)]
        analyzer = ConvergenceAnalyzer(steps)
        assert analyzer.max_iterations() == 42
        assert analyzer.avg_iterations() == 42.0
        assert analyzer.worst_step().n_iterations == 42
        assert analyzer.convergence_rate() == 1.0
        assert analyzer.is_stable() is True

    def test_all_non_converged_rate_zero(self):
        steps = make_steps([10, 20, 30], converged=[False, False, False])
        analyzer = ConvergenceAnalyzer(steps)
        assert analyzer.convergence_rate() == 0.0


# ---------------------------------------------------------------------------
# format_convergence_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_report_contains_step_numbers(self):
        steps = make_steps([10, 20, 30])
        analyzer = ConvergenceAnalyzer(steps)
        report = format_convergence_report(analyzer)
        assert "1" in report
        assert "2" in report
        assert "3" in report
        assert "Step" in report
        assert "MODFLOW Convergence Report" in report

    def test_report_contains_summary_metrics(self):
        steps = make_steps([10, 20])
        analyzer = ConvergenceAnalyzer(steps)
        report = format_convergence_report(analyzer)
        assert "Max iter" in report
        assert "Convergence rate" in report
