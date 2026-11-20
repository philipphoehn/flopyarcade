"""C1: Surrogate model validation – sim-boost factor and recall test.

The surrogate model replaces full MODFLOW runs with a fast neural-network
prediction.  This module provides the metrics and validation logic to
quantify:

* **Sim-boost factor** – how many times faster the surrogate is compared
  to the full simulation (wall-clock speedup).
* **Recall** – the fraction of true-positive (captured) outcomes that the
  surrogate also predicts as captured.
* **Precision** – the fraction of surrogate-positive predictions that are
  actually captured.
* **F1 score** – harmonic mean of precision and recall.

Usage::

    from flopyarcade.surrogate_test import (
        SurrogateValidator,
        compute_boost_factor,
        compute_recall,
    )

    validator = SurrogateValidator(min_recall=0.85, min_boost=5.0)
    validator.add_prediction(predicted_captured=True, ground_truth_captured=True)
    validator.add_prediction(predicted_captured=False, ground_truth_captured=True)
    ...
    report = validator.report(full_sim_time_s=120.0, surrogate_time_s=0.8)
    assert report['pass']
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SurrogateMetrics:
    """Aggregated surrogate-model performance metrics.

    Attributes:
        n_total: Total number of prediction pairs.
        n_tp: True positives (predicted captured, actually captured).
        n_fp: False positives (predicted captured, actually NOT captured).
        n_fn: False negatives (predicted NOT captured, actually captured).
        n_tn: True negatives.
        recall: TP / (TP + FN), or 0.0 if denominator is 0.
        precision: TP / (TP + FP), or 0.0 if denominator is 0.
        f1: 2 * P * R / (P + R), or 0.0 if P + R == 0.
        accuracy: (TP + TN) / total.
    """
    n_total: int = 0
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0
    n_tn: int = 0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0


@dataclass
class BoostReport:
    """Sim-boost factor report.

    Attributes:
        full_sim_time_s: Total wall-clock time for full MODFLOW runs.
        surrogate_time_s: Total wall-clock time for surrogate predictions.
        boost_factor: full / surrogate (>= 1 means surrogate is faster).
        speedup_pct: (1 - surrogate/full) * 100, positive = faster.
        min_boost: Required minimum boost factor to pass.
        pass_boost: True if boost_factor >= min_boost.
    """
    full_sim_time_s: float = 0.0
    surrogate_time_s: float = 0.0
    boost_factor: float = 0.0
    speedup_pct: float = 0.0
    min_boost: float = 1.0
    pass_boost: bool = False


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_boost_factor(
    full_sim_time_s: float,
    surrogate_time_s: float,
    min_boost: float = 1.0,
) -> BoostReport:
    """Compute the simulation speedup (boost) factor.

    Args:
        full_sim_time_s: Total wall-clock seconds for full MODFLOW runs.
        surrogate_time_s: Total wall-clock seconds for surrogate predictions.
        min_boost: Minimum acceptable boost factor (default 1.0 = no speedup).

    Returns:
        BoostReport.

    Raises:
        ValueError: If either time is negative or surrogate_time is 0.
    """
    if full_sim_time_s < 0 or surrogate_time_s < 0:
        raise ValueError("Times must be non-negative")
    if surrogate_time_s == 0:
        raise ValueError("surrogate_time_s must be > 0")

    boost = full_sim_time_s / surrogate_time_s
    speedup_pct = (1.0 - surrogate_time_s / full_sim_time_s) * 100.0 \
        if full_sim_time_s > 0 else 0.0

    return BoostReport(
        full_sim_time_s=full_sim_time_s,
        surrogate_time_s=surrogate_time_s,
        boost_factor=boost,
        speedup_pct=speedup_pct,
        min_boost=min_boost,
        pass_boost=boost >= min_boost,
    )


def compute_recall(
    predictions: List[bool],
    ground_truth: List[bool],
) -> float:
    """Compute binary recall: TP / (TP + FN).

    Args:
        predictions: Predicted outcomes (True = captured).
        ground_truth: Actual outcomes (True = captured).

    Returns:
        Recall in [0, 1]. Returns 0.0 if there are no positive instances.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"ground_truth={len(ground_truth)}"
        )
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0.0


def compute_precision(
    predictions: List[bool],
    ground_truth: List[bool],
) -> float:
    """Compute binary precision: TP / (TP + FP).

    Args:
        predictions: Predicted outcomes.
        ground_truth: Actual outcomes.

    Returns:
        Precision in [0, 1]. Returns 0.0 if there are no positive predictions.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"ground_truth={len(ground_truth)}"
        )
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    denominator = tp + fp
    return tp / denominator if denominator > 0 else 0.0


def compute_f1(
    predictions: List[bool],
    ground_truth: List[bool],
) -> float:
    """Compute F1 score: harmonic mean of precision and recall.

    Returns:
        F1 in [0, 1]. Returns 0.0 if P + R == 0.
    """
    p = compute_precision(predictions, ground_truth)
    r = compute_recall(predictions, ground_truth)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def compute_accuracy(
    predictions: List[bool],
    ground_truth: List[bool],
) -> float:
    """Compute overall accuracy: (TP + TN) / total.

    Returns:
        Accuracy in [0, 1]. Returns 0.0 if total is 0.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Length mismatch")
    n = len(predictions)
    if n == 0:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / n


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class SurrogateValidator:
    """Accumulates surrogate predictions and validates against thresholds.

    Args:
        min_recall: Minimum acceptable recall (default 0.85).
        min_boost: Minimum acceptable boost factor (default 5.0).
        min_f1: Minimum acceptable F1 score (default 0.75).
    """

    def __init__(
        self,
        min_recall: float = 0.85,
        min_boost: float = 5.0,
        min_f1: float = 0.75,
    ) -> None:
        self.min_recall = min_recall
        self.min_boost = min_boost
        self.min_f1 = min_f1
        self._predictions: List[bool] = []
        self._ground_truth: List[bool] = []

    @property
    def n_samples(self) -> int:
        return len(self._predictions)

    def add_prediction(
        self,
        predicted_captured: bool,
        ground_truth_captured: bool,
    ) -> None:
        """Record a single prediction vs ground-truth pair.

        Args:
            predicted_captured: Surrogate's prediction (True = will capture).
            ground_truth_captured: Actual MODFLOW result (True = captured).
        """
        self._predictions.append(bool(predicted_captured))
        self._ground_truth.append(bool(ground_truth_captured))

    def add_batch(
        self,
        predictions: List[bool],
        ground_truth: List[bool],
    ) -> None:
        """Record a batch of predictions.

        Args:
            predictions: List of predicted outcomes.
            ground_truth: List of actual outcomes.

        Raises:
            ValueError: If lengths don't match.
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Batch length mismatch: {len(predictions)} vs {len(ground_truth)}"
            )
        self._predictions.extend(bool(p) for p in predictions)
        self._ground_truth.extend(bool(g) for g in ground_truth)

    def metrics(self) -> SurrogateMetrics:
        """Compute current metrics from accumulated predictions."""
        p = self._predictions
        g = self._ground_truth

        tp = sum(1 for pi, gi in zip(p, g) if pi and gi)
        fp = sum(1 for pi, gi in zip(p, g) if pi and not gi)
        fn = sum(1 for pi, gi in zip(p, g) if not pi and gi)
        tn = sum(1 for pi, gi in zip(p, g) if not pi and not gi)
        n = len(p)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n if n > 0 else 0.0

        return SurrogateMetrics(
            n_total=n,
            n_tp=tp,
            n_fp=fp,
            n_fn=fn,
            n_tn=tn,
            recall=recall,
            precision=precision,
            f1=f1,
            accuracy=accuracy,
        )

    def report(
        self,
        full_sim_time_s: Optional[float] = None,
        surrogate_time_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a full validation report.

        Args:
            full_sim_time_s: Total full-sim time (optional; omit to skip boost check).
            surrogate_time_s: Total surrogate time (optional).

        Returns:
            Dict with metrics, boost info, pass/fail, and issues list.
        """
        m = self.metrics()
        issues: List[str] = []

        # Recall gate
        if m.recall < self.min_recall:
            issues.append(
                f"recall {m.recall:.3f} < min_recall {self.min_recall}"
            )

        # F1 gate
        if m.f1 < self.min_f1:
            issues.append(f"F1 {m.f1:.3f} < min_f1 {self.min_f1}")

        # Boost gate (only if times provided)
        boost_report: Optional[BoostReport] = None
        if full_sim_time_s is not None and surrogate_time_s is not None:
            boost_report = compute_boost_factor(
                full_sim_time_s, surrogate_time_s, min_boost=self.min_boost
            )
            if not boost_report.pass_boost:
                issues.append(
                    f"boost factor {boost_report.boost_factor:.2f}x "
                    f"< min_boost {self.min_boost}x"
                )

        passed = len(issues) == 0

        result: Dict[str, Any] = {
            'pass': passed,
            'issues': issues,
            'metrics': m,
            'n_samples': m.n_total,
        }
        if boost_report is not None:
            result['boost'] = boost_report

        return result

    def reset(self) -> None:
        """Clear all accumulated predictions."""
        self._predictions.clear()
        self._ground_truth.clear()
