"""B25: Agent-vs-human paired t-test with Cohen's d effect size.

Provides paired statistical comparison of two samples collected under
identical conditions (e.g. same 10 seeds) for agent vs. human performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from scipy.stats import t as _t_dist
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

import numpy as np

ALPHA = 0.05


@dataclass
class PairedResult:
    """Container for a single paired t-test result."""
    name: str
    n: int
    mean_diff: float
    t_stat: float
    p_value: float
    cohens_d: float
    significant: bool
    ci_low: float
    ci_high: float


def _t_cdf(x: float, df: int) -> float:
    """Two-sided p-value helper using scipy or manual approximation."""
    if _HAS_SCIPY:
        return _t_dist.cdf(x, df)
    # Fallback: use the regularized incomplete beta function approximation
    # For practical purposes with df >= 1, use a simple approximation.
    # This is a rough normal approximation fallback.
    return 0.5 + 0.5 * math.erf(x / math.sqrt(2.0))


def _t_ppf(p: float, df: int) -> float:
    """Inverse t CDF (quantile) for confidence interval critical value."""
    if _HAS_SCIPY:
        return _t_dist.ppf(p, df)
    # Fallback: use normal quantile approximation (valid for df > 30)
    # For small df this is approximate
    # Using Wilson-Hilferty approximation for t quantile
    z = _normal_ppf(p)
    # Approximate t from z
    return z * (1 + (z * z - 1) / (4 * df) + (5 * z**4 - 2 * z**2 - 2) / (32 * df * df))


def _normal_ppf(p: float) -> float:
    """Approximate inverse normal CDF (Acklam's algorithm)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def paired_t_test(
    sample_a: list[float],
    sample_b: list[float],
    name: str = "paired_test",
) -> PairedResult:
    """Compute a paired t-test between two equal-length samples.

    Parameters
    ----------
    sample_a : list of float
        First sample (e.g. agent scores per seed).
    sample_b : list of float
        Second sample (e.g. human scores per seed).
    name : str
        Label for the result.

    Returns
    -------
    PairedResult

    Raises
    ------
    ValueError
        If samples have different lengths or are empty.
    """
    if len(sample_a) == 0:
        raise ValueError("Samples must not be empty.")
    if len(sample_a) != len(sample_b):
        raise ValueError(
            f"Sample lengths must be equal: got {len(sample_a)} and {len(sample_b)}"
        )

    n = len(sample_a)
    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)

    d = a - b
    mean_d = float(np.mean(d))

    if n < 2:
        raise ValueError("Need at least 2 paired observations.")

    std_d = float(np.std(d, ddof=1))

    if std_d == 0.0:
        # All differences are zero (or nearly so)
        t_stat = 0.0
        p_value = 1.0
        cohens_d = 0.0
        ci_low = mean_d
        ci_high = mean_d
        return PairedResult(
            name=name, n=n, mean_diff=mean_d, t_stat=t_stat,
            p_value=p_value, cohens_d=cohens_d,
            significant=False, ci_low=ci_low, ci_high=ci_high,
        )

    se = std_d / math.sqrt(n)
    t_stat = mean_d / se
    df = n - 1

    # Two-sided p-value
    p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
    # Clamp to [0, 1]
    p_value = max(0.0, min(1.0, p_value))

    # Cohen's d (paired)
    cohens_d = mean_d / std_d

    # 95% CI on mean difference
    t_crit = _t_ppf(1 - ALPHA / 2, df)
    ci_low = mean_d - t_crit * se
    ci_high = mean_d + t_crit * se

    significant = bool(p_value < ALPHA)

    return PairedResult(
        name=name, n=n, mean_diff=float(mean_d), t_stat=float(t_stat),
        p_value=float(p_value), cohens_d=float(cohens_d),
        significant=bool(significant), ci_low=float(ci_low), ci_high=float(ci_high),
    )


def cohens_d_benchmark(sample_a: list[float], sample_b: list[float]) -> float:
    """Compute Cohen's d for paired samples (standalone effect size).

    d = mean(differences) / std(differences, ddof=1)
    """
    if len(sample_a) != len(sample_b):
        raise ValueError("Sample lengths must be equal.")
    if len(sample_a) < 2:
        raise ValueError("Need at least 2 paired observations.")

    d = np.array(sample_a, dtype=np.float64) - np.array(sample_b, dtype=np.float64)
    std_d = float(np.std(d, ddof=1))
    if std_d == 0.0:
        return 0.0
    return float(np.mean(d)) / std_d


def format_paired_report(result: PairedResult) -> str:
    """Format a human-readable paired test report."""
    sig_str = "SIGNIFICANT" if result.significant else "not significant"
    lines = [
        f"Paired t-test: {result.name}",
        f"  n = {result.n}",
        f"  mean_diff = {result.mean_diff:.6f}",
        f"  t({result.n - 1}) = {result.t_stat:.4f}",
        f"  p = {result.p_value:.6f}",
        f"  Cohen's d = {result.cohens_d:.4f}",
        f"  95% CI: [{result.ci_low:.6f}, {result.ci_high:.6f}]",
        f"  Result: {sig_str} (alpha={ALPHA})",
    ]
    return "\n".join(lines)


def batch_paired_test(
    results_a: dict[str, list[float]],
    results_b: dict[str, list[float]],
) -> dict[str, PairedResult]:
    """Run paired t-test for each common key in both dictionaries.

    Non-common keys are skipped.
    """
    common_keys = sorted(set(results_a.keys()) & set(results_b.keys()))
    out: dict[str, PairedResult] = {}
    for key in common_keys:
        out[key] = paired_t_test(results_a[key], results_b[key], name=key)
    return out
