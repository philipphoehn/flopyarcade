"""Bootstrap confidence intervals and seed-sensitivity statistics for benchmark scores.

ROADMAP B18: Bootstrap CI on benchmark scores.
"""

import numpy as np


def bootstrap_ci(scores, n_resamples=1000, ci_level=0.95, seed=None):
    """Compute bootstrap confidence interval for the mean of *scores*.

    Parameters
    ----------
    scores : array-like
        Observed benchmark scores (1-D).
    n_resamples : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level, e.g. 0.95 for a 95 % CI.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``mean``, ``ci_low``, ``ci_high``, ``n``, ``ci_level``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size

    if n == 0:
        raise ValueError("scores must contain at least one element")

    sample_mean = float(scores.mean())

    rng = np.random.default_rng(seed)

    if n == 1:
        # Degenerate case: every resample gives the same value
        ci_low = sample_mean
        ci_high = sample_mean
    else:
        # Vectorised bootstrap: draw n_resamples x n indices, compute means
        idx = rng.integers(0, n, size=(n_resamples, n))
        bootstrap_means = scores[idx].mean(axis=1)

        alpha = 1.0 - ci_level
        ci_low = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        ci_high = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return {
        "mean": sample_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "ci_level": ci_level,
    }


def bootstrap_test(sample_a, sample_b, n_resamples=1000, seed=None):
    """Two-sample bootstrap (permutation) test for difference of means.

    Parameters
    ----------
    sample_a, sample_b : array-like
        Two independent samples.
    n_resamples : int
        Number of bootstrap resamples.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``mean_diff``, ``ci_low``, ``ci_high``, ``p_value``,
    ``significant`` (bool, alpha = 0.05).
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)

    if a.size == 0 or b.size == 0:
        raise ValueError("Both samples must be non-empty")

    mean_diff = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)

    # --- Bootstrap CI for the difference of means ---
    bootstrap_diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        boot_a = a[rng.integers(0, a.size, size=a.size)]
        boot_b = b[rng.integers(0, b.size, size=b.size)]
        bootstrap_diffs[i] = boot_a.mean() - boot_b.mean()

    ci_low = float(np.percentile(bootstrap_diffs, 2.5))
    ci_high = float(np.percentile(bootstrap_diffs, 97.5))

    # --- Permutation p-value ---
    combined = np.concatenate([a, b])
    n_a = a.size
    obs_stat = abs(mean_diff)

    # For very large samples cap permutations at 100000 to stay practical
    max_perms = min(n_resamples, 100_000)
    null_count = 0
    for _ in range(max_perms):
        perm = rng.permutation(combined)
        perm_diff = abs(perm[:n_a].mean() - perm[n_a:].mean())
        if perm_diff >= obs_stat:
            null_count += 1

    p_value = (null_count + 1) / (max_perms + 1)

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def seed_sensitivity_table(results_dict, n_seeds=10):
    """Compute per-envtype summary statistics for seed sensitivity analysis.

    Parameters
    ----------
    results_dict : dict
        Mapping ``{envtype: list_of_scores}``.
    n_seeds : int
        Number of independent seeds (informational; stored in output).

    Returns
    -------
    dict of dicts.  Each inner dict has keys:
    ``mean``, ``std``, ``iqr``, ``cv``, ``n``.
    """
    table = {}
    for envtype, scores in results_dict.items():
        arr = np.asarray(scores, dtype=np.float64)
        n = arr.size
        if n == 0:
            table[envtype] = {
                "mean": 0.0,
                "std": 0.0,
                "iqr": 0.0,
                "cv": 0.0,
                "n": 0,
            }
            continue

        mean_val = float(arr.mean())
        std_val = float(arr.std(ddof=1)) if n > 1 else 0.0
        q75, q25 = np.percentile(arr, [75, 25])
        iqr_val = float(q75 - q25)
        cv_val = std_val / abs(mean_val) if mean_val != 0 else float("inf")

        table[envtype] = {
            "mean": mean_val,
            "std": std_val,
            "iqr": iqr_val,
            "cv": cv_val,
            "n": n,
        }
    return table
