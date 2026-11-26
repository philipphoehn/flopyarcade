"""Tests for P8 checkpoint comparison tool."""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from flopyarcade.checkpoint_compare import (
    PolicyWeights,
    format_comparison_report,
    kl_divergence_p,
    layer_wise_diff,
    load_policy_weights,
    policy_distance,
    save_policy_weights,
    side_by_side_table,
)


# --- helpers ---

def _make_policy(name="test", n_in=4, n_hid=8, n_out=3, seed=0):
    rng = np.random.default_rng(seed)
    w = [
        *rng.standard_normal(n_in * n_hid).tolist(),
        *rng.standard_normal(n_hid * n_out).tolist(),
    ]
    b = [
        *rng.standard_normal(n_hid).tolist(),
        *rng.standard_normal(n_out).tolist(),
    ]
    return PolicyWeights(
        name=name,
        weights=w,
        biases=b,
        layer_sizes=[n_in, n_hid, n_out],
        timestamp="2025-01-01T00:00:00",
    )


def _make_policy_1layer(name="1layer", n_in=3, n_out=1, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n_in * n_out).tolist()
    b = rng.standard_normal(n_out).tolist()
    return PolicyWeights(
        name=name,
        weights=w,
        biases=b,
        layer_sizes=[n_in, n_out],
        timestamp="2025-01-01T00:00:00",
    )


# --- 1. PolicyWeights creation ---

def test_policy_weights_creation():
    p = PolicyWeights(
        name="m1",
        weights=[1.0, 2.0, 3.0],
        biases=[0.1, 0.2],
        layer_sizes=[3, 2],
        timestamp="2025-06-01",
    )
    assert p.name == "m1"
    assert p.weights == [1.0, 2.0, 3.0]
    assert p.biases == [0.1, 0.2]
    assert p.layer_sizes == [3, 2]
    assert p.timestamp == "2025-06-01"


# --- 2. save / load roundtrip ---

def test_save_load_roundtrip():
    p = _make_policy()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_policy_weights(p, path)
        loaded = load_policy_weights(path)
        assert loaded.name == p.name
        assert loaded.weights == p.weights
        assert loaded.biases == p.biases
        assert loaded.layer_sizes == p.layer_sizes
        assert loaded.timestamp == p.timestamp
    finally:
        os.unlink(path)


# --- 3. KL divergence identical ≈ 0 ---

def test_kl_divergence_identical():
    p = _make_policy(seed=7)
    kl = kl_divergence_p(p, p, n_samples=200)
    assert kl < 1e-6, f"KL(identical) should be ~0, got {kl}"


# --- 4. KL divergence different > 0 ---

def test_kl_divergence_different():
    pa = _make_policy(seed=1)
    pb = _make_policy(seed=99)
    kl = kl_divergence_p(pa, pb, n_samples=200)
    assert kl > 0.0, f"KL(different) should be > 0, got {kl}"


# --- 5. KL divergence both directions > 0 ---

def test_kl_divergence_both_directions():
    pa = _make_policy(seed=3)
    pb = _make_policy(seed=42)
    kl_ab = kl_divergence_p(pa, pb, n_samples=200)
    kl_ba = kl_divergence_p(pb, pa, n_samples=200)
    assert kl_ab > 0.0
    assert kl_ba > 0.0


# --- 6. policy_distance structure ---

def test_policy_distance_structure():
    pa = _make_policy(seed=1)
    pb = _make_policy(seed=2)
    d = policy_distance(pa, pb)
    expected_keys = {
        "kl_ab", "kl_ba", "js_divergence",
        "max_weight_diff", "mean_weight_diff", "relative_weight_diff",
    }
    assert set(d.keys()) == expected_keys
    for k in expected_keys:
        assert isinstance(d[k], float), f"{k} should be float, got {type(d[k])}"


# --- 7. policy_distance identical (all zeros) ---

def test_policy_distance_identical():
    p = _make_policy(seed=5)
    d = policy_distance(p, p)
    assert d["kl_ab"] < 1e-6
    assert d["kl_ba"] < 1e-6
    assert d["js_divergence"] < 1e-6
    assert d["max_weight_diff"] == 0.0
    assert d["mean_weight_diff"] == 0.0
    assert d["relative_weight_diff"] < 1e-6


# --- 8. side_by_side_table correct count ---

def test_side_by_side_table_count():
    pa = _make_policy(seed=1)
    pb = _make_policy(seed=2)
    table = side_by_side_table(pa, pb, top_n=5)
    assert len(table) == 5
    for row in table:
        assert "index" in row
        assert "layer" in row
        assert "value_a" in row
        assert "value_b" in row
        assert "diff" in row
        assert "pct_change" in row


# --- 9. side_by_side_table sorted by |diff| descending ---

def test_side_by_side_table_sorted():
    pa = _make_policy(seed=1)
    pb = _make_policy(seed=2)
    table = side_by_side_table(pa, pb, top_n=10)
    diffs = [abs(r["diff"]) for r in table]
    assert diffs == sorted(diffs, reverse=True), "Table not sorted by |diff| desc"


# --- 10. layer_wise_diff structure ---

def test_layer_wise_diff_structure():
    pa = _make_policy(seed=1)  # 4-8-3 → 2 layers
    pb = _make_policy(seed=2)
    layers = layer_wise_diff(pa, pb)
    assert len(layers) == 2
    for l in layers:
        assert "layer" in l
        assert "n_params" in l
        assert "mean_diff" in l
        assert "max_diff" in l
        assert "rms" in l


# --- 11. layer_wise_diff single layer ---

def test_layer_wise_diff_single_layer():
    pa = _make_policy_1layer(seed=1)  # 3-1 → 1 layer
    pb = _make_policy_1layer(seed=2)
    layers = layer_wise_diff(pa, pb)
    assert len(layers) == 1
    assert layers[0]["layer"] == 0
    assert layers[0]["n_params"] == 3  # 3*1


# --- 12. format_comparison_report contains text ---

def test_format_comparison_report_contains_text():
    pa = _make_policy(seed=1)
    pb = _make_policy(seed=2)
    comp = policy_distance(pa, pb)
    table = side_by_side_table(pa, pb, top_n=5)
    layers = layer_wise_diff(pa, pb)
    report = format_comparison_report(comp, table, layers)
    assert "CHECKPOINT COMPARISON REPORT" in report
    assert "KL(A||B)" in report
    assert "Layer" in report
    assert "Most-Changed Weights" in report
    assert len(report) > 100


# --- 13. 2-layer network (2-4-2) ---

def test_two_layer_network():
    rng = np.random.default_rng(0)
    w_a = [
        *rng.standard_normal(2 * 4).tolist(),
        *rng.standard_normal(4 * 2).tolist(),
    ]
    b_a = [
        *rng.standard_normal(4).tolist(),
        *rng.standard_normal(2).tolist(),
    ]
    pa = PolicyWeights(name="2-4-2", weights=w_a, biases=b_a,
                        layer_sizes=[2, 4, 2], timestamp="t")

    w_b = [
        *rng.standard_normal(2 * 4).tolist(),
        *rng.standard_normal(4 * 2).tolist(),
    ]
    b_b = [
        *rng.standard_normal(4).tolist(),
        *rng.standard_normal(2).tolist(),
    ]
    pb = PolicyWeights(name="2-4-2b", weights=w_b, biases=b_b,
                       layer_sizes=[2, 4, 2], timestamp="t")

    kl = kl_divergence_p(pa, pb, n_samples=100)
    assert isinstance(kl, float)
    assert kl >= 0.0

    layers = layer_wise_diff(pa, pb)
    assert len(layers) == 2
    assert layers[0]["n_params"] == 8   # 2*4
    assert layers[1]["n_params"] == 8   # 4*2


# --- 14. 1-layer network (3-1) ---

def test_one_layer_network():
    rng = np.random.default_rng(10)
    w_a = rng.standard_normal(3).tolist()
    b_a = rng.standard_normal(1).tolist()
    pa = PolicyWeights(name="3-1", weights=w_a, biases=b_a,
                       layer_sizes=[3, 1], timestamp="t")

    w_b = (rng.standard_normal(3) + 0.5).tolist()
    b_b = rng.standard_normal(1).tolist()
    pb = PolicyWeights(name="3-1b", weights=w_b, biases=b_b,
                       layer_sizes=[3, 1], timestamp="t")

    kl = kl_divergence_p(pa, pb, n_samples=100)
    assert isinstance(kl, float)

    d = policy_distance(pa, pb)
    assert d["max_weight_diff"] > 0.0


# --- 15. large weight difference detected ---

def test_large_weight_difference_detected():
    pa = _make_policy(seed=1)
    pb = _make_policy(seed=2)
    # make pb very different by adding a large constant
    pb.weights = [w + 100.0 for w in pb.weights]
    d = policy_distance(pa, pb)
    assert d["max_weight_diff"] > 50.0
    assert d["mean_weight_diff"] > 50.0
    assert d["relative_weight_diff"] > 1.0

    table = side_by_side_table(pa, pb, top_n=5)
    assert len(table) == 5
    assert all(abs(r["diff"]) > 50.0 for r in table)
