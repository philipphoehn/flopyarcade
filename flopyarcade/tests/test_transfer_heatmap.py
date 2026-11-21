"""Tests for B21: Cross-env transfer heatmap."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from flopyarcade.transfer_heatmap import (
    TransferCell,
    build_transfer_matrix,
    find_best_transfers,
    find_worst_transfers,
    format_heatmap,
    load_matrix,
    save_matrix,
)


# ── 1. build_transfer_matrix 2x2 basic ──────────────────────────────────
def test_build_2x2_basic():
    results = {
        "env_a": {
            "env_a": [1.0, 2.0, 3.0],
            "env_b": [4.0, 5.0],
        },
        "env_b": {
            "env_a": [0.5, 1.5],
            "env_b": [2.0, 2.0],
        },
    }
    m = build_transfer_matrix(results)
    assert len(m) == 2
    assert len(m[0]) == 2
    # sorted train envs: env_a, env_b
    # sorted eval envs: env_a, env_b
    assert m[0][0].train_env == "env_a"
    assert m[0][0].eval_env == "env_a"
    assert m[0][0].score == pytest.approx(2.0)
    assert m[0][1].score == pytest.approx(4.5)
    assert m[1][0].score == pytest.approx(1.0)
    assert m[1][1].score == pytest.approx(2.0)


# ── 2. build_transfer_matrix 3x3 ────────────────────────────────────────
def test_build_3x3():
    results = {
        "a": {"a": [1.0], "b": [2.0], "c": [3.0]},
        "b": {"a": [4.0], "b": [5.0], "c": [6.0]},
        "c": {"a": [7.0], "b": [8.0], "c": [9.0]},
    }
    m = build_transfer_matrix(results)
    assert len(m) == 3
    for row in m:
        assert len(row) == 3
    assert m[2][2].score == 9.0


# ── 3. scores are means of provided lists ───────────────────────────────
def test_scores_are_means():
    results = {"x": {"y": [10.0, 20.0, 30.0, 40.0]}}
    m = build_transfer_matrix(results)
    assert m[0][0].score == pytest.approx(25.0)


# ── 4. p_value in [0, 1] ────────────────────────────────────────────────
def test_p_value_range():
    results = {"x": {"y": [0.1, -0.1, 0.2, -0.2]}}
    m = build_transfer_matrix(results)
    for row in m:
        for cell in row:
            assert 0.0 <= cell.p_value <= 1.0


# ── 5. significant flag correct ─────────────────────────────────────────
def test_significant_flag():
    # Large scores → very small p → significant
    results = {"x": {"y": [100.0, 101.0, 99.0]}}
    m = build_transfer_matrix(results)
    assert m[0][0].significant is True
    assert m[0][0].p_value < 0.05

    # Tiny scores near zero → p ≈ 1 → not significant
    results2 = {"x": {"y": [0.001, -0.001, 0.002]}}
    m2 = build_transfer_matrix(results2)
    assert m2[0][0].significant is False
    assert m2[0][0].p_value >= 0.05


# ── 6. find_best_transfers ordering ─────────────────────────────────────
def test_find_best_ordering():
    results = {
        "a": {"a": [1.0, 1.0], "b": [10.0, 10.0]},
        "b": {"a": [5.0, 5.0], "b": [2.0, 2.0]},
    }
    m = build_transfer_matrix(results)
    envs = ["a", "b"]
    best = find_best_transfers(m, envs, top_k=2)
    assert best[0] == ("a", "b", 10.0)
    assert best[1] == ("b", "a", 5.0)


# ── 7. find_worst_transfers ordering ────────────────────────────────────
def test_find_worst_ordering():
    results = {
        "a": {"a": [10.0, 10.0], "b": [1.0, 1.0]},
        "b": {"a": [5.0, 5.0], "b": [2.0, 2.0]},
    }
    m = build_transfer_matrix(results)
    envs = ["a", "b"]
    worst = find_worst_transfers(m, envs, top_k=2)
    assert worst[0] == ("a", "b", 1.0)
    assert worst[1] == ("b", "b", 2.0)


# ── 8. format_heatmap contains env names ────────────────────────────────
def test_format_heatmap_contains_env_names():
    results = {
        "env_one": {"env_one": [1.0, 2.0], "env_two": [3.0, 4.0]},
        "env_two": {"env_one": [5.0, 6.0], "env_two": [7.0, 8.0]},
    }
    m = build_transfer_matrix(results)
    envs = ["env_one", "env_two"]
    text = format_heatmap(m, envs)
    assert "env_one" in text
    assert "env_two" in text


# ── 9. save/load roundtrip ──────────────────────────────────────────────
def test_save_load_roundtrip():
    results = {"x": {"y": [1.0, 2.0, 3.0]}}
    m = build_transfer_matrix(results)
    envs = ["x", "y"]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        save_matrix(m, envs, path)
        m2, envs2 = load_matrix(path)
        assert envs2 == envs
        assert len(m2) == len(m)
        for i, row in enumerate(m):
            for j, cell in enumerate(row):
                assert m2[i][j].train_env == cell.train_env
                assert m2[i][j].eval_env == cell.eval_env
                assert m2[i][j].score == cell.score
                assert m2[i][j].p_value == cell.p_value
                assert m2[i][j].significant == cell.significant
    finally:
        path.unlink(missing_ok=True)


# ── 10. empty results returns empty matrix ──────────────────────────────
def test_empty_results():
    m = build_transfer_matrix({})
    assert m == []


# ── 11. single env single score ─────────────────────────────────────────
def test_single_env_single_score():
    results = {"solo": {"solo": [42.0]}}
    m = build_transfer_matrix(results)
    assert len(m) == 1
    assert len(m[0]) == 1
    assert m[0][0].score == 42.0
    # only 1 observation → p_value should be 1.0 (not significant)
    assert m[0][0].p_value == 1.0
    assert m[0][0].significant is False


# ── 12. diagonal cells (train=eval) handled ─────────────────────────────
def test_diagonal_cells():
    results = {
        "a": {"a": [5.0, 6.0], "b": [1.0, 2.0]},
        "b": {"a": [3.0, 4.0], "b": [7.0, 8.0]},
    }
    m = build_transfer_matrix(results)
    # diagonal: (a,a) and (b,b)
    assert m[0][0].train_env == "a"
    assert m[0][0].eval_env == "a"
    assert m[0][0].score == pytest.approx(5.5)
    assert m[1][1].train_env == "b"
    assert m[1][1].eval_env == "b"
    assert m[1][1].score == pytest.approx(7.5)


# ── Extra: format_heatmap has significance stars ────────────────────────
def test_format_heatmap_stars():
    # High scores → significant → should show stars
    results = {"x": {"x": [100.0, 200.0, 300.0]}}
    m = build_transfer_matrix(results)
    text = format_heatmap(m, ["x"])
    assert "*" in text  # at least one star


# ── Extra: find_best with top_k larger than matrix ──────────────────────
def test_best_top_k_exceeds_matrix():
    results = {"a": {"a": [1.0, 1.0]}}
    m = build_transfer_matrix(results)
    best = find_best_transfers(m, ["a"], top_k=10)
    assert len(best) == 1
