"""Tests for flopyarcade.transfer_heatmap (ROADMAP V6)."""

import math
import os
import tempfile

import pytest

from flopyarcade.transfer_heatmap import (
    TransferScore,
    TransferMatrix,
    compare_matrices,
    format_transfer_report,
    generate_demo_matrix,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class TestTransferScore:
    def test_creation(self):
        ts = TransferScore(
            source_env="env_a",
            target_env="env_b",
            score=0.75,
            p_value=0.003,
            n_trials=20,
        )
        assert ts.source_env == "env_a"
        assert ts.target_env == "env_b"
        assert ts.score == 0.75
        assert ts.p_value == 0.003
        assert ts.n_trials == 20

    def test_defaults(self):
        ts = TransferScore(source_env="a", target_env="b", score=0.5)
        assert ts.p_value is None
        assert ts.n_trials == 10


# ---------------------------------------------------------------------------
# Matrix basics
# ---------------------------------------------------------------------------

class TestTransferMatrixBasics:
    def test_init_with_3_envtypes(self):
        tm = TransferMatrix(["a", "b", "c"])
        assert tm.n_envtypes() == 3
        assert tm.envtypes == ["a", "b", "c"]

    def test_set_and_get_score(self):
        tm = TransferMatrix(["x", "y"])
        tm.set_score("x", "y", 0.65, p_value=0.02, n_trials=15)
        ts = tm.get_score("x", "y")
        assert ts.source_env == "x"
        assert ts.target_env == "y"
        assert ts.score == 0.65
        assert ts.p_value == 0.02
        assert ts.n_trials == 15

    def test_matrix_shape(self):
        tm = TransferMatrix(["a", "b", "c", "d"])
        m = tm.matrix()
        assert len(m) == 4
        assert all(len(row) == 4 for row in m)

    def test_p_matrix_shape(self):
        tm = TransferMatrix(["a", "b", "c"])
        pm = tm.p_matrix()
        assert len(pm) == 3
        assert all(len(row) == 3 for row in pm)
        # all NaN by default
        for row in pm:
            for v in row:
                assert math.isnan(v)

    def test_n_envtypes(self):
        tm = TransferMatrix(["e1", "e2", "e3", "e4", "e5", "e6"])
        assert tm.n_envtypes() == 6


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    @pytest.fixture
    def tm3(self):
        tm = TransferMatrix(["A", "B", "C"])
        tm.set_score("A", "A", 1.0)
        tm.set_score("A", "B", 0.8)
        tm.set_score("A", "C", 0.4)
        tm.set_score("B", "A", 0.6)
        tm.set_score("B", "B", 1.0)
        tm.set_score("B", "C", 0.3)
        tm.set_score("C", "A", 0.5)
        tm.set_score("C", "B", 0.7)
        tm.set_score("C", "C", 1.0)
        return tm

    def test_best_transfer_off_diagonal(self, tm3):
        best = tm3.best_transfer()
        assert best.source_env == "A"
        assert best.target_env == "B"
        assert best.score == 0.8

    def test_worst_transfer_off_diagonal(self, tm3):
        worst = tm3.worst_transfer()
        assert worst.source_env == "B"
        assert worst.target_env == "C"
        assert worst.score == 0.3

    def test_significance_stars(self):
        tm = TransferMatrix(["X", "Y", "Z"])
        tm.set_score("X", "Y", 0.5, p_value=0.0005)   # ***
        tm.set_score("X", "Z", 0.4, p_value=0.005)    # **
        tm.set_score("Y", "X", 0.3, p_value=0.03)     # *
        tm.set_score("Y", "Z", 0.2, p_value=0.10)     # ''
        tm.set_score("Z", "X", 0.6, p_value=None)     # ''
        tm.set_score("Z", "Y", 0.7, p_value=0.049)    # *
        stars = tm.significance_stars()
        assert stars[0][0] == "—"
        assert stars[0][1] == "***"
        assert stars[0][2] == "**"
        assert stars[1][0] == "*"
        assert stars[1][1] == "—"
        assert stars[1][2] == ""
        assert stars[2][0] == ""
        assert stars[2][1] == "*"
        assert stars[2][2] == "—"

    def test_row_best(self, tm3):
        rb = tm3.row_best("A")
        assert rb["target"] == "B"
        assert rb["score"] == 0.8

    def test_col_best(self, tm3):
        cb = tm3.col_best("A")
        assert cb["source"] == "B"
        assert cb["score"] == 0.6


# ---------------------------------------------------------------------------
# Export & serialisation
# ---------------------------------------------------------------------------

class TestExport:
    def test_heatmap_data_structure(self):
        tm = generate_demo_matrix(3, seed=1)
        hd = tm.heatmap_data()
        assert "labels" in hd
        assert "values" in hd
        assert "stars" in hd
        assert "vmin" in hd
        assert "vmax" in hd
        assert len(hd["labels"]) == 3
        assert len(hd["values"]) == 3
        assert len(hd["stars"]) == 3
        assert hd["vmin"] <= hd["vmax"]

    def test_save_load_roundtrip(self):
        tm = generate_demo_matrix(4, seed=99)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        tm.save_json(path)
        tm2 = TransferMatrix.load_json(path)
        assert tm2.envtypes == tm.envtypes
        assert tm2.matrix() == tm.matrix()
        # p_values: compare non-NaN positions
        for i in range(4):
            for j in range(4):
                p1 = tm.p_matrix()[i][j]
                p2 = tm2.p_matrix()[i][j]
                if math.isnan(p1):
                    assert math.isnan(p2)
                else:
                    assert p1 == p2
        os.unlink(path)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class TestStatTests:
    def test_t_test_significant(self):
        a = [10.0, 10.5, 9.8, 10.2, 10.1, 9.9, 10.3, 10.0]
        b = [1.0, 1.2, 0.9, 1.1, 1.05, 0.95, 1.0, 1.15]
        result = TransferMatrix.t_test(a, b)
        assert result["significant"] is True
        assert result["t_stat"] > 0
        assert result["df"] > 0
        assert result["p_value"] < 0.05

    def test_t_test_not_significant(self):
        a = [5.0, 5.1, 4.9, 5.0, 5.05]
        b = [5.0, 4.9, 5.1, 5.0, 4.95]
        result = TransferMatrix.t_test(a, b)
        assert result["significant"] is False

    def test_paired_t_test_significant(self):
        paired = [(10.0, 1.0), (10.5, 1.2), (9.8, 0.9), (10.2, 1.1), (10.1, 1.0)]
        result = TransferMatrix.paired_t_test(paired)
        assert result["significant"] is True
        assert result["df"] == 4

    def test_paired_t_test_insufficient_data(self):
        result = TransferMatrix.paired_t_test([(1.0, 2.0)])
        assert result["significant"] is False
        assert math.isnan(result["t_stat"])


# ---------------------------------------------------------------------------
# Demo generator
# ---------------------------------------------------------------------------

class TestDemoMatrix:
    def test_generate_6x6(self):
        tm = generate_demo_matrix(6, seed=42)
        assert tm.n_envtypes() == 6
        m = tm.matrix()
        assert len(m) == 6
        assert all(len(row) == 6 for row in m)

    def test_diagonal_is_one(self):
        tm = generate_demo_matrix(6, seed=42)
        m = tm.matrix()
        for i in range(6):
            assert m[i][i] == 1.0

    def test_off_diagonal_range(self):
        tm = generate_demo_matrix(6, seed=42)
        m = tm.matrix()
        for i in range(6):
            for j in range(6):
                if i != j:
                    assert 0.3 <= m[i][j] <= 0.8


# ---------------------------------------------------------------------------
# Compare & format
# ---------------------------------------------------------------------------

class TestCompareAndFormat:
    def test_compare_matrices_structure(self):
        m1 = generate_demo_matrix(4, seed=10)
        m2 = generate_demo_matrix(4, seed=20)
        result = compare_matrices(m1, m2)
        assert "diffs" in result
        assert "best_improved" in result
        assert "worst_degraded" in result
        assert len(result["diffs"]) == 4
        assert all(len(row) == 4 for row in result["diffs"])
        if result["best_improved"] is not None:
            s, t, d = result["best_improved"]
            assert isinstance(s, str)
            assert isinstance(d, float)

    def test_format_report_contains_stars(self):
        tm = TransferMatrix(["A", "B"])
        tm.set_score("A", "B", 0.7, p_value=0.0003)
        tm.set_score("B", "A", 0.5, p_value=0.03)
        report = format_transfer_report(tm)
        assert "***" in report
        assert "*" in report
        assert "A" in report
        assert "B" in report


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_env(self):
        tm = TransferMatrix(["solo"])
        assert tm.n_envtypes() == 1
        m = tm.matrix()
        assert len(m) == 1 and len(m[0]) == 1
        stars = tm.significance_stars()
        assert stars[0][0] == "—"
        # best/worst fallback to diagonal
        best = tm.best_transfer()
        assert best.source_env == "solo"
        assert best.target_env == "solo"
