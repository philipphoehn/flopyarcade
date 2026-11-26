"""Tests for flopyarcade.surrogate_bayes_opt module."""

import json
import os
import tempfile

import numpy as np
import pytest

from flopyarcade.surrogate_bayes_opt import (
    BayesOptResult,
    SurrogateBayesOptimizer,
    expected_improvement,
    format_bayes_opt_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_search_space():
    return {
        "lr": (1e-5, 1e-2),
        "batch_size": (8, 256),
        "activation": ["relu", "tanh", "elu"],
    }


def sphere(params: dict) -> float:
    """Simple 2D sphere function for optimisation test."""
    return params["x"] ** 2 + params["y"] ** 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        """Test SurrogateBayesOptimizer init defaults."""
        ss = {"a": (0.0, 1.0)}
        bo = SurrogateBayesOptimizer(search_space=ss)
        assert bo.n_cheap_runs == 20
        assert bo.n_expensive_runs == 5
        assert bo.seed is None
        assert bo._history == []
        assert bo._iteration == 0

    def test_custom_params(self):
        ss = {"a": (0.0, 1.0), "b": ["x", "y"]}
        bo = SurrogateBayesOptimizer(search_space=ss, n_cheap_runs=10, n_expensive_runs=3, seed=42)
        assert bo.n_cheap_runs == 10
        assert bo.n_expensive_runs == 3
        assert bo.seed == 42


class TestSuggestParams:
    def test_returns_dict_with_correct_keys(self):
        """Test suggest_params returns dict with correct keys."""
        ss = {"a": (0.0, 1.0), "b": ["x", "y", "z"]}
        bo = SurrogateBayesOptimizer(search_space=ss, seed=42)
        params = bo.suggest_params()
        assert isinstance(params, dict)
        assert set(params.keys()) == {"a", "b"}
        assert 0.0 <= params["a"] <= 1.0
        assert params["b"] in ["x", "y", "z"]


class TestObserve:
    def test_observe_appends_to_history(self):
        """Test observe appends to history."""
        ss = {"a": (0.0, 1.0)}
        bo = SurrogateBayesOptimizer(search_space=ss, seed=42)
        assert len(bo.get_history()) == 0
        result = bo.observe({"a": 0.5}, 0.25)
        assert len(bo.get_history()) == 1
        assert bo.get_history()[0].objective == 0.25
        assert bo.get_history()[0].iteration == 1


class TestPhases:
    def test_run_cheap_phase_count(self):
        """Test run_cheap_phase returns correct count."""
        ss = {"a": (0.0, 1.0)}
        bo = SurrogateBayesOptimizer(search_space=ss, n_cheap_runs=5, seed=42)
        results = bo.run_cheap_phase(lambda p: p["a"] ** 2)
        assert len(results) == 5
        assert len(bo.get_history()) == 5

    def test_run_expensive_phase_count(self):
        """Test run_expensive_phase returns correct count."""
        ss = {"a": (0.0, 1.0)}
        bo = SurrogateBayesOptimizer(search_space=ss, n_cheap_runs=10, n_expensive_runs=3, seed=42)
        bo.run_cheap_phase(lambda p: p["a"] ** 2)
        results = bo.run_expensive_phase(lambda p: p["a"] ** 2)
        assert len(results) == 3
        assert len(bo.get_history()) == 13


class TestGetBest:
    def test_returns_best_objective(self):
        """Test get_best returns best objective."""
        ss = {"a": (0.0, 1.0)}
        bo = SurrogateBayesOptimizer(search_space=ss, seed=42)
        bo.observe({"a": 0.9}, 0.81)
        bo.observe({"a": 0.1}, 0.01)
        bo.observe({"a": 0.5}, 0.25)
        best = bo.get_best()
        assert best.objective == 0.01
        assert best.params["a"] == 0.1

    def test_get_history_length(self):
        ss = {"a": (0.0, 1.0)}
        bo = SurrogateBayesOptimizer(search_space=ss, seed=42)
        for i in range(7):
            bo.observe({"a": i / 7.0}, (i / 7.0) ** 2)
        assert len(bo.get_history()) == 7


class TestExpectedImprovement:
    def test_positive_when_below_best(self):
        """EI should be positive when mu is better (lower) than best_value."""
        ei = expected_improvement(mu=0.5, sigma=0.1, best_value=1.0)
        assert ei > 0.0

    def test_zero_or_negative_when_above_best(self):
        """EI should be ~0 when mu is worse (higher) than best_value."""
        ei = expected_improvement(mu=2.0, sigma=0.1, best_value=1.0)
        assert ei <= 0.0 or ei < 1e-6


class TestSearchSpace:
    def test_continuous_and_categorical(self):
        """Test search_space with continuous and categorical params."""
        ss = {"x": (0.0, 10.0), "choice": ["a", "b", "c"]}
        bo = SurrogateBayesOptimizer(search_space=ss, seed=42)
        p1 = bo.suggest_params()
        assert 0.0 <= p1["x"] <= 10.0
        assert p1["choice"] in ["a", "b", "c"]
        # Verify encode/decode roundtrip
        encoded = bo._encode(p1)
        decoded = bo._decode(encoded)
        assert abs(decoded["x"] - p1["x"]) < 1e-6
        assert decoded["choice"] == p1["choice"]


class TestSerialisation:
    def test_save_load_roundtrip(self):
        """Test save/load roundtrip."""
        ss = {"a": (0.0, 1.0), "b": ["x", "y"]}
        bo = SurrogateBayesOptimizer(search_space=ss, n_cheap_runs=10, n_expensive_runs=3, seed=42)
        bo.observe({"a": 0.5, "b": "x"}, 0.25)
        bo.observe({"a": 0.3, "b": "y"}, 0.09)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            bo.save_state(path)
            bo2 = SurrogateBayesOptimizer.load_state(path)
            assert bo2.n_cheap_runs == 10
            assert bo2.n_expensive_runs == 3
            assert bo2.seed == 42
            assert len(bo2.get_history()) == 2
            assert bo2.get_best().objective == 0.09
            # search_space should be restored
            assert isinstance(bo2.search_space["a"], tuple)
            assert bo2.search_space["b"] == ["x", "y"]
        finally:
            os.unlink(path)


class TestReport:
    def test_format_report_contains_iteration(self):
        """Test format_bayes_opt_report contains iteration info."""
        ss = {"a": (0.0, 1.0)}
        bo = SurrogateBayesOptimizer(search_space=ss, seed=42)
        bo.observe({"a": 0.5}, 0.25)
        report = format_bayes_opt_report(bo)
        assert "Iteration" in report
        assert "1" in report
        assert "Best result" in report
        assert "Bayesian Optimisation Report" in report


class TestDeterminism:
    def test_deterministic_with_fixed_seed(self):
        """Test deterministic with fixed seed."""
        ss = {"a": (0.0, 1.0)}
        bo1 = SurrogateBayesOptimizer(search_space=ss, seed=123)
        bo2 = SurrogateBayesOptimizer(search_space=ss, seed=123)
        p1 = bo1.suggest_params()
        p2 = bo2.suggest_params()
        assert p1 == p2


class TestOptimisation:
    def test_2d_sphere_finds_reasonable_minimum(self):
        """Test 2D search space optimization (sphere) finds reasonable minimum."""
        ss = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
        bo = SurrogateBayesOptimizer(
            search_space=ss, n_cheap_runs=20, n_expensive_runs=5, seed=42
        )
        bo.run_cheap_phase(sphere)
        bo.run_expensive_phase(sphere)
        best = bo.get_best()
        # Should find a value much better than random expectation (12.5 for uniform [-5,5])
        assert best.objective < 12.5, f"Best objective {best.objective} not better than expected random"
        # Should be reasonably close to 0
        assert best.objective < 50.0
