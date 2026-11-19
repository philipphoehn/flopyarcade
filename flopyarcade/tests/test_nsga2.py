#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade - NSGA-II tests
# R4: Multi-objective NSGA-II on (reward, cost, wall-clock); archive to HDF5

"""Tests for flopyarcade.nsga2 module."""

import os
import tempfile

import numpy as np
import pytest

from flopyarcade.nsga2 import NSGA2, NSGA2Config, ParetoFront, make_nsga2


# --- Synthetic objective for testing ---

def synthetic_obj(x: np.ndarray) -> list:
    """3-objective synthetic function for testing."""
    f1 = sum((xi - 0.5) ** 2 for xi in x)
    f2 = sum(abs(xi) for xi in x)
    f3 = x[0] * x[-1] if len(x) > 1 else 0.0
    return [f1, f2, f3]


def bi_obj(x: np.ndarray) -> list:
    """2-objective for simpler tests."""
    f1 = sum((xi - 0.3) ** 2 for xi in x)
    f2 = sum((xi - 0.7) ** 2 for xi in x)
    return [f1, f2]


# --- Config tests ---

class TestConfig:
    def test_defaults(self):
        cfg = NSGA2Config()
        assert cfg.popsize == 100
        assert cfg.maxgen == 50
        assert cfg.n_objectives == 3
        assert cfg.n_params == 8
        assert cfg.crossover_prob == 0.9
        assert cfg.mutation_prob == 0.1
        assert cfg.seed == 42

    def test_custom_params(self):
        cfg = NSGA2Config(popsize=20, maxgen=5, n_params=4)
        assert cfg.popsize == 20
        assert cfg.maxgen == 5
        assert cfg.n_params == 4
        assert len(cfg.param_bounds) == 8  # default_factory

    def test_make_nsga2(self):
        cfg = make_nsga2(n_params=4, popsize=30, maxgen=10)
        assert cfg.n_params == 4
        assert cfg.popsize == 30
        assert cfg.maxgen == 10

    def test_make_nsga2_custom_bounds(self):
        bounds = [(0, 1), (5, 10), (-1, 1)]
        cfg = make_nsga2(n_params=3, param_bounds=bounds)
        assert cfg.param_bounds == bounds


# --- Dominance tests ---

class TestDominance:
    def test_dominates_true(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 3.0])
        assert NSGA2._dominates(a, b) is True

    def test_dominates_false_equal(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        assert NSGA2._dominates(a, b) is False

    def test_dominates_false_cross(self):
        a = np.array([1.0, 3.0])
        b = np.array([2.0, 2.0])
        assert NSGA2._dominates(a, b) is False
        assert NSGA2._dominates(b, a) is False

    def test_dominates_strict_one(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        assert NSGA2._dominates(a, b) is True
        assert NSGA2._dominates(b, a) is False


# --- Non-dominated sort tests ---

class TestNonDominatedSort:
    def test_single_front(self):
        """All non-dominated → one front."""
        opt = NSGA2(NSGA2Config(popsize=4, n_params=2), synthetic_obj)
        fit = np.array([
            [1.0, 3.0, 0.5],
            [2.0, 2.0, 0.5],
            [3.0, 1.0, 0.5],
            [2.5, 2.5, 0.3],
        ])
        fronts, _ = opt._fast_non_dominated_sort(fit)
        assert len(fronts) == 1
        assert len(fronts[0]) == 4

    def test_two_fronts(self):
        """Clear dominated and non-dominated."""
        opt = NSGA2(NSGA2Config(popsize=4, n_params=2), synthetic_obj)
        fit = np.array([
            [1.0, 1.0, 0.5],  # non-dominated
            [2.0, 2.0, 0.5],  # dominated by above
            [3.0, 3.0, 0.5],  # dominated
            [0.5, 2.0, 0.5],  # non-dominated
        ])
        fronts, _ = opt._fast_non_dominated_sort(fit)
        assert len(fronts) >= 2
        assert 0 in fronts[0]
        assert 3 in fronts[0]

    def test_all_dominated_chain(self):
        """Chain: each dominates the next."""
        opt = NSGA2(NSGA2Config(popsize=3, n_params=2), synthetic_obj)
        fit = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ])
        fronts, _ = opt._fast_non_dominated_sort(fit)
        assert len(fronts) == 3
        assert len(fronts[0]) == 1


# --- Crowding distance tests ---

class TestCrowdingDistance:
    def test_two_members_infinite(self):
        opt = NSGA2(NSGA2Config(popsize=2, n_params=2), synthetic_obj)
        fit = np.array([[1.0, 2.0], [2.0, 1.0]])
        dist = opt._crowding_distance(fit, [0, 1])
        assert np.all(np.isinf(dist))

    def test_single_member_infinite(self):
        opt = NSGA2(NSGA2Config(popsize=1, n_params=2), synthetic_obj)
        fit = np.array([[1.0, 2.0]])
        dist = opt._crowding_distance(fit, [0])
        assert np.isinf(dist[0])

    def test_three_members(self):
        opt = NSGA2(NSGA2Config(popsize=3, n_params=2), synthetic_obj)
        fit = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        dist = opt._crowding_distance(fit, [0, 1, 2])
        # End members get infinity
        assert np.isinf(dist[0])
        assert np.isinf(dist[2])
        # Middle gets finite positive value
        assert dist[1] > 0

    def test_uniform_spacing(self):
        """Evenly spaced points get equal crowding."""
        opt = NSGA2(NSGA2Config(popsize=5, n_params=2), synthetic_obj)
        fit = np.linspace(0, 1, 5).reshape(-1, 1)
        dist = opt._crowding_distance(fit, [0, 1, 2, 3, 4])
        # Interior points should have similar distances
        assert abs(dist[1] - dist[2]) < 1e-10
        assert abs(dist[2] - dist[3]) < 1e-10


# --- Crossover tests ---

class TestCrossover:
    def test_returns_two_children(self):
        opt = NSGA2(NSGA2Config(popsize=4, n_params=4), synthetic_obj)
        p1 = np.array([0.2, 0.4, 0.6, 0.8])
        p2 = np.array([0.8, 0.6, 0.4, 0.2])
        c1, c2 = opt._sbx_crossover(p1, p2)
        assert c1.shape == (4,)
        assert c2.shape == (4,)

    def test_children_within_bounds(self):
        opt = NSGA2(NSGA2Config(popsize=4, n_params=3, param_bounds=[(0, 1), (0, 1), (0, 1)]), synthetic_obj)
        rng = np.random.default_rng(0)
        for _ in range(50):
            p1 = rng.uniform(0, 1, 3)
            p2 = rng.uniform(0, 1, 3)
            c1, c2 = opt._sbx_crossover(p1, p2)
            assert np.all(c1 >= 0) and np.all(c1 <= 1)
            assert np.all(c2 >= 0) and np.all(c2 <= 1)

    def test_different_bounds(self):
        cfg = NSGA2Config(popsize=4, n_params=2, param_bounds=[(-5, 5), (0, 10)])
        opt = NSGA2(cfg, synthetic_obj)
        p1 = np.array([-3.0, 2.0])
        p2 = np.array([4.0, 8.0])
        c1, c2 = opt._sbx_crossover(p1, p2)
        assert -5 <= c1[0] <= 5
        assert 0 <= c1[1] <= 10
        assert -5 <= c2[0] <= 5
        assert 0 <= c2[1] <= 10


# --- Mutation tests ---

class TestMutation:
    def test_returns_same_shape(self):
        opt = NSGA2(NSGA2Config(popsize=4, n_params=3), synthetic_obj)
        x = np.array([0.3, 0.5, 0.7])
        mutated = opt._polynomial_mutation(x)
        assert mutated.shape == (3,)

    def test_within_bounds(self):
        cfg = NSGA2Config(popsize=4, n_params=3, param_bounds=[(0, 1), (0, 1), (0, 1)])
        opt = NSGA2(cfg, synthetic_obj)
        rng = np.random.default_rng(0)
        for _ in range(50):
            x = rng.uniform(0, 1, 3)
            m = opt._polynomial_mutation(x)
            assert np.all(m >= 0) and np.all(m <= 1)

    def test_no_mutation_at_zero_prob(self):
        cfg = NSGA2Config(popsize=4, n_params=3, mutation_prob=0.0)
        opt = NSGA2(cfg, synthetic_obj)
        x = np.array([0.3, 0.5, 0.7])
        m = opt._polynomial_mutation(x)
        assert np.allclose(m, x)


# --- Evolution loop tests ---

class TestEvolve:
    def test_evolve_returns_front(self):
        cfg = NSGA2Config(popsize=10, maxgen=3, n_params=3)
        opt = NSGA2(cfg, synthetic_obj)
        front = opt.evolve()
        assert isinstance(front, ParetoFront)
        assert front.n_solutions >= 1

    def test_front_shape(self):
        cfg = NSGA2Config(popsize=10, maxgen=2, n_params=3)
        opt = NSGA2(cfg, synthetic_obj)
        front = opt.evolve()
        assert front.individuals.shape[1] == 3
        assert front.objectives.shape[1] == 3

    def test_front_within_bounds(self):
        cfg = NSGA2Config(popsize=10, maxgen=3, n_params=3, param_bounds=[(0, 1)] * 3)
        opt = NSGA2(cfg, synthetic_obj)
        front = opt.evolve()
        assert np.all(front.individuals >= -1e-10)
        assert np.all(front.individuals <= 1 + 1e-10)

    def test_get_best(self):
        cfg = NSGA2Config(popsize=10, maxgen=3, n_params=3)
        opt = NSGA2(cfg, synthetic_obj)
        front = opt.evolve()
        best_ind, best_obj = front.get_best(objective_idx=0)
        assert best_obj[0] == front.objectives[:, 0].min()

    def test_two_objective(self):
        cfg = NSGA2Config(popsize=8, maxgen=3, n_params=2, n_objectives=2)
        opt = NSGA2(cfg, bi_obj)
        front = opt.evolve()
        assert front.objectives.shape[1] == 2

    def test_reproducible_with_seed(self):
        cfg1 = NSGA2Config(popsize=10, maxgen=3, n_params=3, seed=123)
        cfg2 = NSGA2Config(popsize=10, maxgen=3, n_params=3, seed=123)
        f1 = NSGA2(cfg1, synthetic_obj).evolve()
        f2 = NSGA2(cfg2, synthetic_obj).evolve()
        assert np.allclose(f1.objectives, f2.objectives)

    def test_generation_history(self):
        cfg = NSGA2Config(popsize=10, maxgen=5, n_params=3, log_interval=1)
        opt = NSGA2(cfg, synthetic_obj)
        opt.evolve()
        assert len(opt._generation_history) == 5


# --- HDF5 archive tests ---

class TestHDF5:
    def test_save_and_load(self):
        cfg = NSGA2Config(popsize=8, maxgen=2, n_params=3)
        front = NSGA2(cfg, synthetic_obj).evolve()

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        front.save_hdf5(path)
        assert os.path.exists(path)

        loaded = ParetoFront.load_hdf5(path)
        assert loaded.n_solutions == front.n_solutions
        assert np.allclose(loaded.individuals, front.individuals)
        assert np.allclose(loaded.objectives, front.objectives)
        assert loaded.generation == front.generation

        os.unlink(path)

    def test_save_creates_dirs(self):
        cfg = NSGA2Config(popsize=8, maxgen=2, n_params=3)
        front = NSGA2(cfg, synthetic_obj).evolve()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'sub', 'dir', 'front.h5')
            front.save_hdf5(path)
            assert os.path.exists(path)

    def test_archive_path_in_config(self):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        cfg = NSGA2Config(popsize=8, maxgen=2, n_params=3, archive_path=path)
        front = NSGA2(cfg, synthetic_obj).evolve()
        assert os.path.exists(path)

        os.unlink(path)

    def test_attrs_roundtrip(self):
        cfg = NSGA2Config(popsize=12, maxgen=4, n_params=5)
        front = NSGA2(cfg, synthetic_obj).evolve()

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        front.save_hdf5(path)
        loaded = ParetoFront.load_hdf5(path)
        assert loaded.config.popsize == 12
        assert loaded.config.maxgen == 4
        assert loaded.config.n_params == 5

        os.unlink(path)


# --- Edge cases ---

class TestEdgeCases:
    def test_single_param(self):
        def single_obj(x):
            return [x[0] ** 2, abs(x[0])]

        cfg = NSGA2Config(popsize=6, maxgen=2, n_params=1, param_bounds=[(-1, 1)])
        front = NSGA2(cfg, single_obj).evolve()
        assert front.n_solutions >= 1

    def test_tight_bounds(self):
        cfg = NSGA2Config(popsize=6, maxgen=2, n_params=2, param_bounds=[(0.5, 0.5), (0, 1)])
        opt = NSGA2(cfg, synthetic_obj)
        m = opt._polynomial_mutation(np.array([0.5, 0.5]))
        assert m[0] == 0.5  # zero-width bound stays fixed

    def test_zero_objective(self):
        def zero_obj(x):
            return [0.0, 0.0, 0.0]

        cfg = NSGA2Config(popsize=6, maxgen=2, n_params=2)
        front = NSGA2(cfg, zero_obj).evolve()
        assert np.all(front.objectives == 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
