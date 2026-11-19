#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade - Multi-Objective NSGA-II
# R4: Pareto front on (reward, cost, wall-clock); archive to HDF5

"""NSGA-II multi-objective optimisation for FloPyArcade.

Optimises three objectives simultaneously:
  - Maximise reward (negated for minimisation)
  - Minimise cost (sum of |q_wells| over episode)
  - Minimise wall-clock time (seconds per episode)

The Pareto front is archived to HDF5 for post-hoc analysis.

Usage
-----
    from flopyarcade.nsga2 import NSGA2, NSGA2Config

    cfg = NSGA2Config(popsize=100, maxgen=50)
    opt = NSGA2(cfg, objective_fn=my_evaluation_fn)
    front = opt.evolve()
    front.save_hdf5("pareto_front.h5")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# --- Configuration ---

@dataclass
class NSGA2Config:
    """Configuration for NSGA-II optimiser."""
    popsize: int = 100
    maxgen: int = 50
    n_objectives: int = 3  # reward, cost, wall_clock
    n_params: int = 8     # number of decision variables
    param_bounds: list = field(default_factory=lambda: [(0.0, 1.0)] * 8)
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    crossover_eta: float = 20.0  # SBX distribution index
    mutation_eta: float = 20.0   # polynomial mutation distribution index
    seed: int = 42
    archive_path: Optional[str] = None
    log_interval: int = 10  # print status every N generations


# --- Core NSGA-II ---

class NSGA2:
    """NSGA-II multi-objective evolutionary algorithm.

    Parameters
    ----------
    config : NSGA2Config
        Algorithm configuration.
    objective_fn : callable
        Function taking a parameter vector (np.ndarray of shape (n_params,))
        and returning a list/array of objective values to MINIMISE.
        For reward (to maximise), negate it in the objective function.
    """

    def __init__(self, config: NSGA2Config, objective_fn: Callable):
        self.config = config
        self.objective_fn = objective_fn
        self.rng = np.random.default_rng(config.seed)
        self._population: Optional[np.ndarray] = None
        self._fitness: Optional[np.ndarray] = None
        self._fronts: list[np.ndarray] = []
        self._generation_history: list[dict] = []

    # --- Initialisation ---

    def _random_individual(self) -> np.ndarray:
        """Generate a random individual within bounds."""
        x = np.zeros(self.config.n_params)
        for i in range(self.config.n_params):
            lo, hi = self.config.param_bounds[i % len(self.config.param_bounds)]
            x[i] = self.rng.uniform(lo, hi)
        return x

    def _init_population(self) -> None:
        """Create initial random population and evaluate."""
        self._population = np.array([self._random_individual() for _ in range(self.config.popsize)])
        self._fitness = np.array([self._evaluate(ind) for ind in self._population])

    def _evaluate(self, individual: np.ndarray) -> list:
        """Evaluate objectives for a single individual."""
        return np.array(self.objective_fn(individual), dtype=np.float64)

    # --- Non-dominated sorting ---

    def _fast_non_dominated_sort(self, fitness: np.ndarray) -> tuple:
        """Fast non-dominated sorting (NSGA-II).

        Returns
        -------
        (fronts, dominance_count)
        fronts : list of arrays of indices
        dominance_count : array of domination counts
        """
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [set() for _ in range(n)]
        fronts = [set() for _ in range(n)]  # will trim later

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fitness[i], fitness[j]):
                    dominated_set[i].add(j)
                    domination_count[j] += 1
                elif self._dominates(fitness[j], fitness[i]):
                    dominated_set[j].add(i)
                    domination_count[i] += 1

        # Assign to fronts
        current_front = set()
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].add(i)

        k = 0
        while k < n and len(fronts[k]) > 0:
            k += 1
            if k >= n:
                break
            for i in fronts[k - 1]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        fronts[k].add(j)

        # Trim empty fronts
        fronts = [list(f) for f in fronts if len(f) > 0]
        return fronts, domination_count

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """Check if a dominates b (all objectives <=, at least one <)."""
        return bool(np.all(a <= b) and np.any(a < b))

    # --- Crowding distance ---

    def _crowding_distance(self, fitness: np.ndarray, front: list) -> np.ndarray:
        """Compute crowding distance for members of a front.

        Parameters
        ----------
        fitness : (N, M) array for the full population.
        front : list of global indices into fitness.

        Returns
        -------
        dist : array of length len(front), indexed by local position.
        """
        n = len(front)
        dist = np.zeros(n)
        if n <= 2:
            dist[:] = np.inf
            return dist

        n_obj = fitness.shape[1]
        for m in range(n_obj):
            # Sort local positions by objective m
            sorted_pos = sorted(range(n), key=lambda p: fitness[front[p], m])
            dist[sorted_pos[0]] = np.inf
            dist[sorted_pos[-1]] = np.inf
            f_min = fitness[front[sorted_pos[0]], m]
            f_max = fitness[front[sorted_pos[-1]], m]
            if f_max - f_min < 1e-30:
                continue
            for k in range(1, n - 1):
                p = sorted_pos[k]
                p_prev = sorted_pos[k - 1]
                p_next = sorted_pos[k + 1]
                dist[p] += (
                    fitness[front[p_next], m] - fitness[front[p_prev], m]
                ) / (f_max - f_min)
        return dist

    # --- Selection ---

    def _tournament_select(self, parent_indices: np.ndarray, front_idx: np.ndarray,
                           crowding: np.ndarray, n_select: int) -> np.ndarray:
        """Tournament selection based on rank and crowding distance."""
        selected = []
        for _ in range(n_select):
            i = int(self.rng.integers(len(parent_indices)))
            j = int(self.rng.integers(len(parent_indices)))
            fi = front_idx[parent_indices[i]]
            fj = front_idx[parent_indices[j]]
            if fi < fj:
                selected.append(parent_indices[i])
            elif fi > fj:
                selected.append(parent_indices[j])
            else:
                if crowding[parent_indices[i]] >= crowding[parent_indices[j]]:
                    selected.append(parent_indices[i])
                else:
                    selected.append(parent_indices[j])
        return np.array(selected, dtype=int)

    # --- Crossover and mutation ---

    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple:
        """Simulated Binary Crossover (SBX)."""
        if self.rng.random() > self.config.crossover_prob:
            return p1.copy(), p2.copy()

        child1 = p1.copy().astype(float)
        child2 = p2.copy().astype(float)

        for i in range(len(p1)):
            if self.rng.random() < 0.5:
                lo, hi = self.config.param_bounds[i]
                u = self.rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (self.config.crossover_eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (self.config.crossover_eta + 1))
                c1 = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2 = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                child1[i] = np.clip(c1, lo, hi)
                child2[i] = np.clip(c2, lo, hi)

        return child1, child2

    def _polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        x = x.copy().astype(float)
        for i in range(len(x)):
            if self.rng.random() < self.config.mutation_prob:
                lo, hi = self.config.param_bounds[i]
                if hi - lo < 1e-30:
                    continue
                y = np.clip(x[i], lo, hi)
                u = self.rng.random()
                delta1 = (y - lo) / (hi - lo)
                delta2 = (hi - y) / (hi - lo)
                if u < 0.5:
                    mut = 2 * u
                    delta = mut ** (1 / (self.config.mutation_eta + 1)) - 1
                else:
                    mut = 2 * (1 - u)
                    delta = 1 - mut ** (1 / (self.config.mutation_eta + 1))
                new_val = y + delta * (hi - lo)
                x[i] = np.clip(new_val, lo, hi)
        return x

    # --- Main evolution loop ---

    def evolve(self) -> "ParetoFront":
        """Run NSGA-II evolution. Returns the final Pareto front."""
        cfg = self.config
        self._init_population()

        for gen in range(cfg.maxgen):
            # Fast non-dominated sort
            fronts, _ = self._fast_non_dominated_sort(self._fitness)

            # Assign front indices and crowding distances
            n = cfg.popsize
            front_idx = np.zeros(n, dtype=int)
            crowding = np.zeros(n)

            for rank, front in enumerate(fronts):
                for idx in front:
                    front_idx[idx] = rank
                cd = self._crowding_distance(self._fitness, front)
                for pos, idx in enumerate(front):
                    crowding[idx] = cd[pos]

            # Create offspring
            offspring = []
            while len(offspring) < cfg.popsize:
                p1 = int(self.rng.integers(n))
                p2 = int(self.rng.integers(n))
                c1, c2 = self._sbx_crossover(self._population[p1], self._population[p2])
                c1 = self._polynomial_mutation(c1)
                c2 = self._polynomial_mutation(c2)
                offspring.append(c1)
                if len(offspring) < cfg.popsize:
                    offspring.append(c2)

            offspring = np.array(offspring[:cfg.popsize])
            offspring_fitness = np.array([self._evaluate(ind) for ind in offspring])

            # Combine parent + offspring (2N)
            combined_pop = np.vstack([self._population, offspring])
            combined_fit = np.vstack([self._fitness, offspring_fitness])

            # Sort combined and select top N
            c_fronts, _ = self._fast_non_dominated_sort(combined_fit)
            c_n = len(combined_pop)
            c_front_idx = np.zeros(c_n, dtype=int)
            c_crowding = np.zeros(c_n)

            for rank, front in enumerate(c_fronts):
                for idx in front:
                    c_front_idx[idx] = rank
                cd = self._crowding_distance(combined_fit, front)
                for pos, idx in enumerate(front):
                    c_crowding[idx] = cd[pos]

            # Select top N by rank then crowding
            selected_indices = []
            for rank in range(len(c_fronts)):
                front = c_fronts[rank]
                if len(selected_indices) + len(front) <= cfg.popsize:
                    selected_indices.extend(front)
                else:
                    # Sort by crowding distance (descending) and take what fits
                    remaining = cfg.popsize - len(selected_indices)
                    sorted_front = sorted(front, key=lambda i: c_crowding[i], reverse=True)
                    selected_indices.extend(sorted_front[:remaining])
                    break

            self._population = combined_pop[selected_indices[:cfg.popsize]]
            self._fitness = combined_fit[selected_indices[:cfg.popsize]]

            # Log
            if (gen + 1) % cfg.log_interval == 0 or gen == 0:
                best_reward = -self._fitness[:, 0].min()  # reward was negated
                self._generation_history.append({
                    "generation": gen + 1,
                    "best_reward": float(best_reward),
                    "n_front1": len(c_fronts[0]),
                })
                print(f"  [NSGA2] gen {gen+1}/{cfg.maxgen}: best_reward={best_reward:.4f}, |F1|={len(c_fronts[0])}")

        # Extract final Pareto front
        front_indices, _ = self._fast_non_dominated_sort(self._fitness)
        front_indices = front_indices[0]  # first front

        front = ParetoFront(
            individuals=self._population[front_indices],
            objectives=self._fitness[front_indices],
            generation=cfg.maxgen,
            config=cfg,
        )

        # Save to HDF5 if path specified
        if cfg.archive_path:
            front.save_hdf5(cfg.archive_path)

        return front


# --- Pareto Front ---

@dataclass
class ParetoFront:
    """Container for the non-dominated Pareto front."""
    individuals: np.ndarray  # shape (n_front, n_params)
    objectives: np.ndarray   # shape (n_front, n_objectives)
    generation: int
    config: NSGA2Config

    @property
    def n_solutions(self) -> int:
        return len(self.individuals)

    def get_best(self, objective_idx: int = 0) -> tuple:
        """Get the individual with the best value for a given objective."""
        idx = int(np.argmin(self.objectives[:, objective_idx]))
        return self.individuals[idx], self.objectives[idx]

    def save_hdf5(self, path: str) -> Path:
        """Archive the Pareto front to an HDF5 file."""
        import h5py

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(p), 'w') as f:
            f.create_dataset('individuals', data=self.individuals)
            f.create_dataset('objectives', data=self.objectives)
            f.attrs['n_solutions'] = self.n_solutions
            f.attrs['generation'] = self.generation
            f.attrs['n_params'] = self.config.n_params
            f.attrs['n_objectives'] = self.config.n_objectives
            f.attrs['popsize'] = self.config.popsize
            f.attrs['maxgen'] = self.config.maxgen
            f.attrs['param_bounds'] = json.dumps(self.config.param_bounds)

        return p

    @classmethod
    def load_hdf5(cls, path: str) -> "ParetoFront":
        """Load a Pareto front from HDF5."""
        import h5py

        with h5py.File(str(path), 'r') as f:
            individuals = f['individuals'][:]
            objectives = f['objectives'][:]
            generation = int(f.attrs['generation'])
            n_params = int(f.attrs['n_params'])
            n_objectives = int(f.attrs['n_objectives'])
            popsize = int(f.attrs['popsize'])
            maxgen = int(f.attrs['maxgen'])
            param_bounds = json.loads(f.attrs['param_bounds'])

        cfg = NSGA2Config(
            popsize=popsize,
            maxgen=maxgen,
            n_objectives=n_objectives,
            n_params=n_params,
            param_bounds=param_bounds,
        )
        return cls(
            individuals=individuals,
            objectives=objectives,
            generation=generation,
            config=cfg,
        )


# --- Convenience ---

def make_nsga2(
    n_params: int = 8,
    param_bounds: Optional[list] = None,
    popsize: int = 100,
    maxgen: int = 50,
    seed: int = 42,
) -> NSGA2Config:
    """Create an NSGA2Config with sensible defaults."""
    return NSGA2Config(
        popsize=popsize,
        maxgen=maxgen,
        n_params=n_params,
        param_bounds=param_bounds or [(0.0, 1.0)] * n_params,
        seed=seed,
    )


# --- CLI smoke test ---

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="FloPyArcade NSGA-II smoke test")
    parser.add_argument('--popsize', type=int, default=20)
    parser.add_argument('--maxgen', type=int, default=5)
    parser.add_argument('--n-params', type=int, default=4)
    args = parser.parse_args()

    print(f"NSGA-II smoke test: popsize={args.popsize}, maxgen={args.maxgen}, n_params={args.n_params}")

    # Synthetic multi-objective function (Sphere + Knapsack-like)
    def synthetic_objective(x: np.ndarray) -> list:
        f1 = sum((xi - 0.5) ** 2 for xi in x)  # minimise
        f2 = sum(abs(xi) for xi in x)          # minimise
        f3 = x[0] * x[-1] if len(x) > 1 else 0  # minimise
        return [f1, f2, f3]

    cfg = NSGA2Config(
        popsize=args.popsize,
        maxgen=args.maxgen,
        n_params=args.n_params,
        param_bounds=[(0.0, 1.0)] * args.n_params,
        seed=42,
    )

    opt = NSGA2(cfg, synthetic_objective)
    front = opt.evolve()

    print(f"\nPareto front size: {front.n_solutions}")
    print(f"Best F1 (reward proxy): {-front.objectives[0, 0]:.4f}")

    # Test HDF5 round-trip
    test_path = '/tmp/_nsga2_test.h5'
    front.save_hdf5(test_path)
    loaded = ParetoFront.load_hdf5(test_path)
    assert loaded.n_solutions == front.n_solutions
    assert np.allclose(loaded.objectives, front.objectives)
    print(f"HDF5 round-trip OK: {loaded.n_solutions} solutions")

    import os
    os.unlink(test_path)
    print("\nSmoke test passed.")
