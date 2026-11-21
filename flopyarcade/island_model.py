"""Neuroevolution island model (ROADMAP R9).

4 islands with top-5 migration every 50 generations and a
diversity metric for collapse detection.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class Island:
    """A single island in the neuroevolution model."""

    id: int
    population: List[List[float]] = field(default_factory=list)
    best_fitness: float = float("-inf")
    generation: int = 0


class IslandModel:
    """Neuroevolution island model with migration and diversity tracking."""

    def __init__(
        self,
        n_islands: int = 4,
        population_size: int = 50,
        migrate_top_k: int = 5,
        migrate_every: int = 50,
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        if n_islands < 1:
            raise ValueError("n_islands must be >= 1")
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if migrate_top_k < 1:
            raise ValueError("migrate_top_k must be >= 1")

        self.n_islands = n_islands
        self.population_size = population_size
        self.migrate_top_k = migrate_top_k
        self.migrate_every = migrate_every
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.rng = random.Random(seed)
        self.islands: List[Island] = []
        self.fitness_fn: Optional[Callable[[List[float]], float]] = None
        self._total_generation = 0

    def initialize(self, fitness_fn: Callable[[List[float]], float]) -> None:
        """Create all islands with random populations."""
        self.fitness_fn = fitness_fn
        self.islands = []
        dim = self._infer_dim(fitness_fn)
        for i in range(self.n_islands):
            pop = [[self.rng.gauss(0.0, 1.0) for _ in range(dim)]
                   for _ in range(self.population_size)]
            self.islands.append(Island(
                id=i, population=pop,
                best_fitness=float("-inf"), generation=0,
            ))
        self._total_generation = 0

    @staticmethod
    def _infer_dim(fitness_fn: Callable[[List[float]], float]) -> int:
        for dim in (10, 20, 50, 5, 3):
            try:
                fitness_fn([0.0] * dim)
                return dim
            except (ValueError, TypeError, IndexError):
                continue
        return 10

    def evaluate(self, island_id: int, fitness_fn: Optional[Callable] = None) -> None:
        fn = fitness_fn or self.fitness_fn
        if fn is None:
            raise ValueError("No fitness function provided.")
        island = self.islands[island_id]
        best = float("-inf")
        for weights in island.population:
            fit = fn(weights)
            if fit > best:
                best = fit
        island.best_fitness = best

    def select_best(self, island_id: int) -> List[float]:
        island = self.islands[island_id]
        if not island.population:
            raise ValueError(f"Island {island_id} has empty population")
        fn = self.fitness_fn
        if fn is None:
            raise ValueError("No fitness function.")
        best_idx, best_fit = 0, float("-inf")
        for idx, weights in enumerate(island.population):
            fit = fn(weights)
            if fit > best_fit:
                best_fit, best_idx = fit, idx
        return list(island.population[best_idx])

    def _mutate(self, weights: List[float]) -> List[float]:
        if self.mutation_rate <= 0:
            return list(weights)
        return [
            w + self.rng.gauss(0.0, 0.5)
            if self.rng.random() < self.mutation_rate else w
            for w in weights
        ]

    def migrate(self) -> List[dict]:
        """Move top-k best individuals from each island to every other island."""
        events: List[dict] = []
        top_k_by_island: dict = {}
        for island in self.islands:
            if not island.population:
                top_k_by_island[island.id] = []
                continue
            fn = self.fitness_fn
            if fn is None:
                raise ValueError("No fitness function for migration.")
            scored = [(fn(w), list(w)) for w in island.population]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_k_by_island[island.id] = scored[:self.migrate_top_k]

        for src_id, src_top in top_k_by_island.items():
            if not src_top:
                continue
            for dst_id in top_k_by_island:
                if dst_id == src_id:
                    continue
                dst_island = self.islands[dst_id]
                for _, weights in src_top:
                    migrant = list(weights)
                    if len(dst_island.population) >= self.population_size:
                        worst_idx = self._find_worst(dst_island)
                        dst_island.population[worst_idx] = migrant
                    else:
                        dst_island.population.append(migrant)
                    dst_island.best_fitness = float("-inf")
                events.append({
                    "from_island": src_id,
                    "to_island": dst_id,
                    "count": len(src_top),
                })
        return events

    def _find_worst(self, island: Island) -> int:
        fn = self.fitness_fn
        if fn is None:
            return len(island.population) - 1
        worst_idx, worst_fit = 0, float("inf")
        for idx, w in enumerate(island.population):
            fit = fn(w)
            if fit < worst_fit:
                worst_fit, worst_idx = fit, idx
        return worst_idx

    def step(
        self,
        fitness_fn: Optional[Callable] = None,
        n_generations: int = 1,
    ) -> None:
        """Run n_generations of evaluate+select+mutate, migrating every migrate_every gens."""
        fn = fitness_fn or self.fitness_fn
        if fn is None:
            raise ValueError("No fitness function. Call initialize() first.")
        for _gen in range(n_generations):
            for island in self.islands:
                self.evaluate(island.id, fn)
                pop = island.population
                scored = [(fn(w), idx) for idx, w in enumerate(pop)]
                scored.sort(key=lambda x: x[0], reverse=True)
                half = max(1, len(pop) // 2)
                elites = [pop[idx] for _, idx in scored[:half]]
                new_pop: List[List[float]] = [list(e) for e in elites]
                while len(new_pop) < len(pop):
                    parent = self.rng.choice(elites)
                    new_pop.append(self._mutate(parent))
                island.population = new_pop
                island.generation += 1
            self._total_generation += 1
            if self.migrate_every > 0 and self._total_generation % self.migrate_every == 0:
                self.migrate()

    def diversity_metric(self) -> dict:
        """Return per-island std deviation of weights plus aggregate stats."""
        per_island: dict = {}
        all_stds: List[float] = []
        for island in self.islands:
            all_weights = [w for ind in island.population for w in ind]
            if len(all_weights) < 2:
                std = 0.0
            else:
                mean = sum(all_weights) / len(all_weights)
                variance = sum((x - mean) ** 2 for x in all_weights) / len(all_weights)
                std = math.sqrt(variance)
            per_island[island.id] = std
            all_stds.append(std)
        result = dict(per_island)
        result["mean_diversity"] = sum(all_stds) / len(all_stds) if all_stds else 0.0
        result["min_diversity"] = min(all_stds) if all_stds else 0.0
        result["max_diversity"] = max(all_stds) if all_stds else 0.0
        result["is_collapsed"] = any(s < 1e-6 for s in all_stds)
        return result

    def get_best_overall(
        self, fitness_fn: Optional[Callable] = None
    ) -> Tuple[int, List[float], float]:
        """Return (island_id, weights, fitness) for the global best."""
        fn = fitness_fn or self.fitness_fn
        if fn is None:
            raise ValueError("No fitness function. Call initialize() first.")
        best_id, best_weights, best_fit = 0, [], float("-inf")
        for island in self.islands:
            if not island.population:
                continue
            local_best_idx, local_best_fit = 0, float("-inf")
            for idx, w in enumerate(island.population):
                fit = fn(w)
                if fit > local_best_fit:
                    local_best_fit, local_best_idx = fit, idx
            if local_best_fit > best_fit:
                best_fit, best_id = local_best_fit, island.id
                best_weights = list(island.population[local_best_idx])
        return best_id, best_weights, best_fit

    def save_state(self, path: str) -> None:
        """Serialise model state to a JSON file."""
        state = {
            "n_islands": self.n_islands,
            "population_size": self.population_size,
            "migrate_top_k": self.migrate_top_k,
            "migrate_every": self.migrate_every,
            "mutation_rate": self.mutation_rate,
            "seed": self.seed,
            "total_generation": self._total_generation,
            "islands": [
                {
                    "id": is_.id,
                    "population": is_.population,
                    "best_fitness": is_.best_fitness,
                    "generation": is_.generation,
                }
                for is_ in self.islands
            ],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: str) -> "IslandModel":
        """Load a previously saved model from JSON."""
        with open(path, "r") as f:
            state = json.load(f)
        model = cls(
            n_islands=state["n_islands"],
            population_size=state["population_size"],
            migrate_top_k=state["migrate_top_k"],
            migrate_every=state["migrate_every"],
            mutation_rate=state["mutation_rate"],
            seed=state["seed"],
        )
        model._total_generation = state["total_generation"]
        model.islands = [
            Island(
                id=d["id"],
                population=d["population"],
                best_fitness=d["best_fitness"],
                generation=d["generation"],
            )
            for d in state["islands"]
        ]
        return model


def format_island_report(model: IslandModel) -> str:
    """Produce a human-readable text summary of the island model."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  NEUROEVOLUTION ISLAND MODEL REPORT")
    lines.append("=" * 60)
    lines.append(f"  Islands:          {model.n_islands}")
    lines.append(f"  Population:       {model.population_size} per island")
    lines.append(f"  Migrate top-k:    {model.migrate_top_k}")
    lines.append(f"  Migrate every:    {model.migrate_every} gens")
    lines.append(f"  Mutation rate:    {model.mutation_rate}")
    lines.append(f"  Total generation: {model._total_generation}")
    lines.append("-" * 60)
    div = model.diversity_metric()
    for island in model.islands:
        lines.append(f"  Island {island.id}:")
        lines.append(f"    generation : {island.generation}")
        lines.append(f"    best_fit   : {island.best_fitness:.6f}")
        std = div.get(island.id, 0.0)
        lines.append(f"    diversity  : {std:.6f}")
        lines.append("")
    lines.append("-" * 60)
    lines.append(f"  Mean diversity : {div['mean_diversity']:.6f}")
    lines.append(f"  Min diversity  : {div['min_diversity']:.6f}")
    lines.append(f"  Max diversity  : {div['max_diversity']:.6f}")
    lines.append(f"  Collapsed      : {div['is_collapsed']}")
    lines.append("=" * 60)
    return "\n".join(lines)
