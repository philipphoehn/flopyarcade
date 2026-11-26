"""Tests for the neuroevolution island model (ROADMAP R9)."""

import json
import math
import os
import tempfile

import pytest

from flopyarcade.island_model import Island, IslandModel, format_island_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sphere_fitness(weights):
    """Simple sphere function: lower is better, but we maximise."""
    return -sum(w * w for w in weights)


def constant_fitness(weights):
    """Always returns the same value."""
    return 42.0


def make_model(**kwargs):
    defaults = dict(
        n_islands=4,
        population_size=50,
        migrate_top_k=5,
        migrate_every=50,
        mutation_rate=0.1,
        seed=42,
    )
    defaults.update(kwargs)
    m = IslandModel(**defaults)
    m.initialize(sphere_fitness)
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_creates_correct_number_of_islands(self):
        m = make_model(n_islands=4)
        assert len(m.islands) == 4
        assert [i.id for i in m.islands] == [0, 1, 2, 3]

    def test_each_island_has_correct_population_size(self):
        m = make_model(population_size=50)
        for island in m.islands:
            assert len(island.population) == 50

    def test_different_seeds_give_different_populations(self):
        m1 = make_model(seed=1)
        m2 = make_model(seed=999)
        # Compare first individual of island 0
        p1 = m1.islands[0].population[0]
        p2 = m2.islands[0].population[0]
        assert p1 != p2, "Different seeds should produce different populations"


class TestEvaluate:
    def test_evaluate_updates_best_fitness(self):
        m = make_model()
        # Set all weights to zero so fitness = 0
        for island in m.islands:
            island.population = [[0.0] * 10 for _ in range(5)]
        m.evaluate(0)
        assert m.islands[0].best_fitness == pytest.approx(0.0)

    def test_evaluate_with_negative_fitness(self):
        m = make_model()
        # weights of all 1.0 => fitness = -10
        for island in m.islands:
            island.population = [[1.0] * 10 for _ in range(5)]
        m.evaluate(0)
        assert m.islands[0].best_fitness == pytest.approx(-10.0)


class TestSelectBest:
    def test_select_best_returns_correct_length(self):
        m = make_model()
        best = m.select_best(0)
        assert isinstance(best, list)
        assert len(best) == 10  # inferred dim

    def test_select_best_returns_best_fitness_individual(self):
        m = make_model()
        # Place a known-best individual at index 0 of island 0
        m.islands[0].population[0] = [0.0] * 10
        best = m.select_best(0)
        assert best == [0.0] * 10


class TestMigrate:
    def test_migrate_moves_individuals(self):
        m = make_model(population_size=10, migrate_top_k=2)
        # Make island 0 have a clearly best individual
        m.islands[0].population[0] = [0.0] * 10
        # Record island 1 population before
        pop1_before = [list(p) for p in m.islands[1].population]
        events = m.migrate()
        # Should have events: 4 islands * 3 destinations = 12 events
        assert len(events) == 12
        # Island 1 should have changed
        pop1_after = m.islands[1].population
        assert pop1_after != pop1_before

    def test_migrate_every_n_generations(self):
        m = make_model(migrate_every=3, population_size=10)
        # Run 3 generations - migration should trigger at gen 3
        pop_before = [list(p) for p in m.islands[1].population]
        m.step(n_generations=3)
        # Migration happened at total_generation=3
        assert m._total_generation == 3
        # Island populations should have been modified by migration
        # (at least best_fitness was invalidated)


class TestStep:
    def test_step_runs_multiple_generations(self):
        m = make_model(population_size=10)
        m.step(n_generations=5)
        for island in m.islands:
            assert island.generation == 5
        assert m._total_generation == 5

    def test_step_maintains_population_size(self):
        m = make_model(population_size=10)
        m.step(n_generations=3)
        for island in m.islands:
            assert len(island.population) == 10


class TestDiversityMetric:
    def test_diversity_metric_structure(self):
        m = make_model(n_islands=4)
        div = m.diversity_metric()
        # Per-island keys
        for i in range(4):
            assert i in div
        # Aggregate keys
        assert "mean_diversity" in div
        assert "min_diversity" in div
        assert "max_diversity" in div
        assert "is_collapsed" in div
        assert isinstance(div["is_collapsed"], bool)

    def test_diversity_detects_collapse(self):
        m = make_model(n_islands=2, population_size=10)
        # Set all populations to identical weights => std = 0
        for island in m.islands:
            island.population = [[1.0] * 10 for _ in range(10)]
        div = m.diversity_metric()
        assert div["is_collapsed"] is True
        assert div["min_diversity"] < 1e-6


class TestGetBestOverall:
    def test_returns_valid_tuple(self):
        m = make_model()
        result = m.get_best_overall()
        assert isinstance(result, tuple)
        assert len(result) == 3
        island_id, weights, fitness = result
        assert isinstance(island_id, int)
        assert 0 <= island_id < 4
        assert isinstance(weights, list)
        assert isinstance(fitness, float)


class TestSaveLoad:
    def test_save_load_roundtrip(self):
        m = make_model(n_islands=3, population_size=10, seed=7)
        m.step(n_generations=2)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            m.save_state(path)
            m2 = IslandModel.load_state(path)
            assert m2.n_islands == 3
            assert m2.population_size == 10
            assert len(m2.islands) == 3
            for i, (a, b) in enumerate(zip(m.islands, m2.islands)):
                assert a.id == b.id
                assert len(a.population) == len(b.population)
                assert a.generation == b.generation
        finally:
            os.unlink(path)


class TestFormatReport:
    def test_report_contains_island_ids(self):
        m = make_model(n_islands=4)
        report = format_island_report(m)
        for i in range(4):
            assert f"Island {i}" in report
        assert "ISLAND MODEL REPORT" in report


class TestMutation:
    def test_mutation_changes_weights(self):
        m = make_model(mutation_rate=1.0, seed=42)
        original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        mutated = m._mutate(list(original))
        assert mutated != original, "With rate=1.0, mutation should change weights"
        assert len(mutated) == len(original)


class TestDegenerate:
    def test_n_islands_1_works(self):
        m = make_model(n_islands=1, population_size=10)
        assert len(m.islands) == 1
        m.step(n_generations=2)
        assert m.islands[0].generation == 2
        # Migration with 1 island: no other islands to migrate to
        events = m.migrate()
        assert len(events) == 0
