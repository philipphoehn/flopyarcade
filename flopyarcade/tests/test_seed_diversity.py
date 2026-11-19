"""A9: Tests for seed diversity combiner (combine multiple initial runs)."""

import pytest
import numpy as np

from flopyarcade.seed_diversity import SeedDiversityCombiner, diversity_metrics


class TestSeedDiversityCombinerInit:
    def test_defaults(self):
        c = SeedDiversityCombiner()
        assert c.pop_size == 50
        assert c.n_candidates == 5
        assert c.top_k == 3

    def test_pop_size_zero_raises(self):
        with pytest.raises(ValueError, match="pop_size"):
            SeedDiversityCombiner(pop_size=0)

    def test_n_candidates_zero_raises(self):
        with pytest.raises(ValueError, match="n_candidates"):
            SeedDiversityCombiner(n_candidates=0)

    def test_top_k_too_large_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            SeedDiversityCombiner(n_candidates=3, top_k=5)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            SeedDiversityCombiner(top_k=0)


class TestSeedDiversityCombinerGenerate:
    def test_returns_correct_length(self):
        c = SeedDiversityCombiner(pop_size=20, n_candidates=3, top_k=2, random_seed=42)
        seeds = c.generate()
        assert len(seeds) == 20

    def test_returns_integers(self):
        c = SeedDiversityCombiner(pop_size=10, n_candidates=2, top_k=1, random_seed=1)
        seeds = c.generate()
        assert all(isinstance(s, int) for s in seeds)

    def test_reproducible_with_seed(self):
        c1 = SeedDiversityCombiner(pop_size=10, n_candidates=3, top_k=2, random_seed=99)
        c2 = SeedDiversityCombiner(pop_size=10, n_candidates=3, top_k=2, random_seed=99)
        assert c1.generate() == c2.generate()

    def test_different_seeds_differ(self):
        c1 = SeedDiversityCombiner(pop_size=10, n_candidates=3, top_k=2, random_seed=1)
        c2 = SeedDiversityCombiner(pop_size=10, n_candidates=3, top_k=2, random_seed=2)
        assert c1.generate() != c2.generate()

    def test_base_seed_override(self):
        c = SeedDiversityCombiner(pop_size=10, n_candidates=2, top_k=1)
        s1 = c.generate(base_seed=42)
        s2 = c.generate(base_seed=42)
        assert s1 == s2

    def test_single_candidate(self):
        c = SeedDiversityCombiner(pop_size=5, n_candidates=1, top_k=1, random_seed=7)
        seeds = c.generate()
        assert len(seeds) == 5


class TestDiversityScore:
    def test_identical_seeds_zero(self):
        c = SeedDiversityCombiner(pop_size=10, n_candidates=1, top_k=1)
        pop = np.array([42] * 10)
        assert c._diversity_score(pop) == 0.0

    def test_single_element_zero(self):
        c = SeedDiversityCombiner(pop_size=1, n_candidates=1, top_k=1)
        assert c._diversity_score(np.array([5])) == 0.0

    def test_spread_seeds_high_score(self):
        c = SeedDiversityCombiner(pop_size=10, n_candidates=1, top_k=1)
        pop = np.array([1, 100, 200, 300, 400, 500, 600, 700, 800, 900])
        score = c._diversity_score(pop)
        assert score > 0.3


class TestDiversityMetrics:
    def test_single_seed(self):
        m = diversity_metrics([42])
        assert m['mean_pairwise_distance'] == 0.0
        assert m['unique_ratio'] == 1.0

    def test_all_same(self):
        m = diversity_metrics([5, 5, 5, 5])
        assert m['mean_pairwise_distance'] == 0.0
        assert m['unique_ratio'] == pytest.approx(0.25)

    def test_spread(self):
        m = diversity_metrics([1, 50, 100])
        assert m['mean_pairwise_distance'] > 0
        assert m['std'] > 0
        assert m['min'] == 1.0
        assert m['max'] == 100.0
        assert m['unique_ratio'] == 1.0

    def test_empty_list(self):
        m = diversity_metrics([])
        assert m['mean_pairwise_distance'] == 0.0
        assert m['unique_ratio'] == 1.0
