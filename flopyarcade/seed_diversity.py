"""Combine multiple initial runs to produce a high-variability seed set.

Roadmap A9: Instead of a single random seed generation, run multiple
candidate population initialisations and merge the most diverse ones
into one population. This improves the GA starting point by guaranteeing
higher pairwise seed diversity.

Usage::

    from flopyarcade.seed_diversity import SeedDiversityCombiner

    combiner = SeedDiversityCombiner(pop_size=50, n_candidates=5, top_k=3)
    diverse_seeds = combiner.generate(seed_base=42)
"""

import numpy as np
from typing import List, Optional


class SeedDiversityCombiner:
    """Generate a high-variability seed population by merging diverse candidates.

    Args:
        pop_size: Number of seeds in the final population.
        n_candidates: How many candidate populations to generate and score.
        top_k: How many candidate populations to merge into the final set.
        seed_range: (low, high) exclusive range for random integers.
        random_seed: Optional base seed for reproducibility.
    """

    def __init__(
        self,
        pop_size: int = 50,
        n_candidates: int = 5,
        top_k: int = 3,
        seed_range: tuple = (1, 10_000_000),
        random_seed: Optional[int] = None,
    ):
        if pop_size < 1:
            raise ValueError("pop_size must be >= 1")
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        if top_k < 1 or top_k > n_candidates:
            raise ValueError("top_k must be between 1 and n_candidates")
        self.pop_size = pop_size
        self.n_candidates = n_candidates
        self.top_k = top_k
        self.seed_range = seed_range
        self.random_seed = random_seed

    def generate(self, base_seed: Optional[int] = None) -> List[int]:
        """Generate the final diverse seed population.

        Args:
            base_seed: Optional override for the base random seed.

        Returns:
            List of integers (the diverse seed set), length == pop_size.
        """
        rng_base = base_seed if base_seed is not None else self.random_seed
        if rng_base is not None:
            rng = np.random.default_rng(rng_base)
        else:
            rng = np.random.default_rng()

        candidates = self._generate_candidates(rng)
        scored = [(self._diversity_score(c), i) for i, c in enumerate(candidates)]
        scored.sort(reverse=True)
        top_indices = [idx for _, idx in scored[:self.top_k]]

        merged = self._merge_candidates(candidates, top_indices, rng)
        return [int(s) for s in merged]

    def _generate_candidates(self, rng: np.random.Generator) -> List[np.ndarray]:
        """Generate n_candidates random populations."""
        low, high = self.seed_range
        return [
            rng.integers(low, high, size=self.pop_size)
            for _ in range(self.n_candidates)
        ]

    @staticmethod
    def _diversity_score(pop: np.ndarray) -> float:
        """Mean pairwise distance (normalised) within a population.

        Uses the mean of |seed_i - seed_j| / range for all i < j.
        For large populations, subsamples to keep O(n^2) tractable.
        """
        n = len(pop)
        if n < 2:
            return 0.0
        # Subsample for large populations
        if n > 200:
            idx = np.random.default_rng(0).choice(n, 200, replace=False)
            pop = pop[idx]
            n = 200

        # Vectorised pairwise absolute differences
        diff = np.abs(pop[:, None] - pop[None, :])
        # Exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        mean_diff = diff[mask].mean()
        # Normalise by range (approximate: use max - min of the set)
        rng_span = pop.max() - pop.min()
        if rng_span == 0:
            return 0.0
        return float(mean_diff / rng_span)

    @staticmethod
    def _merge_candidates(
        candidates: List[np.ndarray],
        top_indices: List[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Merge top-k candidate populations into one of the same size.

        Strategy: take the first `pop_size // top_k` seeds from each
top candidate (shuffled), then fill remaining slots from a fresh
random draw to avoid duplicates.
        """
        pop_size = len(candidates[0])
        per_candidate = pop_size // len(top_indices)
        remainder = pop_size - per_candidate * len(top_indices)

        merged = []
        for i, idx in enumerate(top_indices):
            pop = candidates[idx].copy()
            rng.shuffle(pop)
            take = per_candidate + (1 if i < remainder else 0)
            merged.append(pop[:take])

        result = np.concatenate(merged)
        # Trim or pad to exact pop_size
        if len(result) > pop_size:
            result = result[:pop_size]
        elif len(result) < pop_size:
            low, high = candidates[0][0].item() and (1, 10_000_000) or (1, 10_000_000)
            fill = rng.integers(1, 10_000_000, size=pop_size - len(result))
            result = np.concatenate([result, fill])
        return result


def diversity_metrics(seeds: List[int]) -> dict:
    """Compute diversity metrics for a seed set.

    Args:
        seeds: List of integer seeds.

    Returns:
        Dict with 'mean_pairwise_distance', 'std', 'min', 'max', 'unique_ratio'.
    """
    arr = np.array(seeds, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return {
            'mean_pairwise_distance': 0.0,
            'std': 0.0,
            'min': float(arr[0]) if n else 0.0,
            'max': float(arr[0]) if n else 0.0,
            'unique_ratio': 1.0,
        }

    diff = np.abs(arr[:, None] - arr[None, :])
    mask = ~np.eye(n, dtype=bool)
    mean_pw = float(diff[mask].mean())
    unique = len(np.unique(seeds)) / n

    return {
        'mean_pairwise_distance': mean_pw,
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'unique_ratio': unique,
    }
