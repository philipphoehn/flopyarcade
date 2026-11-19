"""Tests for flopyarcade.novelty_fix (A2: novelty metric fix)."""

import numpy as np
import pytest

from flopyarcade.novelty_fix import (
    NoveltyController,
    NoveltyControllerConfig,
    NoveltyElitism,
    NoveltyThreshold,
    length_invariant_score,
)


# ---------------------------------------------------------------------------
# length_invariant_score
# ---------------------------------------------------------------------------

class TestLengthInvariantScore:
    def test_basic_mean(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        assert length_invariant_score(scores) == pytest.approx(2.5)

    def test_single_element(self):
        assert length_invariant_score([7.0]) == pytest.approx(7.0)

    def test_with_episode_counts(self):
        scores = [1.0, 2.0, 3.0]
        counts = [1, 2, 3]  # total = 6
        expected = 6.0 / 6.0
        assert length_invariant_score(scores, counts) == pytest.approx(expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            length_invariant_score([])

    def test_zero_counts_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            length_invariant_score([1.0, 2.0], [0, 0])

    def test_length_invariance_property(self):
        """Doubling episodes with same per-episode score → same mean."""
        s1 = [1.0, 1.0, 1.0]
        s2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert length_invariant_score(s1) == pytest.approx(length_invariant_score(s2))


# ---------------------------------------------------------------------------
# NoveltyThreshold
# ---------------------------------------------------------------------------

class TestNoveltyThreshold:
    def test_defaults(self):
        t = NoveltyThreshold()
        assert t.threshold == 0.05
        assert t.patience == 3
        assert t.min_population == 4

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            NoveltyThreshold(threshold=-0.1)

    def test_invalid_patience(self):
        with pytest.raises(ValueError):
            NoveltyThreshold(patience=0)

    def test_invalid_min_population(self):
        with pytest.raises(ValueError):
            NoveltyThreshold(min_population=1)

    def test_check_below_threshold(self):
        t = NoveltyThreshold(threshold=0.1, patience=2, min_population=3)
        assert t.check(0.05, 10, 2) is True

    def test_check_above_threshold(self):
        t = NoveltyThreshold(threshold=0.1, patience=2, min_population=3)
        assert t.check(0.15, 10, 5) is False

    def test_check_patience_not_reached(self):
        t = NoveltyThreshold(threshold=0.1, patience=3, min_population=3)
        assert t.check(0.05, 10, 2) is False

    def test_check_min_population(self):
        t = NoveltyThreshold(threshold=0.1, patience=1, min_population=5)
        assert t.check(0.05, 3, 10) is False

    def test_update_low_count_increments(self):
        t = NoveltyThreshold(threshold=0.1, min_population=2)
        assert t.update_low_count(0.05, 10, 0) == 1
        assert t.update_low_count(0.05, 10, 1) == 2

    def test_update_low_count_resets(self):
        t = NoveltyThreshold(threshold=0.1, min_population=2)
        t.update_low_count(0.05, 10, 0)
        t.update_low_count(0.05, 10, 1)
        assert t.update_low_count(0.2, 10, 2) == 0

    def test_update_low_count_below_min_pop(self):
        t = NoveltyThreshold(threshold=0.1, min_population=5)
        assert t.update_low_count(0.01, 3, 5) == 0


# ---------------------------------------------------------------------------
# NoveltyElitism
# ---------------------------------------------------------------------------

class TestNoveltyElitism:
    def test_defaults(self):
        e = NoveltyElitism()
        assert e.n_elites == 10
        assert e.n_children == 20

    def test_invalid_elites(self):
        with pytest.raises(ValueError):
            NoveltyElitism(n_elites=0)

    def test_invalid_children(self):
        with pytest.raises(ValueError):
            NoveltyElitism(n_children=-1)

    def test_partition_basic(self):
        e = NoveltyElitism(n_elites=2, n_children=3)
        scores = [0.5, 1.0, 0.3, 0.8, 0.1, 0.7]
        result = e.partition(scores)
        assert len(result['elites']) == 2
        assert len(result['children']) == 3
        assert len(result['discarded']) == 1
        # Top 2: indices 1 (1.0) and 3 (0.8)
        assert sorted(result['elites']) == [1, 3]

    def test_partition_more_agents_than_slots(self):
        e = NoveltyElitism(n_elites=2, n_children=2)
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        result = e.partition(scores)
        assert len(result['elites']) == 2
        assert len(result['children']) == 2
        assert len(result['discarded']) == 4

    def test_partition_fewer_agents_than_elites(self):
        e = NoveltyElitism(n_elites=10, n_children=20)
        scores = [0.5, 1.0, 0.3]
        result = e.partition(scores)
        assert len(result['elites']) == 3
        assert len(result['children']) == 0
        assert len(result['discarded']) == 0

    def test_partition_empty(self):
        e = NoveltyElitism()
        result = e.partition([])
        assert result == {'elites': [], 'children': [], 'discarded': []}

    def test_is_elite(self):
        e = NoveltyElitism(n_elites=1, n_children=5)
        scores = [0.1, 0.9, 0.5]
        assert e.is_elite(1, scores) is True
        assert e.is_elite(0, scores) is False

    def test_partition_disjoint(self):
        e = NoveltyElitism(n_elites=3, n_children=4)
        scores = np.random.rand(20).tolist()
        result = e.partition(scores)
        all_indices = set(result['elites'] + result['children'] + result['discarded'])
        assert all_indices == set(range(20))
        # No overlap
        assert len(all_indices) == len(result['elites']) + len(result['children']) + len(result['discarded'])


# ---------------------------------------------------------------------------
# NoveltyController
# ---------------------------------------------------------------------------

class TestNoveltyController:
    def test_defaults(self):
        ctrl = NoveltyController()
        assert ctrl.threshold.threshold == 0.05
        assert ctrl.elitism.n_elites == 10

    def test_custom_config(self):
        cfg = NoveltyControllerConfig(threshold=0.1, n_elites=5, n_children=10)
        ctrl = NoveltyController(config=cfg)
        assert ctrl.threshold.threshold == 0.1
        assert ctrl.elitism.n_elites == 5
        assert ctrl.elitism.n_children == 10

    def test_evaluate_high_novelty_no_stop(self):
        ctrl = NoveltyController(NoveltyControllerConfig(
            threshold=0.1, patience=3, min_population=2,
            n_elites=2, n_children=3
        ))
        d = ctrl.evaluate([0.5, 0.6, 0.7, 0.8, 0.9])
        assert d['stop'] is False
        assert d['mean_novelty'] == pytest.approx(0.7)
        assert d['n_unique'] == 5
        assert len(d['elites']) == 2
        assert len(d['children']) == 3

    def test_evaluate_stop_after_patience(self):
        ctrl = NoveltyController(NoveltyControllerConfig(
            threshold=0.1, patience=2, min_population=2,
            n_elites=2, n_children=3
        ))
        low = [0.05, 0.04, 0.03, 0.02, 0.01]
        d1 = ctrl.evaluate(low)
        assert d1['stop'] is False
        assert d1['low_count'] == 1
        d2 = ctrl.evaluate(low)
        assert d2['stop'] is True
        assert d2['low_count'] == 2

    def test_evaluate_empty(self):
        ctrl = NoveltyController()
        d = ctrl.evaluate([])
        assert d['stop'] is True
        assert d['n_unique'] == 0

    def test_reset(self):
        ctrl = NoveltyController(NoveltyControllerConfig(
            threshold=0.1, patience=1, min_population=2
        ))
        ctrl.evaluate([0.05, 0.04])  # low_count → 1
        assert ctrl.low_count == 1
        ctrl.reset()
        assert ctrl.low_count == 0

    def test_to_dict(self):
        ctrl = NoveltyController(NoveltyControllerConfig(
            threshold=0.2, patience=5, min_population=8,
            n_elites=4, n_children=12
        ))
        d = ctrl.to_dict()
        assert d['threshold'] == 0.2
        assert d['patience'] == 5
        assert d['min_population'] == 8
        assert d['n_elites'] == 4
        assert d['n_children'] == 12

    def test_repr(self):
        ctrl = NoveltyController()
        r = repr(ctrl)
        assert 'NoveltyController' in r
        assert 'threshold=' in r

    def test_statistics_in_result(self):
        ctrl = NoveltyController()
        scores = [1.0, 3.0, 2.0]
        d = ctrl.evaluate(scores)
        assert d['min_novelty'] == pytest.approx(1.0)
        assert d['max_novelty'] == pytest.approx(3.0)
        assert d['mean_novelty'] == pytest.approx(2.0)

    def test_novelty_resets_low_count(self):
        ctrl = NoveltyController(NoveltyControllerConfig(
            threshold=0.1, patience=5, min_population=2
        ))
        ctrl.evaluate([0.05, 0.04])  # low
        ctrl.evaluate([0.05, 0.04])  # low → count=2
        ctrl.evaluate([0.5, 0.6])    # high → resets
        assert ctrl.low_count == 0
