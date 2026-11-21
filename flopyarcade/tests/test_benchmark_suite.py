"""Tests for flopyarcade.benchmark_suite (B17)."""

import json
import math
import os
import tempfile

import pytest

from flopyarcade.benchmark_suite import (
    BenchmarkConfig,
    BenchmarkResult,
    compare_tables,
    load_results,
    run_benchmark,
    save_results,
    score_table,
)


# 1. Test BenchmarkResult creation
def test_benchmark_result_creation():
    r = BenchmarkResult(
        envtype='1s-d',
        agent_type='random',
        seed=42,
        score=0.75,
        episodes=10,
        total_seconds=1.5,
        metadata={'source': 'mock'},
    )
    assert r.envtype == '1s-d'
    assert r.agent_type == 'random'
    assert r.seed == 42
    assert r.score == 0.75
    assert r.episodes == 10
    assert r.total_seconds == 1.5
    assert r.metadata == {'source': 'mock'}

    # metadata optional
    r2 = BenchmarkResult(
        envtype='2s-d',
        agent_type='trained',
        seed=123,
        score=0.5,
        episodes=5,
        total_seconds=0.3,
    )
    assert r2.metadata is None


# 2. Test BenchmarkConfig defaults
def test_benchmark_config_defaults():
    cfg = BenchmarkConfig()
    assert cfg.envtypes == ['1s-d', '2s-d', '3s-d', '1s-c', '2s-c', '3s-c']
    assert cfg.seeds == [42, 123, 456, 789, 101]
    assert cfg.agent_types == ['random', 'trained']
    assert cfg.episodes_per_cell == 10
    assert cfg.output_path == 'benchmark_results.json'


# 3. Test run_benchmark returns correct number of results
def test_run_benchmark_result_count():
    cfg = BenchmarkConfig(
        envtypes=['1s-d', '2s-d'],
        seeds=[42, 123],
        agent_types=['random', 'trained'],
        episodes_per_cell=2,
    )
    results = run_benchmark(cfg)
    expected = len(cfg.envtypes) * len(cfg.agent_types) * len(cfg.seeds)
    assert len(results) == expected  # 2 * 2 * 2 = 8


# 4. Test run_benchmark results have valid scores
def test_run_benchmark_valid_scores():
    cfg = BenchmarkConfig(
        envtypes=['1s-d'],
        seeds=[42],
        agent_types=['random'],
        episodes_per_cell=3,
    )
    results = run_benchmark(cfg)
    for r in results:
        assert isinstance(r.score, float)
        assert r.score >= 0.0
        assert r.episodes == 3
        assert isinstance(r.total_seconds, float)
        assert r.total_seconds >= 0.0


# 5. Test score_table structure
def test_score_table_structure():
    cfg = BenchmarkConfig(
        envtypes=['1s-d', '2s-d'],
        seeds=[42, 123, 456],
        agent_types=['random', 'trained'],
        episodes_per_cell=2,
    )
    results = run_benchmark(cfg)
    table = score_table(results)

    assert set(table.keys()) == {'1s-d', '2s-d'}
    for envtype in table:
        assert set(table[envtype].keys()) == {'random', 'trained'}
        for agent_type in table[envtype]:
            stats = table[envtype][agent_type]
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert 'n' in stats


# 6. Test score_table mean/std values
def test_score_table_mean_std():
    # Use known scores for manual verification
    results = [
        BenchmarkResult('1s-d', 'random', 42, 0.2, 1, 0.0, None),
        BenchmarkResult('1s-d', 'random', 43, 0.4, 1, 0.0, None),
        BenchmarkResult('1s-d', 'random', 44, 0.6, 1, 0.0, None),
    ]
    table = score_table(results)
    stats = table['1s-d']['random']

    expected_mean = (0.2 + 0.4 + 0.6) / 3
    expected_std = math.sqrt(
        ((0.2 - expected_mean) ** 2 + (0.4 - expected_mean) ** 2 + (0.6 - expected_mean) ** 2) / 3
    )
    assert stats['mean'] == pytest.approx(expected_mean)
    assert stats['std'] == pytest.approx(expected_std)
    assert stats['min'] == pytest.approx(0.2)
    assert stats['max'] == pytest.approx(0.6)
    assert stats['n'] == 3.0


# 7. Test save/load roundtrip
def test_save_load_roundtrip():
    cfg = BenchmarkConfig(
        envtypes=['1s-d'],
        seeds=[42, 123],
        agent_types=['random'],
        episodes_per_cell=3,
    )
    results = run_benchmark(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'results.json')
        save_results(results, path)

        # File exists and is valid JSON
        assert os.path.exists(path)
        with open(path) as f:
            raw = json.load(f)
        assert len(raw) == len(results)

        loaded = load_results(path)
        assert len(loaded) == len(results)
        for orig, ld in zip(results, loaded):
            assert orig.envtype == ld.envtype
            assert orig.agent_type == ld.agent_type
            assert orig.seed == ld.seed
            assert orig.score == ld.score
            assert orig.episodes == ld.episodes
            assert orig.total_seconds == ld.total_seconds
            assert orig.metadata == ld.metadata


# 8. Test compare_tables with identical tables (all zeros)
def test_compare_tables_identical():
    table = {
        '1s-d': {'random': {'mean': 0.5, 'std': 0.1, 'min': 0.4, 'max': 0.6, 'n': 5.0}},
        '2s-d': {'random': {'mean': 0.3, 'std': 0.2, 'min': 0.1, 'max': 0.5, 'n': 5.0}},
    }
    deltas = compare_tables(table, table)
    for envtype, delta in deltas.items():
        assert delta == pytest.approx(0.0)


# 9. Test compare_tables with different scores
def test_compare_tables_different():
    table_a = {
        '1s-d': {'random': {'mean': 0.3, 'std': 0.1, 'min': 0.2, 'max': 0.4, 'n': 5.0}},
    }
    table_b = {
        '1s-d': {'random': {'mean': 0.6, 'std': 0.1, 'min': 0.5, 'max': 0.7, 'n': 5.0}},
    }
    deltas = compare_tables(table_a, table_b)
    assert '1s-d' in deltas
    assert deltas['1s-d'] == pytest.approx(0.3)  # 0.6 - 0.3


# 10. Test deterministic: same config gives same results
def test_deterministic_results():
    cfg = BenchmarkConfig(
        envtypes=['1s-d', '2s-d'],
        seeds=[42, 123],
        agent_types=['random', 'trained'],
        episodes_per_cell=3,
    )
    results_1 = run_benchmark(cfg)
    results_2 = run_benchmark(cfg)

    assert len(results_1) == len(results_2)
    for r1, r2 in zip(results_1, results_2):
        assert r1.envtype == r2.envtype
        assert r1.agent_type == r2.agent_type
        assert r1.seed == r2.seed
        assert r1.score == r2.score


# 11. Test different seeds give different scores (at least one differs)
def test_different_seeds_different_scores():
    cfg = BenchmarkConfig(
        envtypes=['1s-d'],
        seeds=[42, 999],
        agent_types=['random'],
        episodes_per_cell=3,
    )
    results = run_benchmark(cfg)
    scores = [r.score for r in results]
    # With mock scoring, different seeds should produce different scores
    assert len(set(scores)) > 1, f"Expected different scores for different seeds, got {scores}"


# 12. Test empty envtypes list returns empty results
def test_empty_envtypes():
    cfg = BenchmarkConfig(
        envtypes=[],
        seeds=[42, 123],
        agent_types=['random'],
        episodes_per_cell=3,
    )
    results = run_benchmark(cfg)
    assert results == []


# 13. Bonus: test score_table with empty results
def test_score_table_empty():
    table = score_table([])
    assert table == {}


# 14. Bonus: test compare_tables with non-overlapping envtypes
def test_compare_tables_no_overlap():
    table_a = {
        '1s-d': {'random': {'mean': 0.5, 'std': 0.1, 'min': 0.4, 'max': 0.6, 'n': 5.0}},
    }
    table_b = {
        '2s-d': {'random': {'mean': 0.7, 'std': 0.1, 'min': 0.6, 'max': 0.8, 'n': 5.0}},
    }
    deltas = compare_tables(table_a, table_b)
    # No common envtypes, so result should be empty
    assert deltas == {}
