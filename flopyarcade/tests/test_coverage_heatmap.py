"""Tests for flopyarcade.coverage_heatmap (ROADMAP P9)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from flopyarcade.coverage_heatmap import (
    ActionBucket,
    CoverageTracker,
    GenerationCoverage,
    compare_coverage,
    format_coverage_report,
    heatmap_data,
)


# ---------------------------------------------------------------------------
# Dataclass creation
# ---------------------------------------------------------------------------

def test_action_bucket_creation():
    b = ActionBucket(action="up", count=10, fraction=0.5)
    assert b.action == "up"
    assert b.count == 10
    assert b.fraction == pytest.approx(0.5)


def test_generation_coverage_creation():
    buckets = [ActionBucket(action="up", count=5, fraction=0.5),
               ActionBucket(action="down", count=5, fraction=0.5)]
    gc = GenerationCoverage(
        generation=1,
        buckets=buckets,
        total_actions=10,
        n_unique_actions=2,
        coverage=0.4,
    )
    assert gc.generation == 1
    assert len(gc.buckets) == 2
    assert gc.total_actions == 10
    assert gc.n_unique_actions == 2
    assert gc.coverage == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Tracker init
# ---------------------------------------------------------------------------

def test_tracker_init_defaults():
    t = CoverageTracker()
    assert t.action_space == ['up', 'down', 'left', 'right', 'keep']
    assert t.n_generations() == 0
    assert t.generations() == []


# ---------------------------------------------------------------------------
# record_generation
# ---------------------------------------------------------------------------

def test_record_generation_correct_fractions():
    t = CoverageTracker(action_space=['up', 'down', 'left', 'right', 'keep'])
    actions = ['up', 'up', 'down', 'left']  # 4 actions total
    gc = t.record_generation(0, actions)

    assert gc.total_actions == 4
    # up: 2/4=0.5, down: 1/4=0.25, left: 1/4=0.25, right: 0, keep: 0
    bucket_map = {b.action: b for b in gc.buckets}
    assert bucket_map['up'].count == 2
    assert bucket_map['up'].fraction == pytest.approx(0.5)
    assert bucket_map['down'].count == 1
    assert bucket_map['down'].fraction == pytest.approx(0.25)
    assert bucket_map['left'].count == 1
    assert bucket_map['left'].fraction == pytest.approx(0.25)
    assert bucket_map['right'].count == 0
    assert bucket_map['right'].fraction == pytest.approx(0.0)
    assert bucket_map['keep'].count == 0
    assert bucket_map['keep'].fraction == pytest.approx(0.0)
    # coverage: 3 unique out of 5
    assert gc.n_unique_actions == 3
    assert gc.coverage == pytest.approx(3 / 5)


def test_record_generation_all_same_action_coverage_02():
    t = CoverageTracker()  # 5 actions default
    actions = ['up'] * 50
    gc = t.record_generation(0, actions)
    # only 1 unique action out of 5
    assert gc.n_unique_actions == 1
    assert gc.coverage == pytest.approx(0.2)  # 1/5


def test_record_generation_all_actions_coverage_1():
    t = CoverageTracker()  # up, down, left, right, keep
    actions = ['up', 'down', 'left', 'right', 'keep']
    gc = t.record_generation(0, actions)
    assert gc.n_unique_actions == 5
    assert gc.coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# overall_coverage
# ---------------------------------------------------------------------------

def test_overall_coverage_mean():
    t = CoverageTracker()
    t.record_generation(0, ['up'] * 10)           # coverage 0.2
    t.record_generation(1, ['up', 'down', 'left', 'right', 'keep'])  # coverage 1.0
    # mean = (0.2 + 1.0) / 2 = 0.6
    assert t.overall_coverage() == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# coverage_trend
# ---------------------------------------------------------------------------

def test_coverage_trend_improving():
    t = CoverageTracker()
    t.record_generation(0, ['up'] * 10)                    # 0.2
    t.record_generation(1, ['up', 'down'] * 5)             # 0.4
    t.record_generation(2, ['up', 'down', 'left'] * 3)     # 0.6
    t.record_generation(3, ['up', 'down', 'left', 'right', 'keep'] * 1)  # 1.0
    trend = t.coverage_trend()
    assert trend['improving'] is True
    assert trend['slope'] > 0
    assert trend['first'] == pytest.approx(0.2)
    assert trend['last'] == pytest.approx(1.0)


def test_coverage_trend_worsening():
    t = CoverageTracker()
    t.record_generation(0, ['up', 'down', 'left', 'right', 'keep'])  # 1.0
    t.record_generation(1, ['up', 'down', 'left'] * 3)              # 0.6
    t.record_generation(2, ['up', 'down'] * 5)                       # 0.4
    t.record_generation(3, ['up'] * 10)                               # 0.2
    trend = t.coverage_trend()
    assert trend['improving'] is False
    assert trend['slope'] < 0


# ---------------------------------------------------------------------------
# most / least used
# ---------------------------------------------------------------------------

def test_most_used_action():
    t = CoverageTracker()
    # 'up' appears most across all generations
    t.record_generation(0, ['up', 'up', 'down'])
    t.record_generation(1, ['up', 'left'])
    assert t.most_used_action() == 'up'


def test_least_used_action():
    t = CoverageTracker()
    t.record_generation(0, ['up', 'up', 'down'])
    t.record_generation(1, ['up', 'left'])
    # 'right' and 'keep' both have 0 count; min returns first in action_space order
    assert t.least_used_action() in ('right', 'keep')


# ---------------------------------------------------------------------------
# gap_analysis
# ---------------------------------------------------------------------------

def test_gap_analysis_identifies_rare_actions():
    t = CoverageTracker()
    # 'keep' is never used → fraction 0 < 0.1
    t.record_generation(0, ['up'] * 20 + ['down'] * 20 + ['left'] * 20 + ['right'] * 20)
    t.record_generation(1, ['up'] * 10 + ['down'] * 10)
    gaps = t.gap_analysis()
    gap_actions = [g['action'] for g in gaps]
    assert 'keep' in gap_actions
    # 'left' and 'right' also have low overall fraction
    for g in gaps:
        assert g['fraction'] < 0.1
        assert g['action'] in t.action_space


# ---------------------------------------------------------------------------
# compare_coverage
# ---------------------------------------------------------------------------

def test_compare_coverage_structure():
    ta = CoverageTracker()
    ta.record_generation(0, ['up'] * 10)         # 0.2
    ta.record_generation(1, ['up', 'down'] * 5)  # 0.4

    tb = CoverageTracker()
    tb.record_generation(0, ['up', 'down', 'left', 'right', 'keep'])  # 1.0
    tb.record_generation(1, ['up'] * 10)                                # 0.2

    result = compare_coverage(ta, tb)
    assert 'per_generation' in result
    assert 'overall_a' in result
    assert 'overall_b' in result
    assert 'overall_diff' in result

    pg = result['per_generation']
    assert '0' in pg and '1' in pg
    assert pg['0']['a'] == pytest.approx(0.2)
    assert pg['0']['b'] == pytest.approx(1.0)
    assert pg['0']['diff'] == pytest.approx(-0.8)

    assert result['overall_a'] == pytest.approx(0.3)
    assert result['overall_b'] == pytest.approx(0.6)
    assert result['overall_diff'] == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# format_coverage_report
# ---------------------------------------------------------------------------

def test_format_coverage_report_contains_action_names():
    t = CoverageTracker()
    t.record_generation(0, ['up', 'down', 'left'])
    report = format_coverage_report(t)
    assert 'up' in report
    assert 'down' in report
    assert 'left' in report
    assert 'right' in report
    assert 'keep' in report
    assert 'COVERAGE HEATMAP REPORT' in report


# ---------------------------------------------------------------------------
# heatmap_data
# ---------------------------------------------------------------------------

def test_heatmap_data_matrix_shape():
    t = CoverageTracker()  # 5 actions
    t.record_generation(0, ['up', 'down'])
    t.record_generation(1, ['left', 'right', 'keep'])

    data = heatmap_data(t)
    assert len(data['generations']) == 2
    assert len(data['actions']) == 5
    assert len(data['matrix']) == 2
    for row in data['matrix']:
        assert len(row) == 5
    # gen 0: up=0.5, down=0.5, left=0, right=0, keep=0
    assert data['matrix'][0][0] == pytest.approx(0.5)  # up
    assert data['matrix'][0][1] == pytest.approx(0.5)  # down
    assert data['matrix'][0][2] == pytest.approx(0.0)  # left


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    t = CoverageTracker()
    t.record_generation(0, ['up', 'down', 'left'])
    t.record_generation(1, ['right'] * 5)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name

    try:
        t.save_json(path)
        t2 = CoverageTracker.load_json(path)
        assert t2.action_space == t.action_space
        assert t2.n_generations() == 2
        assert t2.overall_coverage() == pytest.approx(t.overall_coverage())
        assert t2.most_used_action() == t.most_used_action()
        assert t2.least_used_action() == t.least_used_action()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# empty tracker
# ---------------------------------------------------------------------------

def test_empty_tracker():
    t = CoverageTracker()
    assert t.n_generations() == 0
    assert t.overall_coverage() == 0.0
    assert t.most_used_action() == 'up'
    assert t.least_used_action() == 'up'
    trend = t.coverage_trend()
    assert trend['improving'] is False
    assert trend['slope'] == 0.0
    dist = t.action_distribution()
    for a in t.action_space:
        assert dist[a] == 0.0
    assert t.gap_analysis() == []
    report = format_coverage_report(t)
    assert 'Generations:  0' in report
    data = heatmap_data(t)
    assert data['matrix'] == []
    assert len(data['actions']) == 5

