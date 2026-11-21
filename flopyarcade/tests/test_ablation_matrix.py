"""Tests for B20: ablation matrix (reward component impact analysis)."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flopyarcade.ablation_matrix import (
    RewardComponent,
    compute_ablation_scores,
    build_ablation_table,
    rank_components,
    find_critical_components,
    format_ablation_report,
)


# ---------- RewardComponent ----------

def test_reward_component_creation():
    """Test RewardComponent dataclass creation with defaults and explicit values."""
    c = RewardComponent(name='flow')
    assert c.name == 'flow'
    assert c.weight == 1.0
    assert c.description == ''

    c2 = RewardComponent(name='head', weight=2.5, description='head tracking')
    assert c2.name == 'head'
    assert c2.weight == 2.5
    assert c2.description == 'head tracking'


# ---------- compute_ablation_scores ----------

def test_compute_ablation_scores_basic():
    """Single envtype, single component."""
    base = {'1s-d': 100.0}
    comp = {'1s-d': {'flow': 70.0}}
    result = compute_ablation_scores(base, comp)
    assert result['1s-d'] == pytest.approx(30.0)


def test_compute_ablation_scores_multiple_envtypes():
    """Multiple envtypes with multiple components."""
    base = {'1s-d': 200.0, '2s-c': 300.0}
    comp = {
        '1s-d': {'flow': 150.0, 'head': 160.0},
        '2s-c': {'flow': 250.0, 'head': 270.0},
    }
    result = compute_ablation_scores(base, comp)
    # 1s-d: avg removed = (150+160)/2 = 155, delta = 200-155 = 45
    assert result['1s-d'] == pytest.approx(45.0)
    # 2s-c: avg removed = (250+270)/2 = 260, delta = 300-260 = 40
    assert result['2s-c'] == pytest.approx(40.0)


# ---------- build_ablation_table ----------

def test_build_ablation_table_mock():
    """Basic table build with a simple scores_fn."""
    components = [
        RewardComponent(name='flow'),
        RewardComponent(name='head'),
    ]

    def scores_fn(envtype, keep):
        # full score = 100, removing 'flow' drops to 60, removing 'head' drops to 80
        if 'flow' not in keep and 'head' in keep:
            return 60.0
        elif 'head' not in keep and 'flow' in keep:
            return 80.0
        elif len(keep) == 0:
            return 40.0
        return 100.0

    table = build_ablation_table(['1s-d'], components, scores_fn)
    assert len(table) == 2
    # find flow row
    flow_row = [r for r in table if r['component'] == 'flow'][0]
    assert flow_row['score_full'] == 100.0
    assert flow_row['score_ablated'] == 60.0
    assert flow_row['delta'] == 40.0


def test_build_ablation_table_sorted_by_delta():
    """Rows must be sorted by |delta| descending."""
    components = [
        RewardComponent(name='a'),
        RewardComponent(name='b'),
        RewardComponent(name='c'),
    ]

    def scores_fn(envtype, keep):
        # Removing 'a' drops 50, 'b' drops 10, 'c' drops 20
        if 'a' not in keep:
            return 50.0
        elif 'b' not in keep:
            return 90.0
        elif 'c' not in keep:
            return 80.0
        return 100.0

    table = build_ablation_table(['env1'], components, scores_fn)
    deltas = [abs(r['delta']) for r in table]
    assert deltas == sorted(deltas, reverse=True)
    # 'a' should be first (largest delta)
    assert table[0]['component'] == 'a'


def test_build_ablation_table_pct_change():
    """Verify pct_change calculation."""
    components = [RewardComponent(name='x')]

    def scores_fn(envtype, keep):
        if 'x' not in keep:
            return 50.0
        return 100.0

    table = build_ablation_table(['e1'], components, scores_fn)
    assert table[0]['pct_change'] == pytest.approx(50.0)

    # Edge case: full_score = 0 -> pct_change should be 0
    def scores_fn_zero(envtype, keep):
        return 0.0

    table_zero = build_ablation_table(['e1'], components, scores_fn_zero)
    assert table_zero[0]['pct_change'] == 0.0


# ---------- rank_components ----------

def test_rank_components_ordering():
    """Components ranked by first appearance (which is |delta| desc in table)."""
    table = [
        {'component': 'z', 'delta': 50.0, 'pct_change': 50.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 50.0},
        {'component': 'a', 'delta': 30.0, 'pct_change': 30.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 70.0},
        {'component': 'm', 'delta': 10.0, 'pct_change': 10.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 90.0},
    ]
    ranked = rank_components(table)
    assert ranked == ['z', 'a', 'm']


def test_rank_components_deduplication():
    """Same component appearing in multiple envtypes should appear only once."""
    table = [
        {'component': 'flow', 'delta': 40.0, 'pct_change': 40.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 60.0},
        {'component': 'head', 'delta': 20.0, 'pct_change': 20.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 80.0},
        {'component': 'flow', 'delta': 15.0, 'pct_change': 15.0, 'envtype': 'e2', 'score_full': 100.0, 'score_ablated': 85.0},
    ]
    ranked = rank_components(table)
    assert ranked == ['flow', 'head']


# ---------- find_critical_components ----------

def test_find_critical_components_with_threshold():
    """Components with pct_change > 10% are critical."""
    table = [
        {'component': 'big', 'delta': 30.0, 'pct_change': 30.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 70.0},
        {'component': 'small', 'delta': 5.0, 'pct_change': 5.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 95.0},
        {'component': 'edge', 'delta': 11.0, 'pct_change': 11.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 89.0},
    ]
    critical = find_critical_components(table, threshold_pct=10.0)
    assert 'big' in critical
    assert 'edge' in critical
    assert 'small' not in critical


def test_find_critical_components_no_critical():
    """All components below threshold -> empty list."""
    table = [
        {'component': 'a', 'delta': 3.0, 'pct_change': 3.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 97.0},
        {'component': 'b', 'delta': 5.0, 'pct_change': 5.0, 'envtype': 'e1', 'score_full': 100.0, 'score_ablated': 95.0},
    ]
    critical = find_critical_components(table, threshold_pct=10.0)
    assert critical == []


# ---------- format_ablation_report ----------

def test_format_ablation_report_contains_expected():
    """Report should contain envtype names, component names, and ranking."""
    table = [
        {'component': 'flow', 'delta': 40.0, 'pct_change': 40.0, 'envtype': '1s-d', 'score_full': 100.0, 'score_ablated': 60.0},
        {'component': 'head', 'delta': 10.0, 'pct_change': 10.0, 'envtype': '1s-d', 'score_full': 100.0, 'score_ablated': 90.0},
    ]
    report = format_ablation_report(table)
    assert 'Ablation Report' in report
    assert '1s-d' in report
    assert 'flow' in report
    assert 'head' in report
    assert 'Ranking' in report or 'ranking' in report.lower()
    assert 'Critical' in report or 'critical' in report.lower()


# ---------- Edge cases ----------

def test_empty_components_list():
    """build_ablation_table with empty components returns empty table."""
    table = build_ablation_table(['1s-d'], [], lambda e, k: 100.0)
    assert table == []
    assert rank_components(table) == []
    assert find_critical_components(table) == []
    assert 'No data' in format_ablation_report(table)


def test_single_envtype_single_component():
    """Minimal case: one env, one component."""
    components = [RewardComponent(name='only', weight=1.0, description='the one')]

    def scores_fn(envtype, keep):
        if 'only' in keep:
            return 200.0
        return 120.0

    table = build_ablation_table(['1s-d'], components, scores_fn)
    assert len(table) == 1
    row = table[0]
    assert row['envtype'] == '1s-d'
    assert row['component'] == 'only'
    assert row['score_full'] == 200.0
    assert row['score_ablated'] == 120.0
    assert row['delta'] == pytest.approx(80.0)
    assert row['pct_change'] == pytest.approx(40.0)

    # rank
    assert rank_components(table) == ['only']
    # critical (>10%)
    assert 'only' in find_critical_components(table)
    # not critical (>50%)
    assert 'only' not in find_critical_components(table, threshold_pct=50.0)
