"""Tests for flopyarcade.best_agent_diag (C3 + C11)."""

import math
import pytest

from flopyarcade.best_agent_diag import (
    DiagnosticReport,
    AveragingReport,
    verify_reward_alignment,
    verify_best_agent_consistency,
    verify_sorted_indices,
    verify_averaging,
    run_full_diagnostic,
    diagnose_generation,
)


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestDiagnosticReport:
    def test_defaults(self):
        r = DiagnosticReport()
        assert r.ok is True
        assert r.issues == []
        assert r.checks == 0
        assert r.details == {}

    def test_add_issue(self):
        r = DiagnosticReport()
        r.add_issue("something broke")
        assert r.ok is False
        assert r.issues == ["something broke"]

    def test_add_multiple_issues(self):
        r = DiagnosticReport()
        r.add_issue("a")
        r.add_issue("b")
        assert r.issues == ["a", "b"]
        assert r.ok is False


class TestAveragingReport:
    def test_defaults(self):
        a = AveragingReport()
        assert a.n_runs == 0
        assert a.rewards_per_run == []
        assert a.mean_reward == 0.0
        assert a.std_reward == 0.0
        assert a.cv == 0.0
        assert a.consistent is False
        assert a.max_deviation_pct == 0.0


# ---------------------------------------------------------------------------
# verify_reward_alignment
# ---------------------------------------------------------------------------

class TestVerifyRewardAlignment:
    def test_pass(self):
        rewards = [1.0, 2.0, 3.0]
        report = verify_reward_alignment(rewards, [0, 1], n_agents=3)
        assert report.ok
        assert report.checks == 4

    def test_length_mismatch(self):
        rewards = [1.0, 2.0]
        report = verify_reward_alignment(rewards, [0], n_agents=5)
        assert not report.ok
        assert any("length" in i for i in report.issues)

    def test_index_out_of_range(self):
        rewards = [1.0, 2.0, 3.0]
        report = verify_reward_alignment(rewards, [5], n_agents=3)
        assert not report.ok
        assert any("out of range" in i for i in report.issues)

    def test_negative_index(self):
        rewards = [1.0, 2.0]
        report = verify_reward_alignment(rewards, [-1], n_agents=2)
        assert not report.ok

    def test_duplicate_indices(self):
        rewards = [1.0, 2.0, 3.0]
        report = verify_reward_alignment(rewards, [1, 1], n_agents=3)
        assert not report.ok
        assert any("duplicate" in i for i in report.issues)

    def test_nan_reward(self):
        rewards = [1.0, float('nan'), 3.0]
        report = verify_reward_alignment(rewards, [0], n_agents=3)
        assert not report.ok
        assert any("NaN" in i for i in report.issues)

    def test_inf_reward(self):
        rewards = [1.0, float('inf'), 3.0]
        report = verify_reward_alignment(rewards, [0], n_agents=3)
        assert not report.ok
        assert any("Inf" in i for i in report.issues)

    def test_details(self):
        rewards = [1.0, 2.0, 3.0]
        report = verify_reward_alignment(rewards, [0, 2], n_agents=3)
        assert report.details['n_rewards'] == 3
        assert report.details['n_agents'] == 3
        assert report.details['n_indices'] == 2


# ---------------------------------------------------------------------------
# verify_best_agent_consistency
# ---------------------------------------------------------------------------

class TestVerifyBestAgentConsistency:
    def test_pass(self):
        rewards = [1.0, 5.0, 3.0]
        report = verify_best_agent_consistency(rewards, best_index=1, best_reward=5.0)
        assert report.ok
        assert report.checks == 3

    def test_wrong_reward(self):
        rewards = [1.0, 5.0, 3.0]
        report = verify_best_agent_consistency(rewards, best_index=1, best_reward=4.0)
        assert not report.ok
        assert any("!= max" in i for i in report.issues)

    def test_wrong_index(self):
        rewards = [1.0, 5.0, 3.0]
        report = verify_best_agent_consistency(rewards, best_index=0, best_reward=5.0)
        assert not report.ok
        assert any("best_index" in i for i in report.issues)

    def test_tie_detected(self):
        rewards = [5.0, 5.0, 3.0]
        report = verify_best_agent_consistency(rewards, best_index=0, best_reward=5.0)
        # Ties are noted but not errors
        assert report.ok
        assert report.details.get('tie_count', 1) == 2

    def test_details(self):
        rewards = [2.0, 8.0, 4.0]
        report = verify_best_agent_consistency(rewards, best_index=1, best_reward=8.0)
        assert report.details['best_index'] == 1
        assert report.details['best_reward'] == 8.0
        assert report.details['actual_max'] == 8.0
        assert report.details['actual_max_idx'] == 1


# ---------------------------------------------------------------------------
# verify_sorted_indices
# ---------------------------------------------------------------------------

class TestVerifySortedIndices:
    def test_pass(self):
        rewards = [1.0, 5.0, 3.0, 2.0]
        report = verify_sorted_indices(rewards, [1, 2, 3], n_elites=3)
        assert report.ok
        assert report.checks == 3

    def test_wrong_length(self):
        rewards = [1.0, 5.0, 3.0]
        report = verify_sorted_indices(rewards, [1, 2], n_elites=3)
        assert not report.ok
        assert any("len(sorted" in i for i in report.issues)

    def test_wrong_order(self):
        rewards = [1.0, 5.0, 3.0]
        report = verify_sorted_indices(rewards, [2, 1], n_elites=2)
        assert not report.ok
        assert any("sorted order violated" in i for i in report.issues)

    def test_out_of_range(self):
        rewards = [1.0, 2.0]
        report = verify_sorted_indices(rewards, [5], n_elites=1)
        assert not report.ok
        assert any("out of range" in i for i in report.issues)

    def test_single_element(self):
        rewards = [5.0]
        report = verify_sorted_indices(rewards, [0], n_elites=1)
        assert report.ok

    def test_details(self):
        rewards = [1.0, 5.0, 3.0]
        report = verify_sorted_indices(rewards, [1, 2], n_elites=2)
        assert report.details['n_sorted'] == 2
        assert report.details['rewards_at_idx'] == [5.0, 3.0]


# ---------------------------------------------------------------------------
# verify_averaging (C11)
# ---------------------------------------------------------------------------

class TestVerifyAveraging:
    def test_empty(self):
        r = verify_averaging([])
        assert r.n_runs == 0
        assert r.consistent is False

    def test_single_run(self):
        r = verify_averaging([100.0])
        assert r.n_runs == 1
        assert r.mean_reward == 100.0
        assert r.std_reward == 0.0
        assert r.cv == 0.0
        assert r.consistent is True

    def test_consistent(self):
        r = verify_averaging([100.0, 101.0, 99.0, 100.5])
        assert r.n_runs == 4
        assert r.consistent is True
        assert r.cv < 0.1

    def test_inconsistent(self):
        r = verify_averaging([100.0, 200.0, 50.0])
        assert r.consistent is False
        assert r.cv > 0.1

    def test_zero_mean(self):
        r = verify_averaging([0.0, 0.0])
        assert r.mean_reward == 0.0
        assert r.cv == 0.0
        assert r.consistent is True

    def test_negative_rewards(self):
        r = verify_averaging([-100.0, -101.0, -99.0])
        assert r.mean_reward < 0
        assert r.cv < 0.1
        assert r.consistent is True

    def test_max_deviation(self):
        r = verify_averaging([100.0, 120.0, 80.0])
        assert r.max_deviation_pct == pytest.approx(20.0)

    def test_custom_threshold(self):
        r = verify_averaging([100.0, 105.0, 95.0], cv_threshold=0.01)
        assert r.consistent is False  # CV ~0.04 > 0.01


# ---------------------------------------------------------------------------
# run_full_diagnostic
# ---------------------------------------------------------------------------

class TestRunFullDiagnostic:
    def test_all_pass(self):
        rewards = [1.0, 5.0, 3.0, 2.0]
        report = run_full_diagnostic(
            rewards=rewards,
            best_index=1,
            best_reward=5.0,
            sorted_indices=[1, 2, 3],
            n_elites=3,
        )
        assert report.ok
        assert report.checks >= 10
        assert report.details['alignment'] is not None
        assert report.details['best_agent'] is not None
        assert report.details['sorted'] is not None

    def test_fail_wrong_best(self):
        rewards = [1.0, 5.0, 3.0]
        report = run_full_diagnostic(
            rewards=rewards,
            best_index=0,
            best_reward=1.0,
        )
        assert not report.ok

    def test_no_sorted_indices(self):
        rewards = [1.0, 5.0, 3.0]
        report = run_full_diagnostic(
            rewards=rewards,
            best_index=1,
            best_reward=5.0,
        )
        assert report.ok
        assert 'sorted' not in report.details

    def test_nan_in_rewards(self):
        rewards = [1.0, float('nan'), 3.0]
        report = run_full_diagnostic(
            rewards=rewards,
            best_index=0,
            best_reward=1.0,
        )
        assert not report.ok


# ---------------------------------------------------------------------------
# diagnose_generation
# ---------------------------------------------------------------------------

class TestDiagnoseGeneration:
    def test_ok_generation(self):
        result = diagnose_generation(
            rewards=[10.0, 50.0, 30.0, 20.0],
            sorted_parent_idxs=[1, 2, 3],
            best_agent_reward=50.0,
            n_agents=4,
            n_elites=3,
        )
        assert result['ok'] is True
        assert result['issues'] == []
        assert result['root_cause_hints'] == []

    def test_stale_reward(self):
        result = diagnose_generation(
            rewards=[10.0, 50.0, 30.0, 20.0],
            sorted_parent_idxs=[1],
            best_agent_reward=45.0,  # wrong!
            n_agents=4,
            n_elites=1,
        )
        assert result['ok'] is False
        assert any("!= max" in i for i in result['issues'])
        assert len(result['root_cause_hints']) >= 1

    def test_length_mismatch_hint(self):
        result = diagnose_generation(
            rewards=[10.0, 50.0],
            sorted_parent_idxs=[0, 1],
            best_agent_reward=50.0,
            n_agents=4,  # mismatch!
            n_elites=2,
        )
        assert result['ok'] is False
        assert any("length" in i.lower() for i in result['issues'])

    def test_nan_hint(self):
        result = diagnose_generation(
            rewards=[10.0, float('nan'), 30.0],
            sorted_parent_idxs=[0],
            best_agent_reward=10.0,
            n_agents=3,
            n_elites=1,
        )
        assert result['ok'] is False
        assert any("NaN" in i for i in result['issues'])
        assert any("MODFLOW" in h or "convergence" in h.lower() for h in result['root_cause_hints'])

    def test_checks_run_count(self):
        result = diagnose_generation(
            rewards=[10.0, 50.0, 30.0],
            sorted_parent_idxs=[1, 2],
            best_agent_reward=50.0,
            n_agents=3,
            n_elites=2,
        )
        assert result['checks_run'] >= 10

    def test_empty_sorted(self):
        result = diagnose_generation(
            rewards=[10.0, 50.0],
            sorted_parent_idxs=[],
            best_agent_reward=50.0,
            n_agents=2,
            n_elites=1,
        )
        # best_index defaults to 0 when sorted_parent_idxs is empty
        # rewards[0]=10.0 != 50.0, so this should fail
        assert result['ok'] is False
