"""C9: Tests for population pruning and ensemble validation."""

import pytest

from flopyarcade.pruning_ensemble import (
    PopulationPruner,
    PruneResult,
    EnsembleValidator,
    ValidationReport,
    prune_and_validate,
)


# ---------------------------------------------------------------------------
# PopulationPruner
# ---------------------------------------------------------------------------

class TestPopulationPrunerInit:
    def test_defaults(self):
        p = PopulationPruner()
        assert p.keep_fraction == 0.7
        assert p.min_keep == 2

    def test_custom(self):
        p = PopulationPruner(keep_fraction=0.5, min_keep=3)
        assert p.keep_fraction == 0.5
        assert p.min_keep == 3

    def test_invalid_keep_fraction_zero(self):
        with pytest.raises(ValueError, match="keep_fraction"):
            PopulationPruner(keep_fraction=0)

    def test_invalid_keep_fraction_negative(self):
        with pytest.raises(ValueError, match="keep_fraction"):
            PopulationPruner(keep_fraction=-0.5)

    def test_invalid_keep_fraction_above_one(self):
        with pytest.raises(ValueError, match="keep_fraction"):
            PopulationPruner(keep_fraction=1.5)

    def test_invalid_min_keep(self):
        with pytest.raises(ValueError, match="min_keep"):
            PopulationPruner(min_keep=0)


class TestPopulationPrunerPrune:
    def test_empty_population(self):
        p = PopulationPruner()
        result = p.prune([], [])
        assert result.kept_count == 0
        assert result.removed_count == 0
        assert result.surviving_indices == []

    def test_single_agent(self):
        p = PopulationPruner(min_keep=1)
        result = p.prune(["a"], [10.0])
        assert result.kept_count == 1
        assert result.removed_count == 0
        assert result.surviving_indices == [0]

    def test_keeps_top_fraction(self):
        p = PopulationPruner(keep_fraction=0.5, min_keep=1)
        agents = [f"a{i}" for i in range(10)]
        rewards = list(range(10))  # 0..9
        result = p.prune(agents, rewards)
        assert result.kept_count == 5
        # Top 5 rewards: 5,6,7,8,9 → indices 5..9
        assert set(result.surviving_indices) == {5, 6, 7, 8, 9}

    def test_threshold_is_worst_survivor(self):
        p = PopulationPruner(keep_fraction=0.5, min_keep=1)
        agents = ["a"] * 10
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = p.prune(agents, rewards)
        assert result.threshold == 6.0  # worst of top 5

    def test_min_keep_enforced(self):
        p = PopulationPruner(keep_fraction=0.1, min_keep=3)
        agents = ["a"] * 10
        rewards = list(range(10))
        result = p.prune(agents, rewards)
        assert result.kept_count == 3

    def test_length_mismatch_raises(self):
        p = PopulationPruner()
        with pytest.raises(ValueError, match="same length"):
            p.prune(["a", "b"], [1.0])

    def test_surviving_indices_sorted(self):
        p = PopulationPruner(keep_fraction=0.5, min_keep=1)
        agents = ["a"] * 8
        rewards = [3, 1, 4, 1, 5, 9, 2, 6]
        result = p.prune(agents, rewards)
        assert result.surviving_indices == sorted(result.surviving_indices)

    def test_all_removed_when_n_zero(self):
        p = PopulationPruner(keep_fraction=0.1, min_keep=1)
        result = p.prune([], [])
        assert result.removed_count == 0


class TestPruneAgents:
    def test_returns_surviving_and_result(self):
        p = PopulationPruner(keep_fraction=0.5, min_keep=1)
        agents = ["a0", "a1", "a2", "a3"]
        rewards = [1.0, 2.0, 3.0, 4.0]
        surviving, result = p.prune_agents(agents, rewards)
        assert surviving == ["a2", "a3"]
        assert result.kept_count == 2

    def test_all_survive(self):
        p = PopulationPruner(keep_fraction=1.0, min_keep=1)
        agents = ["x", "y"]
        rewards = [1.0, 2.0]
        surviving, result = p.prune_agents(agents, rewards)
        assert len(surviving) == 2


# ---------------------------------------------------------------------------
# EnsembleValidator
# ---------------------------------------------------------------------------

class TestEnsembleValidatorInit:
    def test_defaults(self):
        v = EnsembleValidator()
        assert v.n_episodes == 10
        assert v.seed is None
        assert v.viability_threshold == 0.0

    def test_custom(self):
        v = EnsembleValidator(n_episodes=5, seed=42, viability_threshold=1.0)
        assert v.n_episodes == 5
        assert v.seed == 42
        assert v.viability_threshold == 1.0

    def test_invalid_n_episodes(self):
        with pytest.raises(ValueError, match="n_episodes"):
            EnsembleValidator(n_episodes=0)


class TestValidateWithRewards:
    def test_basic_stats(self):
        v = EnsembleValidator(viability_threshold=0.0)
        rewards = [10.0, 20.0, 30.0, 40.0, 50.0]
        report = v.validate_with_rewards(rewards)
        assert report.n_episodes == 5
        assert report.mean_reward == 30.0
        assert report.min_reward == 10.0
        assert report.max_reward == 50.0
        assert report.median_reward == 30.0
        assert report.passed is True
        assert report.pass_rate == 1.0

    def test_single_episode(self):
        v = EnsembleValidator()
        report = v.validate_with_rewards([5.0])
        assert report.n_episodes == 1
        assert report.std_reward == 0.0
        assert report.mean_reward == 5.0

    def test_empty_rewards(self):
        v = EnsembleValidator()
        report = v.validate_with_rewards([])
        assert report.n_episodes == 0
        assert report.passed is False

    def test_negative_rewards(self):
        v = EnsembleValidator(viability_threshold=0.0)
        report = v.validate_with_rewards([-5.0, -3.0, -1.0])
        assert report.passed is False
        assert report.pass_rate == 0.0

    def test_mixed_pass_fail(self):
        v = EnsembleValidator(viability_threshold=0.0)
        rewards = [1.0, -1.0, 2.0, -2.0]
        report = v.validate_with_rewards(rewards)
        assert report.pass_rate == 0.5
        # mean = 0.0, threshold = 0.0, so passed (>=)
        assert report.passed is True

    def test_std_computation(self):
        v = EnsembleValidator()
        rewards = [1.0, 2.0, 3.0]
        report = v.validate_with_rewards(rewards)
        import statistics
        expected_std = statistics.stdev([1.0, 2.0, 3.0])
        assert abs(report.std_reward - expected_std) < 1e-9


class TestValidateWithEnv:
    def test_env_validation(self):
        """Use a mock env/agent to test the validate() method."""

        class MockEnv:
            def __init__(self, episodes_reward):
                self.episodes_reward = list(episodes_reward)
                self._ep = 0

            def reset(self, seed=None):
                self._ep += 1
                return (0,), {}

            def step(self, action):
                reward = self.episodes_reward[min(self._ep - 1, len(self.episodes_reward) - 1)]
                return (0,), reward, True, {}, {}

        class MockAgent:
            def act(self, obs):
                return 0

        v = EnsembleValidator(n_episodes=3, seed=42)
        env = MockEnv([10.0, 20.0, 30.0])
        agent = MockAgent()
        report = v.validate(env, agent)
        assert report.n_episodes == 3
        assert report.mean_reward == 20.0
        assert report.passed is True


# ---------------------------------------------------------------------------
# prune_and_validate pipeline
# ---------------------------------------------------------------------------

class TestPruneAndValidate:
    def test_without_env(self):
        agents = ["a0", "a1", "a2", "a3"]
        rewards = [1.0, 2.0, 3.0, 4.0]
        result = prune_and_validate(agents, rewards)
        assert "prune_result" in result
        assert result["best_index"] == 3
        assert result["best_reward"] == 4.0
        assert "validation" not in result

    def test_with_env(self):
        class MockEnv:
            def reset(self, seed=None):
                return (0,), {}
            def step(self, action):
                return (0,), 5.0, True, {}, {}

        class MockAgent:
            def act(self, obs):
                return 0

        agents = [MockAgent(), MockAgent(), MockAgent()]
        rewards = [1.0, 2.0, 3.0]
        env = MockEnv()
        result = prune_and_validate(agents, rewards, env=env)
        assert "validation" in result
        assert result["validation"].n_episodes == 10  # default
        assert result["best_index"] == 2

    def test_empty_population(self):
        result = prune_and_validate([], [])
        assert result["best_index"] == -1
        assert result["best_reward"] == 0.0

    def test_custom_pruner_validator(self):
        pruner = PopulationPruner(keep_fraction=1.0, min_keep=1)
        validator = EnsembleValidator(n_episodes=2)
        agents = ["a"]
        rewards = [10.0]
        result = prune_and_validate(agents, rewards, pruner=pruner, validator=validator)
        assert result["best_index"] == 0
        assert result["best_reward"] == 10.0
