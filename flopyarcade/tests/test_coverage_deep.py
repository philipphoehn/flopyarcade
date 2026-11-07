"""Deep coverage tests for agent.py, render.py, play.py.
Uses real installed dependencies. Only mocks ray.rllib.agents (missing in this Ray version)."""
import pytest
import numpy as np
import os
import sys
import types
from unittest.mock import MagicMock, patch
from collections import deque

# Only mock what's actually missing
if 'ray.rllib.agents' not in sys.modules:
    try:
        import ray.rllib.agents
    except ImportError:
        _ra = types.ModuleType('ray.rllib.agents')
        sys.modules['ray.rllib.agents'] = _ra
        for _sub in ['dqn', 'ppo', 'appo', 'impala']:
            _m = types.ModuleType(f'ray.rllib.agents.{_sub}')
            _m.__getattr__ = lambda name: MagicMock()
            sys.modules[f'ray.rllib.agents.{_sub}'] = _m
            setattr(_ra, _sub, _m)


# === PLAY.PY ===

class TestPlayModule:
    def test_str_to_bool_true(self):
        from flopyarcade.play import str_to_bool
        for v in [True, 'true', 'True', 't', 'T', '1', 'yes', 'y', 'Y']:
            assert str_to_bool(v) is True

    def test_str_to_bool_false(self):
        from flopyarcade.play import str_to_bool
        for v in [False, 'false', 'False', 'f', 'F', '0', 'no', 'n', 'N']:
            assert str_to_bool(v) is False

    def test_str_to_bool_invalid(self):
        from flopyarcade.play import str_to_bool
        for v in ['maybe', '2', 'xyz', '']:
            with pytest.raises(ValueError):
                str_to_bool(v)

    def test_has_arguments_all_defaults(self):
        import argparse
        from flopyarcade.play import has_arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--foo', default=42)
        parser.add_argument('--bar', default='hello')
        args = parser.parse_args([])
        assert has_arguments(parser, args) is False

    def test_has_arguments_one_changed(self):
        import argparse
        from flopyarcade.play import has_arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--foo', default=42)
        args = parser.parse_args(['--foo', '99'])
        assert has_arguments(parser, args) is True

    def test_has_arguments_none_changed_multi(self):
        import argparse
        from flopyarcade.play import has_arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--a', default=1)
        parser.add_argument('--b', default=2)
        parser.add_argument('--c', default=3)
        args = parser.parse_args([])
        assert has_arguments(parser, args) is False


# === AGENT.PY (biggest gap at 28%) ===

class TestFloPyAgentLogic:
    """Test pure-logic methods of FloPyAgent (random mode)."""

    @pytest.fixture
    def agent(self):
        from flopyarcade.agent import FloPyAgent
        a = FloPyAgent(
            observationsVector=np.random.rand(10),
            actionSpace=['keep','up','down','left','right'],
            mode='random',
        )
        yield a

    def test_get_action_type_discrete(self, agent):
        assert agent.getActionType('1s-d') == 'discrete'
        assert agent.getActionType('3s-d') == 'discrete'
        assert agent.getActionType('2r-d') == 'discrete'

    def test_get_action_type_continuous(self, agent):
        assert agent.getActionType('1s-c') == 'continuous'
        assert agent.getActionType('5r-c') == 'continuous'

    def test_get_workspace(self, agent):
        ws = agent._get_workspace()
        assert isinstance(ws, str) and len(ws) > 0

    def test_create_folder(self, agent):
        import tempfile, shutil
        p = os.path.join(tempfile.gettempdir(), 'a0_cov_test_xyz')
        agent._create_folder(p, assign_attr='test_attr')
        assert os.path.exists(p)
        assert agent.test_attr == p
        shutil.rmtree(p, ignore_errors=True)

    def test_create_folder_no_assign(self, agent):
        import tempfile, shutil
        p = os.path.join(tempfile.gettempdir(), 'a0_cov_test_xyz2')
        agent._create_folder(p)
        assert os.path.exists(p)
        shutil.rmtree(p, ignore_errors=True)

    def test_update_mutation_history(self, agent):
        r = agent.updateMutationHistory({}, 1, 12345, mutationSeeds=[], mutationSeed=None)
        assert r['agent1']['creationSeed'] == 12345
        assert r['agent1']['mutationSeeds'] == []

    def test_update_mutation_history_with_seed(self, agent):
        r = agent.updateMutationHistory({}, 2, 999, mutationSeeds=[1,2], mutationSeed=3)
        assert r['agent2']['mutationSeeds'] == [1,2,3]

    def test_create_mutation_record(self, agent):
        rec = agent.createMutationRecord(42, mutationSeeds=[], mutationSeed=None)
        assert rec['creationSeed'] == 42
        assert rec['mutationSeeds'] == []

    def test_create_mutation_record_with_seed(self, agent):
        rec = agent.createMutationRecord(77, mutationSeeds=[1], mutationSeed=2)
        assert rec['mutationSeeds'] == [1, 2]

    def test_check_parameter_mean(self, agent):
        mock = MagicMock()
        mock.get_weights.return_value = [np.array([1.0,2.0,3.0]), np.array([0.5,0.5])]
        result = agent.checkParameterMean(mock)
        expected = np.mean([np.mean([1,2,3]), np.mean([0.5,0.5])])
        assert abs(result - expected) < 1e-10

    def test_update_replay_memory(self, agent):
        agent.replayMemory = deque(maxlen=100)
        agent.updateReplayMemory((np.array([1]), 0, 1.0, np.array([2])))
        assert len(agent.replayMemory) == 1

    def test_update_replay_memory_zero_reward(self, agent):
        agent.replayMemory = deque(maxlen=100)
        t1 = [np.array([1]), 0, 1.0, np.array([2])]
        t2 = [np.array([3]), 1, 2.0, np.array([4])]
        agent.replayMemory.extend([t1, t2])
        agent.updateReplayMemoryZeroReward(2)
        assert t1[2] == 0.0 and t2[2] == 0.0

    def test_set_seeds(self, agent):
        agent.envSettings = {'SEEDAGENT': 42}
        agent.setSeeds()
        assert agent.SEED == 42

    def test_yield_chunks(self, agent):
        chunks = list(agent.yieldChunks(list(range(10)), 3))
        assert sum(len(c) for c in chunks) == 10
        assert len(chunks) == 4

    def test_yield_chunks_exact_division(self, agent):
        chunks = list(agent.yieldChunks(list(range(9)), 3))
        assert sum(len(c) for c in chunks) == 9
        assert len(chunks) == 3

    def test_yield_chunks_single(self, agent):
        chunks = list(agent.yieldChunks([1,2,3], 10))
        assert len(chunks) == 1
        assert chunks[0] == [1,2,3]

    def test_pickle_dump_and_load(self, agent):
        import tempfile
        p = os.path.join(tempfile.gettempdir(), 'a0_cov_pickle_test.pkl')
        data = {'key': [1, 2, 3], 'nested': {'a': 1}}
        agent.pickleDump(p, data)
        result = agent.pickleLoad(p)
        assert result == data
        os.remove(p)

    def test_count_parameters(self, agent):
        mock_model = MagicMock()
        mock_model.get_weights.return_value = [np.zeros((2,3)), np.zeros((3,))]
        count = agent.countParameters(mock_model)
        assert count == 9

    def test_mutate_decisions(self, agent):
        shape = (5, 4)
        result = agent.mutateDecisions(0.3, shape)
        assert result is not None

    def test_action_novelty_metric_discrete(self, agent):
        a1 = ['up', 'down', 'left', 'right', 'keep']
        a2 = ['up', 'up', 'down', 'left', 'keep']
        result = agent.actionNoveltyMetric(a1, a2, 'discrete')
        assert result is not None
        assert result >= 0

    def test_action_novelty_metric_same(self, agent):
        a1 = ['up', 'down', 'left', 'right', 'keep']
        result = agent.actionNoveltyMetric(a1, a1, 'discrete')
        assert result < 1e-10

    def test_decay_epsilon(self, agent):
        agent.epsilon = 1.0
        agent.hyParams = {'EPSILONDECAY': 0.9, 'EPSILONMIN': 0.01}
        agent._decay_epsilon()
        assert agent.epsilon == pytest.approx(0.9)

    def test_get_action_random(self, agent):
        action = agent.getAction(mode='random')
        assert action in agent.actionSpace

    def test_get_action_random_multiple(self, agent):
        actions = [agent.getAction(mode='random') for _ in range(10)]
        assert all(a in agent.actionSpace for a in actions)

    def test_init_seeds(self, agent):
        agent.hyParams = {'NAGENTS': 2, 'NGENERATIONS': 2, 'NGAMESAVERAGED': 2}
        agent.envSettings = {'SEEDSRANDOM': True}
        agent._initialize_seeds()
        assert hasattr(agent, 'mutationSeeds')


# === RENDER.PY (35% gap) ===

@pytest.mark.slow

class TestEnvEdge:
    @pytest.fixture
    def env(self):
        from flopyarcade.flopyarcade import FloPyEnv
        e = FloPyEnv(ENVTYPE='1s-d', flagRender=False)
        yield e
        e.close()

    def test_reset_returns_obs(self, env):
        obs = env.reset(seed=123)
        assert obs is not None

    def test_step_keep(self, env):
        env.reset(seed=42)
        result = env.step('keep')
        assert len(result) == 5
        obs, rew, done, trunc, info = result
        assert obs is not None

    def test_step_up(self, env):
        env.reset(seed=42)
        obs, rew, done, trunc, info = env.step('up')
        assert obs is not None

    def test_step_down(self, env):
        env.reset(seed=42)
        obs, rew, done, trunc, info = env.step('down')
        assert obs is not None

    def test_multiple_steps(self, env):
        env.reset(seed=42)
        for i in range(5):
            obs, rew, done, trunc, info = env.step('keep')
            assert obs is not None
            if done or trunc:
                break

    def test_different_envtypes(self):
        for et in ['1s-d', '2s-d']:
            from flopyarcade.flopyarcade import FloPyEnv
            e = FloPyEnv(ENVTYPE=et, flagRender=False)
            obs = e.reset(seed=42)
            assert obs is not None
            e.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--timeout=180'])
