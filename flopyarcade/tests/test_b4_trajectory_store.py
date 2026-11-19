"""B4: Tests for trajectory store and ensemble plot."""

import os
import numpy as np
import pytest

from flopyarcade.trajectory_store import TrajectoryStore


def _make_traj(n_segs=2, n_points=10, seed=42):
    rng = np.random.RandomState(seed)
    return {
        'x': [rng.rand(n_points) for _ in range(n_segs)],
        'y': [rng.rand(n_points) for _ in range(n_segs)],
        'z': [],
    }


class TestInit:
    def test_defaults(self):
        s = TrajectoryStore()
        assert s.max_trajectories == 0
        assert s.figsize == (10, 8)
        assert s.colormap == 'plasma'
        assert s.count == 0

    def test_custom(self):
        s = TrajectoryStore(max_trajectories=5, figsize=(4, 3), colormap='viridis')
        assert s.max_trajectories == 5
        assert s.figsize == (4, 3)
        assert s.colormap == 'viridis'


class TestAdd:
    def test_add_single(self):
        s = TrajectoryStore()
        s.add(_make_traj(), reward=100.0, n_steps=50, envtype='1s-d', success=True)
        assert s.count == 1
        assert s.trajectories[0]['reward'] == 100.0
        assert s.trajectories[0]['success'] is True
        assert s.trajectories[0]['envtype'] == '1s-d'

    def test_add_multiple(self):
        s = TrajectoryStore()
        for i in range(5):
            s.add(_make_traj(seed=i), reward=float(i * 10), success=(i > 2))
        assert s.count == 5

    def test_max_trajectories_enforced(self):
        s = TrajectoryStore(max_trajectories=3)
        for i in range(10):
            s.add(_make_traj(seed=i), reward=float(i))
        assert s.count == 3
        # Should keep the last 3 (indices 7, 8, 9)
        assert s.trajectories[0]['reward'] == 7.0

    def test_add_empty(self):
        s = TrajectoryStore()
        s.add({'x': [], 'y': []}, reward=0.0)
        assert s.count == 1


class TestAddFromEnv:
    def test_mock_env(self):
        class MockEnv:
            trajectories = {'x': [np.array([0, 1, 2])], 'y': [np.array([3, 4, 5])], 'z': []}
            rewardTotal = 42.0
            timeSteps = 30
            envtype = '2s-d'
            success = True

        s = TrajectoryStore()
        s.add_from_env(MockEnv())
        assert s.count == 1
        assert s.trajectories[0]['reward'] == 42.0
        assert s.trajectories[0]['n_steps'] == 30
        assert s.trajectories[0]['envtype'] == '2s-d'

    def test_env_with_reward_attr(self):
        class MockEnv:
            trajectories = {'x': [np.array([0, 1])], 'y': [np.array([1, 2])], 'z': []}
            reward = 7.5

        s = TrajectoryStore()
        s.add_from_env(MockEnv())
        assert s.trajectories[0]['reward'] == 7.5


class TestRender:
    def test_render_creates_axes(self):
        s = TrajectoryStore(figsize=(4, 3))
        s.add(_make_traj(), reward=50.0, success=True)
        ax = s.render()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_render_empty_raises(self):
        s = TrajectoryStore()
        with pytest.raises(ValueError, match="No trajectories"):
            s.render()

    def test_render_multiple(self):
        s = TrajectoryStore(figsize=(4, 3))
        for i in range(4):
            s.add(_make_traj(seed=i), reward=float(i * 20), success=(i >= 2))
        ax = s.render(title='Test Ensemble')
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')


class TestSave:
    def test_save_png(self, tmp_path):
        s = TrajectoryStore(figsize=(3, 3))
        s.add(_make_traj(), reward=100.0)
        out = str(tmp_path / 'traj.png')
        result = s.save(out)
        assert os.path.isfile(result)
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_save_json_roundtrip(self, tmp_path):
        s = TrajectoryStore()
        s.add(_make_traj(n_segs=3), reward=55.0, n_steps=20, envtype='3s-d', success=True)
        s.add(_make_traj(seed=1, n_segs=1), reward=10.0, success=False)

        out = str(tmp_path / 'traj.json')
        s.save_json(out)
        assert os.path.isfile(out)

        s2 = TrajectoryStore()
        n = s2.load_json(out)
        assert n == 2
        assert s2.count == 2
        assert s2.trajectories[0]['reward'] == 55.0
        assert s2.trajectories[0]['success'] is True


class TestSummary:
    def test_summary_empty(self):
        s = TrajectoryStore()
        assert s.summary() == {'count': 0}

    def test_summary_populated(self):
        s = TrajectoryStore()
        s.add(_make_traj(), reward=100.0, envtype='1s-d', success=True)
        s.add(_make_traj(seed=1), reward=50.0, envtype='2s-d', success=False)
        s.add(_make_traj(seed=2), reward=75.0, envtype='1s-d', success=True)

        summ = s.summary()
        assert summ['count'] == 3
        assert summ['n_success'] == 2
        assert summ['n_failed'] == 1
        assert summ['reward_min'] == 50.0
        assert summ['reward_max'] == 100.0
        assert '1s-d' in summ['envtypes']
        assert '2s-d' in summ['envtypes']


class TestClear:
    def test_clear_resets(self):
        s = TrajectoryStore()
        s.add(_make_traj(), reward=10.0)
        s.render()
        s.clear()
        assert s.count == 0
        assert s._fig is None
