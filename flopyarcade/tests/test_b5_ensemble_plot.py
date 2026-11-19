"""B5: Tests for generational trajectory ensemble plot."""

import os
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flopyarcade.ensemble_plot import GenerationalEnsemblePlot


def _make_trajectory(n_segs=2, n_points=10, seed=42):
    """Create a synthetic trajectory dict."""
    rng = np.random.RandomState(seed)
    x_segs = [rng.rand(n_points) for _ in range(n_segs)]
    y_segs = [rng.rand(n_points) for _ in range(n_segs)]
    return {'x': x_segs, 'y': y_segs}


class TestInit:
    def test_defaults(self):
        p = GenerationalEnsemblePlot()
        assert p.figsize == (10, 8)
        assert p.colormap == 'viridis'
        assert p.alpha_range == (0.3, 1.0)
        assert p.linewidth_range == (0.5, 2.0)
        assert p.show_best is True
        assert p.n_generations == 0

    def test_custom_params(self):
        p = GenerationalEnsemblePlot(figsize=(5, 5), colormap='plasma',
                                     alpha_range=(0.1, 0.9), linewidth_range=(1, 3),
                                     show_best=False)
        assert p.figsize == (5, 5)
        assert p.colormap == 'plasma'
        assert p.alpha_range == (0.1, 0.9)
        assert p.linewidth_range == (1, 3)
        assert p.show_best is False


class TestAddGeneration:
    def test_add_single(self):
        p = GenerationalEnsemblePlot()
        p.add_generation(_make_trajectory(), reward=0.5)
        assert p.n_generations == 1
        assert p.generations[0]['reward'] == 0.5

    def test_add_multiple(self):
        p = GenerationalEnsemblePlot()
        for i in range(5):
            p.add_generation(_make_trajectory(seed=i), reward=float(i))
        assert p.n_generations == 5

    def test_segments_stored(self):
        p = GenerationalEnsemblePlot()
        p.add_generation(_make_trajectory(n_segs=3), reward=1.0)
        assert p.generations[0]['n_segments'] == 3

    def test_empty_trajectory(self):
        p = GenerationalEnsemblePlot()
        p.add_generation({'x': [], 'y': []}, reward=0.0)
        assert p.n_generations == 1
        assert p.generations[0]['n_segments'] == 0


class TestAddFromEnv:
    def test_mock_env(self):
        class MockEnv:
            trajectories = {'x': [np.array([1, 2, 3])], 'y': [np.array([4, 5, 6])]}
            reward = 0.99

        p = GenerationalEnsemblePlot()
        p.add_from_env(MockEnv())
        assert p.n_generations == 1
        assert p.generations[0]['reward'] == 0.99


class TestRender:
    def test_render_creates_axes(self):
        p = GenerationalEnsemblePlot()
        p.add_generation(_make_trajectory(), reward=1.0)
        ax = p.render()
        assert ax is not None
        plt.close('all')

    def test_render_empty_raises(self):
        p = GenerationalEnsemblePlot()
        with pytest.raises(ValueError, match="No generations"):
            p.render()

    def test_render_custom_title(self):
        p = GenerationalEnsemblePlot()
        p.add_generation(_make_trajectory(), reward=0.5)
        ax = p.render(title='My Custom Title')
        assert 'My Custom Title' in ax.get_title()
        plt.close('all')

    def test_render_multiple_generations(self):
        p = GenerationalEnsemblePlot()
        for i in range(3):
            p.add_generation(_make_trajectory(seed=i), reward=float(i))
        ax = p.render()
        assert ax is not None
        plt.close('all')


class TestSave:
    def test_save_png(self, tmp_path):
        p = GenerationalEnsemblePlot(figsize=(4, 3))
        p.add_generation(_make_trajectory(), reward=0.5)
        out = str(tmp_path / 'ensemble.png')
        result = p.save(out)
        assert os.path.isfile(result)
        assert result == out
        plt.close('all')

    def test_save_auto_renders(self, tmp_path):
        p = GenerationalEnsemblePlot(figsize=(3, 3))
        p.add_generation(_make_trajectory(), reward=1.0)
        out = str(tmp_path / 'auto.png')
        p.save(out)
        assert os.path.isfile(out)
        plt.close('all')


class TestClear:
    def test_clear_resets(self):
        p = GenerationalEnsemblePlot()
        p.add_generation(_make_trajectory(), reward=0.5)
        p.render()
        p.clear()
        assert p.n_generations == 0
        assert p._fig is None
        assert p._ax is None


class TestShowBest:
    def test_best_highlighted(self):
        p = GenerationalEnsemblePlot(show_best=True)
        p.add_generation(_make_trajectory(seed=0), reward=0.1)
        p.add_generation(_make_trajectory(seed=1), reward=0.9)  # best
        p.add_generation(_make_trajectory(seed=2), reward=0.5)
        ax = p.render()
        # Best should have a legend entry
        handles, labels = ax.get_legend_handles_labels()
        assert len(handles) >= 1
        plt.close('all')

    def test_no_best_legend(self):
        p = GenerationalEnsemblePlot(show_best=False)
        p.add_generation(_make_trajectory(seed=0), reward=0.1)
        p.add_generation(_make_trajectory(seed=1), reward=0.9)
        ax = p.render()
        handles, labels = ax.get_legend_handles_labels()
        assert len(handles) == 0
        plt.close('all')
