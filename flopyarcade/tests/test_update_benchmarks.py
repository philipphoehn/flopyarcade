"""B12: Tests for auto-update benchmark figures on website."""

import json
import os
import tempfile
import pytest

from flopyarcade.update_benchmarks import (
    load_benchmark_results,
    plot_operator_scores,
    plot_average_evolutions,
    update_benchmarks,
)


SAMPLE_RESULTS = [
    {
        'envtype': '1s-d',
        'model': 'model_1s-d.p',
        'n_games': 1000,
        'seed_base': 42,
        'mean_reward': 186.61,
        'std_reward': 326.3,
        'min_reward': -247.15,
        'max_reward': 982.57,
        'median_reward': 0.0,
        'pct_nonzero': 25.0,
        'avg_runtime_s': 1.197,
        'total_runtime_s': 1196.9,
    },
    {
        'envtype': '2s-d',
        'model': 'model_2s-d.p',
        'n_games': 1000,
        'seed_base': 42,
        'mean_reward': 95.2,
        'std_reward': 200.1,
        'min_reward': -150.0,
        'max_reward': 800.0,
        'median_reward': 0.0,
        'pct_nonzero': 18.0,
        'avg_runtime_s': 2.5,
        'total_runtime_s': 2500.0,
    },
    {
        'envtype': '3s-d',
        'model': 'model_3s-d.p',
        'n_games': 1000,
        'seed_base': 42,
        'mean_reward': 45.8,
        'std_reward': 150.0,
        'min_reward': -100.0,
        'max_reward': 600.0,
        'median_reward': 0.0,
        'pct_nonzero': 12.0,
        'avg_runtime_s': 3.1,
        'total_runtime_s': 3100.0,
    },
]


def _write_sample_json(path):
    with open(path, 'w') as f:
        json.dump(SAMPLE_RESULTS, f)
    return path


class TestLoadBenchmarkResults:
    def test_load_valid_json(self, tmp_path):
        p = str(tmp_path / 'results.json')
        _write_sample_json(p)
        results = load_benchmark_results(p)
        assert len(results) == 3
        assert results[0]['envtype'] == '1s-d'

    def test_load_missing_file_raises(self, tmp_path):
        p = str(tmp_path / 'nonexistent.json')
        with pytest.raises(FileNotFoundError, match='not found'):
            load_benchmark_results(p)

    def test_load_non_list_raises(self, tmp_path):
        p = str(tmp_path / 'bad.json')
        with open(p, 'w') as f:
            json.dump({'key': 'value'}, f)
        with pytest.raises(ValueError, match='Expected a JSON list'):
            load_benchmark_results(p)

    def test_load_empty_list(self, tmp_path):
        p = str(tmp_path / 'empty.json')
        with open(p, 'w') as f:
            json.dump([], f)
        results = load_benchmark_results(p)
        assert results == []


class TestPlotOperatorScores:
    def test_creates_png_file(self, tmp_path):
        out = str(tmp_path / 'out' / 'scores.png')
        result = plot_operator_scores(SAMPLE_RESULTS, out)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_creates_parent_dirs(self, tmp_path):
        out = str(tmp_path / 'a' / 'b' / 'c' / 'scores.png')
        plot_operator_scores(SAMPLE_RESULTS, out)
        assert os.path.exists(out)

    def test_single_result(self, tmp_path):
        out = str(tmp_path / 'single.png')
        plot_operator_scores(SAMPLE_RESULTS[:1], out)
        assert os.path.exists(out)


class TestPlotAverageEvolutions:
    def test_creates_png_file(self, tmp_path):
        out = str(tmp_path / 'out' / 'evol.png')
        result = plot_average_evolutions(SAMPLE_RESULTS, out)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_missing_optional_fields(self, tmp_path):
        """Results without min/median/max still render."""
        minimal = [{'envtype': '1s-d', 'mean_reward': 100.0}]
        out = str(tmp_path / 'minimal.png')
        plot_average_evolutions(minimal, out)
        assert os.path.exists(out)


class TestUpdateBenchmarks:
    def test_full_pipeline(self, tmp_path):
        results_path = str(tmp_path / 'results.json')
        _write_sample_json(results_path)
        out_dir = str(tmp_path / 'figures')

        paths = update_benchmarks(results_path, out_dir)

        assert 'operator_scores' in paths
        assert 'average_evolutions' in paths
        assert os.path.exists(paths['operator_scores'])
        assert os.path.exists(paths['average_evolutions'])

    def test_output_filenames(self, tmp_path):
        results_path = str(tmp_path / 'results.json')
        _write_sample_json(results_path)
        out_dir = str(tmp_path / 'figs')

        paths = update_benchmarks(results_path, out_dir)
        assert os.path.basename(paths['operator_scores']) == 'benchmarks_operatorScores.png'
        assert os.path.basename(paths['average_evolutions']) == 'benchmarks_averageEvolutions.png'


class TestMainCLI:
    def test_main_with_args(self, tmp_path, capsys, monkeypatch):
        results_path = str(tmp_path / 'results.json')
        _write_sample_json(results_path)
        out_dir = str(tmp_path / 'out')

        import sys
        monkeypatch.setattr(sys, 'argv', [
            'update_benchmarks',
            '--results', results_path,
            '--output-dir', out_dir,
        ])

        from flopyarcade.update_benchmarks import main
        main()

        captured = capsys.readouterr()
        assert 'operator_scores' in captured.out
        assert 'average_evolutions' in captured.out
        assert 'Done.' in captured.out

    def test_default_args(self, tmp_path, capsys, monkeypatch):
        """main() with no args should parse without error."""
        import sys
        monkeypatch.setattr(sys, 'argv', ['update_benchmarks', '--help'])

        from flopyarcade.update_benchmarks import main
        with pytest.raises(SystemExit):
            main()
