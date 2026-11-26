"""Tests for flopyarcade.mlflow_integration."""

import json
import os
import tempfile

import pytest

from flopyarcade.mlflow_integration import (
    MetricRecorder,
    MLflowLogger,
    TensorBoardWriter,
    TrackingConfig,
    compare_runs,
    format_run_report,
)


# ---------------------------------------------------------------------------
# TrackingConfig
# ---------------------------------------------------------------------------

def test_tracking_config_defaults():
    cfg = TrackingConfig()
    assert cfg.experiment_name == 'flopyarcade'
    assert cfg.run_name == ''
    assert cfg.tracking_uri == 'file:./mlruns'
    assert cfg.tensorboard_dir == './tb_logs'


def test_tracking_config_custom():
    cfg = TrackingConfig(experiment_name='my_exp', run_name='run1')
    assert cfg.experiment_name == 'my_exp'
    assert cfg.run_name == 'run1'


# ---------------------------------------------------------------------------
# MetricRecorder
# ---------------------------------------------------------------------------

def test_metric_recorder_init():
    rec = MetricRecorder()
    assert rec.params() == {}
    assert rec.metrics() == {}
    assert rec.artifacts() == []
    assert rec.tags() == {}


def test_log_param_stores():
    rec = MetricRecorder()
    rec.log_param('lr', 0.001)
    rec.log_param('batch_size', 64)
    assert rec.params() == {'lr': 0.001, 'batch_size': 64}


def test_log_metric_stores_at_step():
    rec = MetricRecorder()
    rec.log_metric(0, 'reward', 0.5)
    rec.log_metric(10, 'reward', 0.8)
    metrics = rec.metrics()
    assert metrics['reward'] == [(0, 0.5), (10, 0.8)]


def test_log_artifact_appends():
    rec = MetricRecorder()
    rec.log_artifact('models/best.pt')
    rec.log_artifact('logs/train.log')
    assert rec.artifacts() == ['models/best.pt', 'logs/train.log']


def test_set_tag_and_tags():
    rec = MetricRecorder()
    rec.set_tag('framework', 'flopyarcade')
    rec.set_tag('version', '0.7.15')
    assert rec.tags() == {'framework': 'flopyarcade', 'version': '0.7.15'}


def test_multiple_metrics_same_key_different_steps():
    rec = MetricRecorder()
    rec.log_metric(0, 'loss', 1.0)
    rec.log_metric(100, 'loss', 0.5)
    rec.log_metric(200, 'loss', 0.25)
    pts = rec.metrics()['loss']
    assert len(pts) == 3
    assert pts[0] == (0, 1.0)
    assert pts[1] == (100, 0.5)
    assert pts[2] == (200, 0.25)


def test_empty_recorder_no_metrics_logged():
    rec = MetricRecorder()
    assert rec.metrics() == {}
    assert rec.params() == {}
    assert rec.artifacts() == []
    assert rec.tags() == {}


# ---------------------------------------------------------------------------
# MLflowLogger
# ---------------------------------------------------------------------------

def test_mlflow_logger_start_run_returns_id():
    rec = MetricRecorder()
    logger = MLflowLogger(rec)
    run_id = logger.start_run()
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_mlflow_logger_end_run_sets_status():
    rec = MetricRecorder()
    logger = MLflowLogger(rec)
    logger.start_run()
    logger.end_run(status='FINISHED')
    assert logger._status == 'FINISHED'


def test_mlflow_logger_export_json_structure():
    rec = MetricRecorder()
    rec.log_param('lr', 0.01)
    rec.log_metric(0, 'reward', 0.5)
    rec.log_artifact('model.pt')
    rec.set_tag('env', 'flopy')

    logger = MLflowLogger(rec)
    logger.start_run()
    logger.end_run()

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name
    logger.export_json(path)
    try:
        with open(path) as f:
            data = json.load(f)
        assert 'run_id' in data
        assert 'params' in data
        assert 'metrics' in data
        assert 'artifacts' in data
        assert 'tags' in data
        assert 'status' in data
        assert 'timestamps' in data
        assert data['params']['lr'] == 0.01
        assert data['artifacts'] == ['model.pt']
        assert data['tags']['env'] == 'flopy'
        assert data['status'] == 'FINISHED'
        assert 'started_at' in data['timestamps']
        assert 'ended_at' in data['timestamps']
    finally:
        os.unlink(path)


def test_mlflow_logger_export_tensorboard_format():
    rec = MetricRecorder()
    rec.log_metric(0, 'reward', 0.5)
    rec.log_metric(10, 'reward', 0.7)
    rec.log_metric(0, 'loss', 1.2)

    logger = MLflowLogger(rec)
    logger.start_run()
    logger.end_run()

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        path = f.name
    logger.export_tensorboard(path)
    try:
        with open(path) as f:
            lines = f.read().strip().split('\n')
        # Each line should be tab-separated: step\tkey\tvalue
        for line in lines:
            parts = line.split('\t')
            assert len(parts) == 3, f"Expected 3 tab-separated fields, got {parts}"
            int(parts[0])  # step is int
            float(parts[2])  # value is float
        assert len(lines) == 3
    finally:
        os.unlink(path)


def test_mlflow_logger_summary():
    rec = MetricRecorder()
    rec.log_param('a', 1)
    rec.log_param('b', 2)
    rec.log_metric(0, 'm1', 0.5)
    rec.log_metric(1, 'm1', 0.6)
    rec.log_artifact('x.pt')

    logger = MLflowLogger(rec)
    logger.start_run()
    s = logger.summary()
    assert s['n_params'] == 2
    assert s['n_metrics'] == 2
    assert s['n_artifacts'] == 1
    assert s['status'] == 'RUNNING'


# ---------------------------------------------------------------------------
# TensorBoardWriter
# ---------------------------------------------------------------------------

def test_tensorboard_writer_write_scalar():
    writer = TensorBoardWriter('/tmp/tb_test')
    writer.write_scalar('loss', 0.9, 0)
    writer.write_scalar('loss', 0.5, 100)
    writer.write_scalar('acc', 0.8, 0)
    assert writer.n_entries() == 3


def test_tensorboard_writer_flush_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(tmpdir)
        writer.write_scalar('loss', 0.9, 0)
        writer.write_scalar('loss', 0.5, 100)
        writer.flush()
        # Should create loss.txt
        expected = os.path.join(tmpdir, 'loss.txt')
        assert os.path.exists(expected)
        with open(expected) as f:
            content = f.read()
        lines = content.strip().split('\n')
        assert len(lines) == 2
        for line in lines:
            parts = line.split('\t')
            assert len(parts) == 3


def test_tensorboard_writer_summary():
    writer = TensorBoardWriter('/tmp/tb_sum')
    writer.write_scalar('loss', 0.9, 0)
    writer.write_scalar('acc', 0.7, 0)
    s = writer.summary()
    assert s['n_tags'] == 2
    assert s['n_entries'] == 2
    assert s['log_dir'] == '/tmp/tb_sum'


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

def test_compare_runs_structure():
    run_a = {
        'params': {'lr': 0.01, 'bs': 32},
        'metrics': {'reward': [[0, 0.5], [10, 0.8]]},
        'artifacts': [],
        'tags': {},
    }
    run_b = {
        'params': {'lr': 0.001, 'bs': 32},
        'metrics': {'reward': [[0, 0.3], [10, 0.6]]},
        'artifacts': [],
        'tags': {},
    }
    result = compare_runs(run_a, run_b)
    assert 'param_diffs' in result
    assert 'metric_diffs' in result
    assert 'winner' in result


def test_compare_runs_metric_diffs():
    run_a = {
        'params': {},
        'metrics': {'reward': [[0, 0.5], [10, 0.8]]},
    }
    run_b = {
        'params': {},
        'metrics': {'reward': [[0, 0.3], [10, 0.6]]},
    }
    result = compare_runs(run_a, run_b)
    assert 'reward' in result['metric_diffs']
    md = result['metric_diffs']['reward']
    assert md['mean_a'] == pytest.approx(0.65)
    assert md['mean_b'] == pytest.approx(0.45)
    assert md['diff'] == pytest.approx(-0.2)
    # Winner should be 'a' since final reward a (0.8) > b (0.6)
    assert result['winner'] == 'a'


def test_compare_runs_param_diffs():
    run_a = {'params': {'lr': 0.01}, 'metrics': {}}
    run_b = {'params': {'lr': 0.001}, 'metrics': {}}
    result = compare_runs(run_a, run_b)
    assert 'lr' in result['param_diffs']
    assert result['param_diffs']['lr']['a'] == 0.01
    assert result['param_diffs']['lr']['b'] == 0.001


# ---------------------------------------------------------------------------
# format_run_report
# ---------------------------------------------------------------------------

def test_format_run_report_contains_text():
    rec = MetricRecorder()
    rec.log_param('lr', 0.01)
    rec.log_metric(0, 'reward', 0.5)
    rec.log_artifact('model.pt')
    rec.set_tag('env', 'flopy')

    report = format_run_report(rec)
    assert 'FloPyArcade' in report
    assert 'lr' in report
    assert 'reward' in report
    assert 'model.pt' in report
    assert 'flopy' in report
    assert 'Params' in report
    assert 'Metrics' in report
    assert 'Artifacts' in report
    assert 'Tags' in report
