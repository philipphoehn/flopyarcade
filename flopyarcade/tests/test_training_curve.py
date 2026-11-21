"""Tests for flopyarcade.training_curve"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flopyarcade.training_curve import (
    StepMetric,
    TrainingCurveLogger,
    compare_curves,
    format_curve_report,
)


# 1. StepMetric creation with all optional fields
def test_stepmetric_creation():
    m = StepMetric(step=1, loss=0.5, reward=1.2, epsilon=0.9, extra={'lr': 0.001})
    assert m.step == 1
    assert m.loss == 0.5
    assert m.reward == 1.2
    assert m.epsilon == 0.9
    assert m.extra == {'lr': 0.001}

    m2 = StepMetric(step=0)
    assert m2.step == 0
    assert m2.loss is None
    assert m2.reward is None
    assert m2.epsilon is None
    assert m2.extra is None


# 2. TrainingCurveLogger init defaults
def test_logger_init_defaults():
    logger = TrainingCurveLogger()
    assert logger.run_name == 'default'
    assert logger.experiment == 'flopyarcade'
    assert logger.n_steps() == 0


# 3. record appends
def test_record_appends():
    logger = TrainingCurveLogger()
    logger.record(step=1, loss=0.9)
    logger.record(step=2, loss=0.7, reward=0.5)
    assert logger.n_steps() == 2
    steps = logger.steps()
    assert steps[0].step == 1
    assert steps[1].step == 2


# 4. record with only loss
def test_record_only_loss():
    logger = TrainingCurveLogger()
    logger.record(step=5, loss=0.33)
    m = logger.steps()[0]
    assert m.loss == 0.33
    assert m.reward is None
    assert m.epsilon is None


# 5. loss_history filters non-None
def test_loss_history_filters():
    logger = TrainingCurveLogger()
    logger.record(step=1, loss=0.9)
    logger.record(step=2)  # no loss
    logger.record(step=3, loss=0.5)
    assert logger.loss_history() == [0.9, 0.5]


# 6. best_reward correct
def test_best_reward():
    logger = TrainingCurveLogger()
    logger.record(step=1, reward=0.3)
    logger.record(step=2, reward=0.8)
    logger.record(step=3, reward=0.5)
    assert logger.best_reward() == 0.8


# 7. worst_loss correct
def test_worst_loss():
    logger = TrainingCurveLogger()
    logger.record(step=1, loss=0.2)
    logger.record(step=2, loss=0.9)
    logger.record(step=3, loss=0.4)
    assert logger.worst_loss() == 0.9


# 8. moving_average basic
def test_moving_average_basic():
    hist = [1.0, 2.0, 3.0, 4.0, 5.0]
    ma = TrainingCurveLogger.moving_average(hist, window=3)
    assert len(ma) == 5
    assert ma[0] == 1.0       # partial window [1]
    assert ma[1] == 1.5       # [1,2]
    assert ma[2] == 2.0       # [1,2,3]
    assert ma[3] == 3.0       # [2,3,4]
    assert ma[4] == 4.0       # [3,4,5]


# 9. moving_average short list
def test_moving_average_short():
    hist = [10.0, 20.0]
    ma = TrainingCurveLogger.moving_average(hist, window=5)
    assert len(ma) == 2
    assert ma[0] == 10.0
    assert ma[1] == 15.0


# 10. save_json creates valid file
def test_save_json_creates_file():
    logger = TrainingCurveLogger(run_name='test_run', experiment='test_exp')
    logger.record(step=1, loss=0.9, reward=0.5, epsilon=1.0)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        path = tf.name
    try:
        logger.save_json(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data['experiment'] == 'test_exp'
        assert data['run_name'] == 'test_run'
        assert 'timestamp' in data
    finally:
        os.unlink(path)


# 11. save_json TensorBoard structure (scalars keys present)
def test_save_json_tensorboard_structure():
    logger = TrainingCurveLogger()
    logger.record(step=1, loss=0.5, reward=0.8, epsilon=0.9, extra={'lr': 0.01})
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        path = tf.name
    try:
        logger.save_json(path)
        with open(path) as f:
            data = json.load(f)
        scalars = data['scalars']
        assert 'loss' in scalars
        assert 'reward' in scalars
        assert 'epsilon' in scalars
        assert 'extra' in scalars
        assert scalars['loss'][0]['step'] == 1
        assert scalars['loss'][0]['value'] == 0.5
        assert 'lr' in scalars['extra']
    finally:
        os.unlink(path)


# 12. load_json roundtrip
def test_load_json_roundtrip():
    logger = TrainingCurveLogger(run_name='rt', experiment='exp')
    logger.record(step=1, loss=0.9, reward=0.3, epsilon=1.0, extra={'a': 1})
    logger.record(step=2, loss=0.5, reward=0.7, epsilon=0.8)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        path = tf.name
    try:
        logger.save_json(path)
        loaded = TrainingCurveLogger.load_json(path)
        assert loaded.run_name == 'rt'
        assert loaded.experiment == 'exp'
        assert loaded.n_steps() == 2
        assert loaded.loss_history() == [0.9, 0.5]
        assert loaded.reward_history() == [0.3, 0.7]
        assert loaded.epsilon_history() == [1.0, 0.8]
    finally:
        os.unlink(path)


# 13. save_tensorboard_text format (tab-separated)
def test_save_tensorboard_text():
    logger = TrainingCurveLogger()
    logger.record(step=1, loss=0.5, reward=0.8)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
        path = tf.name
    try:
        logger.save_tensorboard_text(path)
        with open(path) as f:
            lines = f.read().strip().split('\n')
        # Should have 2 lines (loss + reward for step 1)
        assert len(lines) == 2
        parts = lines[0].split('\t')
        assert parts[0] == '1'
        assert parts[1] == 'loss'
        assert parts[2] == '0.5'
    finally:
        os.unlink(path)


# 14. summary structure
def test_summary_structure():
    logger = TrainingCurveLogger()
    for i in range(15):
        logger.record(step=i, loss=1.0 - i * 0.05, reward=i * 0.1, epsilon=1.0 - i * 0.05)
    s = logger.summary()
    assert s['n_steps'] == 15
    assert s['best_reward'] == pytest.approx(1.4)
    assert s['worst_loss'] == 1.0
    assert s['final_epsilon'] == pytest.approx(1.0 - 14 * 0.05)
    assert s['loss_trend'] is not None
    assert 'first_10_avg' in s['loss_trend']
    assert 'last_10_avg' in s['loss_trend']


# 15. compare_curves structure
def test_compare_curves():
    a = TrainingCurveLogger()
    a.record(step=1, loss=0.5, reward=0.9)
    b = TrainingCurveLogger()
    b.record(step=1, loss=0.3, reward=0.7)
    result = compare_curves(a, b)
    assert result['a_best'] == 0.9
    assert result['b_best'] == 0.7
    assert result['winner'] == 'a'
    assert result['reward_diff'] == pytest.approx(0.2)
    assert result['loss_diff'] == pytest.approx(0.2)


# 16. format_curve_report contains text
def test_format_curve_report():
    logger = TrainingCurveLogger(run_name='demo', experiment='test')
    logger.record(step=1, loss=0.9, reward=0.5, epsilon=1.0)
    logger.record(step=2, loss=0.7, reward=0.8, epsilon=0.9)
    report = format_curve_report(logger)
    assert 'Training Curve Report' in report
    assert 'test/demo' in report
    assert 'Steps recorded' in report
    assert 'Best reward' in report


# 17. empty logger summary
def test_empty_logger_summary():
    logger = TrainingCurveLogger()
    s = logger.summary()
    assert s['n_steps'] == 0
    assert s['best_reward'] is None
    assert s['worst_loss'] is None
    assert s['final_epsilon'] is None
    assert s['loss_trend'] is None


# 18. extra dict recorded
def test_extra_dict_recorded():
    logger = TrainingCurveLogger()
    logger.record(step=1, loss=0.5, extra={'lr': 0.001, 'batch': 32})
    m = logger.steps()[0]
    assert m.extra == {'lr': 0.001, 'batch': 32}
