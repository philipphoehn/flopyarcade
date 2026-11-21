"""R10: Tests for MAML reward weight meta-learning."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from flopyarcade.maml_reward import (
    ENV_TYPES,
    MAMLRewardTrainer,
    MetaParams,
    TaskSpec,
    convergence_schedule,
    format_maml_report,
)


def test_task_spec_creation():
    """TaskSpec dataclass holds expected fields with defaults."""
    t = TaskSpec(envtype="confined", reward_weights=[0.1, 0.2, 0.3, 0.4])
    assert t.envtype == "confined"
    assert t.reward_weights == [0.1, 0.2, 0.3, 0.4]
    assert t.n_episodes == 5
    assert t.inner_lr == 0.01
    assert t.outer_lr == 0.001


def test_meta_params_creation():
    """MetaParams holds base reward weights."""
    mp = MetaParams(theta=[0.5, 0.5, 0.5, 0.5])
    assert mp.theta == [0.5, 0.5, 0.5, 0.5]
    assert len(mp.theta) == 4


def test_maml_trainer_init_defaults():
    """Trainer initializes with documented defaults."""
    trainer = MAMLRewardTrainer(seed=42)
    assert trainer.n_params == 4
    assert trainer.inner_lr == 0.01
    assert trainer.outer_lr == 0.001
    assert trainer.n_inner_steps == 5
    assert trainer.n_tasks == 10
    assert len(trainer.theta) == 4
    # theta should be in [0, 1]
    for val in trainer.theta:
        assert 0.0 <= val <= 1.0


def test_sample_task_returns_valid():
    """sample_task returns a valid TaskSpec with correct envtype."""
    trainer = MAMLRewardTrainer(seed=1)
    task = trainer.sample_task()
    assert isinstance(task, TaskSpec)
    assert task.envtype in ENV_TYPES
    assert len(task.reward_weights) == trainer.n_params
    for w in task.reward_weights:
        assert 0.0 <= w <= 1.0


def test_inner_loop_adaptation_changes_weights():
    """inner_loop moves theta toward target (adaptation happens)."""
    trainer = MAMLRewardTrainer(seed=10)
    theta = [0.1, 0.1, 0.1, 0.1]
    task = TaskSpec(envtype="confined", reward_weights=[0.9, 0.9, 0.9, 0.9], inner_lr=0.1)
    adapted = trainer.inner_loop(theta, task)
    # Weights should move toward target
    assert adapted != theta
    for i in range(4):
        assert adapted[i] > theta[i], f"weight[{i}] did not move toward target"


def test_inner_loop_stable_same_task():
    """Repeated inner_loop on same task converges (approaches target)."""
    trainer = MAMLRewardTrainer(seed=20)
    theta = [0.5, 0.5, 0.5, 0.5]
    target = [0.8, 0.8, 0.8, 0.8]
    task = TaskSpec(envtype="leaky", reward_weights=target, inner_lr=0.1)
    current = list(theta)
    prev_dist = sum((current[i] - target[i]) ** 2 for i in range(4))
    for _ in range(10):
        current = trainer.inner_loop(current, task)
        dist = sum((current[i] - target[i]) ** 2 for i in range(4))
        assert dist < prev_dist + 1e-12, "loss should decrease each step"
        prev_dist = dist
    # After 10 steps with lr=0.1, should be close to target
    assert all(abs(current[i] - target[i]) < 0.05 for i in range(4))


def test_outer_loop_updates_theta():
    """outer_loop changes theta based on task signals."""
    trainer = MAMLRewardTrainer(seed=30)
    theta = [0.5, 0.5, 0.5, 0.5]
    tasks = [
        TaskSpec(envtype="confined", reward_weights=[0.8, 0.2, 0.6, 0.4], inner_lr=0.05),
        TaskSpec(envtype="unconfined", reward_weights=[0.3, 0.7, 0.5, 0.9], inner_lr=0.05),
    ]
    new_theta = trainer.outer_loop(theta, tasks)
    assert new_theta != theta
    assert len(new_theta) == 4


def test_train_runs_and_loss_decreases():
    """Full training loop runs and overall loss decreases."""
    trainer = MAMLRewardTrainer(seed=42, n_tasks=20, inner_lr=0.1, outer_lr=0.01)
    result = trainer.train(n_outer_steps=50)
    loss_hist = result["loss_history"]
    assert len(loss_hist) == 50
    # Loss should decrease overall (first vs last quarter)
    first_quarter = sum(loss_hist[:10]) / 10
    last_quarter = sum(loss_hist[-10:]) / 10
    assert last_quarter < first_quarter, f"loss did not decrease: {first_quarter:.4f} -> {last_quarter:.4f}"


def test_train_returns_correct_keys():
    """train() returns dict with required keys."""
    trainer = MAMLRewardTrainer(seed=1)
    result = trainer.train(n_outer_steps=5)
    assert "theta_history" in result
    assert "loss_history" in result
    assert "n_steps" in result
    assert result["n_steps"] == 5
    # theta_history has n_steps+1 entries (initial + each step)
    assert len(result["theta_history"]) == 6
    assert len(result["loss_history"]) == 5


def test_adapt_single_task():
    """adapt() performs single-step inner adaptation."""
    trainer = MAMLRewardTrainer(seed=5)
    theta = [0.3, 0.3, 0.3, 0.3]
    task = TaskSpec(envtype="fault", reward_weights=[0.7, 0.7, 0.7, 0.7], inner_lr=0.1)
    adapted = trainer.adapt(theta, task)
    assert len(adapted) == 4
    assert adapted != theta
    # Should move toward target
    for i in range(4):
        assert adapted[i] > theta[i]


def test_evaluate_adaptation_structure():
    """evaluate_adaptation returns expected dict structure."""
    trainer = MAMLRewardTrainer(seed=7)
    result = trainer.evaluate_adaptation(trainer.theta, n_tasks=10)
    assert "mean_episodes_to_converge" in result
    assert "std" in result
    assert "best_episode" in result
    assert "worst_episode" in result
    if result["mean_episodes_to_converge"] >= 0:
        assert result["best_episode"] <= result["worst_episode"]
        assert result["std"] >= 0.0


def test_convergence_schedule_basic():
    """convergence_schedule returns first index below threshold."""
    losses = [0.5, 0.4, 0.2, 0.09, 0.05]
    assert convergence_schedule(losses, threshold=0.1) == 3
    assert convergence_schedule(losses, threshold=0.3) == 2
    assert convergence_schedule(losses, threshold=0.6) == 0


def test_convergence_schedule_never_reached():
    """convergence_schedule returns -1 when threshold never met."""
    losses = [0.5, 0.6, 0.7, 0.8]
    assert convergence_schedule(losses, threshold=0.1) == -1
    assert convergence_schedule([], threshold=0.1) == -1


def test_save_load_roundtrip():
    """save_state / load_state preserves trainer state."""
    trainer = MAMLRewardTrainer(seed=99, n_params=6)
    # Modify theta to a known value
    trainer.theta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    trainer.save_state(path)
    loaded = MAMLRewardTrainer.load_state(path)

    assert loaded.n_params == 6
    assert loaded.theta == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert loaded.inner_lr == trainer.inner_lr
    assert loaded.outer_lr == trainer.outer_lr
    assert loaded.seed == 99

    Path(path).unlink(missing_ok=True)


def test_format_maml_report_contains_text():
    """format_maml_report produces meaningful report text."""
    trainer = MAMLRewardTrainer(seed=123)
    result = trainer.train(n_outer_steps=10)
    report = format_maml_report(trainer, result)
    assert "MAML" in report
    assert "Training Report" in report
    assert "Parameters" in report
    assert "Initial loss" in report
    assert "Final loss" in report
    assert len(report) > 50


def test_deterministic_with_fixed_seed():
    """Two trainers with same seed produce identical results."""
    t1 = MAMLRewardTrainer(seed=42, n_tasks=5)
    t2 = MAMLRewardTrainer(seed=42, n_tasks=5)

    # Initial theta should match
    assert t1.theta == t2.theta

    # Sample tasks should match
    task1 = t1.sample_task()
    task2 = t2.sample_task()
    assert task1.envtype == task2.envtype
    assert task1.reward_weights == task2.reward_weights

    # Training should produce same loss history
    r1 = t1.train(n_outer_steps=10)
    r2 = t2.train(n_outer_steps=10)
    assert r1["loss_history"] == r2["loss_history"]
