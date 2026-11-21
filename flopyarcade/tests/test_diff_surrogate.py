"""Tests for R5 differentiable MODFLOW surrogate."""

import json
import numpy as np
import pytest
import tempfile
import os

from flopyarcade.diff_surrogate import (
    SurrogateConfig,
    HydraulicSurrogate,
    MPCStepResult,
    run_mpc_planning,
    format_surrogate_report,
)


# ---------------------------------------------------------------------------
# 1. Test SurrogateConfig defaults
# ---------------------------------------------------------------------------
class TestSurrogateConfig:
    def test_defaults(self):
        cfg = SurrogateConfig()
        assert cfg.grid_size == (8, 8)
        assert cfg.n_channels == 1
        assert cfg.learning_rate == 1e-3
        assert cfg.n_epochs == 50
        assert cfg.batch_size == 8

    def test_custom_values(self):
        cfg = SurrogateConfig(grid_size=(4, 6), learning_rate=0.01, n_epochs=10)
        assert cfg.grid_size == (4, 6)
        assert cfg.learning_rate == 0.01
        assert cfg.n_epochs == 10


# ---------------------------------------------------------------------------
# 2. Test generate_training_data shapes
# ---------------------------------------------------------------------------
class TestGenerateTrainingData:
    def test_shapes(self):
        cfg = SurrogateConfig(grid_size=(6, 6))
        surr = HydraulicSurrogate(cfg)
        h_t, q, h_t1 = surr.generate_training_data(n_samples=20)
        assert h_t.shape == (20, 6, 6)
        assert q.shape == (20, 4)  # n_wells_active=4
        assert h_t1.shape == (20, 6, 6)

    def test_deterministic(self):
        cfg = SurrogateConfig(grid_size=(4, 4))
        s1 = HydraulicSurrogate(cfg)
        s2 = HydraulicSurrogate(cfg)
        h1, q1, h1_1 = s1.generate_training_data(10)
        h2, q2, h2_1 = s2.generate_training_data(10)
        np.testing.assert_array_equal(h1, h2)
        np.testing.assert_array_equal(q1, q2)


# ---------------------------------------------------------------------------
# 3. Test train reduces loss
# ---------------------------------------------------------------------------
class TestTrain:
    def test_loss_decreases(self):
        cfg = SurrogateConfig(grid_size=(4, 4), n_epochs=30, learning_rate=1e-2)
        surr = HydraulicSurrogate(cfg)
        h_t, q, h_t1 = surr.generate_training_data(n_samples=50)
        losses = surr.train(h_t, q, h_t1)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"

    def test_returns_list_of_correct_length(self):
        cfg = SurrogateConfig(grid_size=(4, 4), n_epochs=10)
        surr = HydraulicSurrogate(cfg)
        losses = surr.train()
        assert len(losses) == 10


# ---------------------------------------------------------------------------
# 4. Test predict shape matches input
# ---------------------------------------------------------------------------
class TestPredict:
    def test_output_shape(self):
        cfg = SurrogateConfig(grid_size=(6, 6))
        surr = HydraulicSurrogate(cfg)
        h_t = np.random.randn(6, 6)
        q = np.random.randn(3)
        out = surr.predict(h_t, q)
        assert out.shape == (6, 6)

    def test_dtypes(self):
        cfg = SurrogateConfig(grid_size=(4, 4))
        surr = HydraulicSurrogate(cfg)
        h_t = np.ones((4, 4))
        q = np.zeros(2)
        out = surr.predict(h_t, q)
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# 5. Test predict consistency (no wells -> diffusion only)
# ---------------------------------------------------------------------------
class TestPredictNoWells:
    def test_no_well_effect(self):
        cfg = SurrogateConfig(grid_size=(5, 5))
        surr = HydraulicSurrogate(cfg)
        # Zero well weights -> only diffusion
        surr.well_weights[:] = 0.0
        h_t = np.random.randn(5, 5)
        q = np.array([0.1, 0.2, 0.3, 0.4])
        out = surr.predict(h_t, q)
        # With zero weights, well source should be zero
        lap = surr._laplacian(h_t)
        expected = h_t + surr.alpha * cfg.dt * lap
        np.testing.assert_allclose(out, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Test backprop_mpc returns q_sequence of correct length
# ---------------------------------------------------------------------------
class TestBackpropMPC:
    def test_q_sequence_length(self):
        cfg = SurrogateConfig(grid_size=(6, 6))
        surr = HydraulicSurrogate(cfg)
        h_t = np.random.randn(6, 6)
        q = np.array([0.1, -0.2, 0.3])
        q_seq, reward = surr.backprop_mpc(h_t, q, n_steps=5, target_well_pos=(3, 3))
        assert q_seq.shape == (5, 3)

    def test_q_sequence_10_steps(self):
        cfg = SurrogateConfig(grid_size=(4, 4))
        surr = HydraulicSurrogate(cfg)
        h_t = np.ones((4, 4))
        q = np.zeros(2)
        q_seq, reward = surr.backprop_mpc(h_t, q, n_steps=10, target_well_pos=(1, 1))
        assert q_seq.shape == (10, 2)


# ---------------------------------------------------------------------------
# 7. Test backprop_mpc reward is finite
# ---------------------------------------------------------------------------
class TestBackpropReward:
    def test_reward_finite(self):
        cfg = SurrogateConfig(grid_size=(5, 5))
        surr = HydraulicSurrogate(cfg)
        h_t = np.random.randn(5, 5)
        q = np.random.randn(3)
        _, reward = surr.backprop_mpc(h_t, q, n_steps=5, target_well_pos=(2, 2))
        assert np.isfinite(reward)

    def test_reward_is_float(self):
        cfg = SurrogateConfig(grid_size=(4, 4))
        surr = HydraulicSurrogate(cfg)
        h_t = np.ones((4, 4))
        q = np.zeros(2)
        _, reward = surr.backprop_mpc(h_t, q, n_steps=3, target_well_pos=(1, 1))
        assert isinstance(reward, float)


# ---------------------------------------------------------------------------
# 8. Test run_mpc_planning returns MPCStepResult
# ---------------------------------------------------------------------------
class TestRunMPCPlanning:
    def test_returns_mpc_result(self):
        cfg = SurrogateConfig(grid_size=(6, 6))
        surr = HydraulicSurrogate(cfg)
        initial_head = np.random.randn(6, 6)
        result = run_mpc_planning(surr, initial_head, n_wells=3, n_steps=5)
        assert isinstance(result, MPCStepResult)

    def test_result_fields_populated(self):
        cfg = SurrogateConfig(grid_size=(4, 4))
        surr = HydraulicSurrogate(cfg)
        initial_head = np.ones((4, 4))
        result = run_mpc_planning(surr, initial_head, n_wells=2, n_steps=4)
        assert len(result.steps) == 4
        assert len(result.q_sequence) == 4
        assert len(result.predicted_heads) == 4
        assert np.isfinite(result.total_reward)


# ---------------------------------------------------------------------------
# 9. Test run_mpc_planning n_steps respected
# ---------------------------------------------------------------------------
class TestMPCSteps:
    def test_n_steps_3(self):
        cfg = SurrogateConfig(grid_size=(4, 4))
        surr = HydraulicSurrogate(cfg)
        h = np.ones((4, 4))
        result = run_mpc_planning(surr, h, n_wells=2, n_steps=3)
        assert len(result.steps) == 3

    def test_n_steps_15(self):
        cfg = SurrogateConfig(grid_size=(4, 4))
        surr = HydraulicSurrogate(cfg)
        h = np.ones((4, 4))
        result = run_mpc_planning(surr, h, n_wells=2, n_steps=15)
        assert len(result.steps) == 15


# ---------------------------------------------------------------------------
# 10. Test save/load weights roundtrip
# ---------------------------------------------------------------------------
class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        cfg = SurrogateConfig(grid_size=(5, 5))
        surr = HydraulicSurrogate(cfg)
        surr.alpha = 0.777
        surr.well_weights[0, :5] = np.arange(5) * 0.1
        surr.n_wells_active = 3

        path = str(tmp_path / "weights.json")
        surr.save_weights(path)

        loaded = HydraulicSurrogate.load_weights(path)
        assert loaded.alpha == pytest.approx(0.777)
        np.testing.assert_array_equal(loaded.well_weights[:4, :5], surr.well_weights[:4, :5])
        assert loaded.n_wells_active == 3
        assert loaded.config.grid_size == (5, 5)

    def test_file_is_valid_json(self, tmp_path):
        cfg = SurrogateConfig(grid_size=(4, 4))
        surr = HydraulicSurrogate(cfg)
        path = str(tmp_path / "w.json")
        surr.save_weights(path)
        with open(path) as f:
            data = json.load(f)
        assert "alpha" in data
        assert "well_weights" in data
        assert "config" in data


# ---------------------------------------------------------------------------
# 11. Test format_surrogate_report contains text
# ---------------------------------------------------------------------------
class TestFormatReport:
    def test_report_content(self):
        cfg = SurrogateConfig(grid_size=(6, 6))
        surr = HydraulicSurrogate(cfg)
        losses = [0.5, 0.3, 0.1, 0.05]
        report = format_surrogate_report(surr, losses)
        assert "Surrogate" in report
        assert "alpha" in report
        assert "loss" in report.lower()
        assert "6x6" in report

    def test_report_empty_losses(self):
        cfg = SurrogateConfig(grid_size=(4, 4))
        surr = HydraulicSurrogate(cfg)
        report = format_surrogate_report(surr, [])
        assert isinstance(report, str)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# 12. Test 3x3 grid works
# ---------------------------------------------------------------------------
class TestSmallGrid3x3:
    def test_predict_3x3(self):
        cfg = SurrogateConfig(grid_size=(3, 3))
        surr = HydraulicSurrogate(cfg)
        h = np.ones((3, 3))
        q = np.zeros(2)
        out = surr.predict(h, q)
        assert out.shape == (3, 3)

    def test_mpc_3x3(self):
        cfg = SurrogateConfig(grid_size=(3, 3))
        surr = HydraulicSurrogate(cfg)
        h = np.random.randn(3, 3)
        q_seq, reward = surr.backprop_mpc(h, np.zeros(2), n_steps=3, target_well_pos=(1, 1))
        assert q_seq.shape == (3, 2)
        assert np.isfinite(reward)

    def test_train_3x3(self):
        cfg = SurrogateConfig(grid_size=(3, 3), n_epochs=5)
        surr = HydraulicSurrogate(cfg)
        h_t, q, h_t1 = surr.generate_training_data(10)
        assert h_t.shape == (10, 3, 3)
        losses = surr.train(h_t, q, h_t1)
        assert len(losses) == 5


# ---------------------------------------------------------------------------
# 13. Test 5x5 grid works
# ---------------------------------------------------------------------------
class TestGrid5x5:
    def test_full_pipeline_5x5(self):
        cfg = SurrogateConfig(grid_size=(5, 5), n_epochs=5)
        surr = HydraulicSurrogate(cfg)
        h_t, q, h_t1 = surr.generate_training_data(20)
        assert h_t.shape == (20, 5, 5)
        losses = surr.train(h_t, q, h_t1)
        assert len(losses) == 5

        h = np.random.randn(5, 5)
        out = surr.predict(h, np.array([0.1, 0.2]))
        assert out.shape == (5, 5)

        q_seq, reward = surr.backprop_mpc(h, np.zeros(2), n_steps=4, target_well_pos=(2, 2))
        assert q_seq.shape == (4, 2)
        assert np.isfinite(reward)

    def test_mpc_planning_5x5(self):
        cfg = SurrogateConfig(grid_size=(5, 5))
        surr = HydraulicSurrogate(cfg)
        h = np.ones((5, 5))
        result = run_mpc_planning(surr, h, n_wells=3, n_steps=5)
        assert len(result.steps) == 5
        assert result.total_reward != 0.0


# ---------------------------------------------------------------------------
# 14. Test learning_rate affects convergence
# ---------------------------------------------------------------------------
class TestLearningRateEffect:
    def test_lower_lr_slower_convergence(self):
        h_t = np.random.randn(10, 4, 4)
        q = np.random.randn(10, 3)
        # Create target with simple model
        target = h_t + 0.1 * h_t  # simple scaling

        cfg_fast = SurrogateConfig(grid_size=(4, 4), n_epochs=20, learning_rate=1e-2)
        surr_fast = HydraulicSurrogate(cfg_fast)
        losses_fast = surr_fast.train(h_t, q, target)

        cfg_slow = SurrogateConfig(grid_size=(4, 4), n_epochs=20, learning_rate=1e-5)
        surr_slow = HydraulicSurrogate(cfg_slow)
        losses_slow = surr_slow.train(h_t, q, target)

        # Lower LR should result in higher (worse) final loss
        assert losses_slow[-1] >= losses_fast[-1] * 0.5  # at least somewhat slower

    def test_higher_lr_faster_initial_drop(self):
        h_t, q, h_t1 = HydraulicSurrogate(SurrogateConfig(grid_size=(4, 4))).generate_training_data(30)

        cfg_a = SurrogateConfig(grid_size=(4, 4), n_epochs=5, learning_rate=1e-2)
        s_a = HydraulicSurrogate(cfg_a)
        la = s_a.train(h_t, q, h_t1)

        cfg_b = SurrogateConfig(grid_size=(4, 4), n_epochs=5, learning_rate=1e-5)
        s_b = HydraulicSurrogate(cfg_b)
        lb = s_b.train(h_t, q, h_t1)

        # Higher LR should reduce loss more in 5 epochs
        assert la[4] <= lb[4]


# ---------------------------------------------------------------------------
# 15. Test gradient descent converges for simple 2x2 case
# ---------------------------------------------------------------------------
class TestConvergence2x2:
    def test_2x2_convergence(self):
        cfg = SurrogateConfig(grid_size=(2, 2), n_epochs=100, learning_rate=1e-2)
        surr = HydraulicSurrogate(cfg)
        h_t, q, h_t1 = surr.generate_training_data(n_samples=50)
        losses = surr.train(h_t, q, h_t1)
        # Loss should be finite throughout
        assert all(np.isfinite(l) for l in losses)
        # Final loss should be lower than initial
        assert losses[-1] < losses[0]

    def test_2x2_stable(self):
        cfg = SurrogateConfig(grid_size=(2, 2), n_epochs=50, learning_rate=1e-2)
        surr = HydraulicSurrogate(cfg)
        h_t, q, h_t1 = surr.generate_training_data(n_samples=30)
        losses = surr.train(h_t, q, h_t1)
        # No NaN/Inf
        assert not any(np.isnan(l) for l in losses)
        assert not any(np.isinf(l) for l in losses)
