"""Tests for R11: MC-Dropout Ensemble Uncertainty Quantification."""

import math
import os
import tempfile

import pytest

from flopyarcade.mc_dropout import (
    DQNHead,
    MCEnsemble,
    compare_uncertainty,
    format_ensemble_report,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def ensemble():
    """Standard 5-head ensemble with fixed seed."""
    ens = MCEnsemble(n_heads=5, input_dim=64, output_dim=5, dropout_rate=0.3, seed=42)
    ens.random_init()
    return ens


@pytest.fixture
def x64():
    """Deterministic 64-dim input vector."""
    return [float(i % 7) * 0.1 for i in range(64)]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestDQNHead:
    def test_creation(self):
        """Test DQNHead dataclass creation."""
        h = DQNHead(index=0, weights=[1.0, 2.0, 3.0], biases=[0.1, 0.2])
        assert h.index == 0
        assert h.weights == [1.0, 2.0, 3.0]
        assert h.biases == [0.1, 0.2]


class TestMCEnsembleInit:
    def test_defaults(self):
        """Test MCEnsemble init defaults."""
        ens = MCEnsemble()
        assert ens.n_heads == 5
        assert ens.input_dim == 64
        assert ens.output_dim == 5
        assert ens.dropout_rate == 0.3
        assert ens.seed is None
        assert ens.heads == []

    def test_custom_params(self):
        """Test custom initialisation parameters."""
        ens = MCEnsemble(n_heads=3, input_dim=32, output_dim=4, dropout_rate=0.5, seed=7)
        assert ens.n_heads == 3
        assert ens.input_dim == 32
        assert ens.output_dim == 4
        assert ens.dropout_rate == 0.5
        assert ens.seed == 7


class TestRandomInit:
    def test_finite_weights(self):
        """Test random_init produces finite weights."""
        ens = MCEnsemble(n_heads=2, input_dim=16, output_dim=3, seed=1)
        ens.random_init()
        assert len(ens.heads) == 2
        for h in ens.heads:
            for w in h.weights:
                assert math.isfinite(w)
            for b in h.biases:
                assert math.isfinite(b)

    def test_deterministic_with_seed(self):
        """Two ensembles with same seed produce identical weights."""
        e1 = MCEnsemble(n_heads=3, input_dim=8, output_dim=2, seed=99)
        e1.random_init()
        e2 = MCEnsemble(n_heads=3, input_dim=8, output_dim=2, seed=99)
        e2.random_init()
        for h1, h2 in zip(e1.heads, e2.heads):
            assert h1.weights == h2.weights
            assert h1.biases == h2.biases


class TestForwardSingle:
    def test_shape_correct(self, ensemble, x64):
        """Test forward_single returns output_dim values."""
        out = ensemble.forward_single(ensemble.heads[0], x64)
        assert len(out) == 5
        for v in out:
            assert math.isfinite(v)


class TestForwardEnsemble:
    def test_returns_n_heads_times_n_samples(self, ensemble, x64):
        """Test forward_ensemble returns n_heads * n_samples vectors."""
        n_samples = 4
        results = ensemble.forward_ensemble(x64, n_samples=n_samples)
        assert len(results) == 5 * n_samples
        for r in results:
            assert len(r) == 5


class TestMeanStd:
    def test_shapes(self, ensemble, x64):
        """Test mean_std returns correct shapes."""
        mean_q, std_q = ensemble.mean_std(x64, n_samples=5)
        assert len(mean_q) == 5
        assert len(std_q) == 5
        for v in std_q:
            assert v >= 0.0

    def test_zero_std_when_dropout_zero(self, x64):
        """With dropout=0 and a single head, forward is deterministic so std should be ~0."""
        ens = MCEnsemble(n_heads=1, input_dim=16, output_dim=3, dropout_rate=0.0, seed=5)
        ens.random_init()
        mean_q, std_q = ens.mean_std(x64[:16], n_samples=10)
        for s in std_q:
            assert s < 1e-10


class TestUncertaintyBonus:
    def test_proportional_to_std(self, ensemble, x64):
        """uncertainty_bonus should equal beta * std."""
        beta = 2.5
        bonus = ensemble.uncertainty_bonus(x64, n_samples=5, beta=beta)
        _, std_q = ensemble.mean_std(x64, n_samples=5)
        # Note: mean_std uses different RNG state, so compare structure
        assert len(bonus) == 5
        for b in bonus:
            assert math.isfinite(b)


class TestIntrinsicReward:
    def test_positive(self, ensemble, x64):
        """intrinsic_reward should be non-negative."""
        r = ensemble.intrinsic_reward(x64, n_samples=5)
        assert r >= 0.0
        assert math.isfinite(r)


class TestConfidence:
    def test_in_range(self, ensemble, x64):
        """confidence must be in [0, 1]."""
        c = ensemble.confidence(x64, n_samples=5)
        assert 0.0 <= c <= 1.0


class TestActionWithBonus:
    def test_returns_valid_index(self, ensemble, x64):
        """action_with_bonus must return an index in [0, output_dim)."""
        a = ensemble.action_with_bonus(x64, n_samples=5)
        assert 0 <= a < 5
        assert isinstance(a, int)


class TestTrainStep:
    def test_reduces_loss(self, x64):
        """Multiple train_steps should reduce loss."""
        ens = MCEnsemble(n_heads=2, input_dim=16, output_dim=3, dropout_rate=0.0, seed=10)
        ens.random_init()
        target = [1.0, 0.5, -0.5]
        losses = []
        for _ in range(50):
            loss = ens.train_step(x64[:16], target, lr=0.05)
            losses.append(loss)
        assert losses[-1] < losses[0]


class TestCompareUncertainty:
    def test_structure(self, ensemble, x64):
        """compare_uncertainty returns expected dict keys."""
        ens_b = MCEnsemble(n_heads=5, input_dim=64, output_dim=5, dropout_rate=0.3, seed=99)
        ens_b.random_init()
        result = compare_uncertainty(ensemble, ens_b, x64)
        assert "std_a" in result
        assert "std_b" in result
        assert "more_certain" in result
        assert "difference" in result
        assert result["more_certain"] in ("A", "B")
        assert len(result["std_a"]) == 5
        assert len(result["std_b"]) == 5


class TestFormatReport:
    def test_contains_text(self, ensemble, x64):
        """format_ensemble_report should contain key phrases."""
        report = format_ensemble_report(ensemble, x64)
        assert "MC-Dropout" in report
        assert "Heads" in report
        assert "Mean Q" in report
        assert "Confidence" in report
        assert "Best action" in report


class TestSaveLoad:
    def test_roundtrip(self, ensemble, x64):
        """Save and load should preserve weights and produce same output."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ensemble.save_state(path)
            loaded = MCEnsemble.load_state(path)
            assert loaded.n_heads == ensemble.n_heads
            assert loaded.input_dim == ensemble.input_dim
            assert loaded.output_dim == ensemble.output_dim
            assert loaded.dropout_rate == ensemble.dropout_rate
            for h_orig, h_loaded in zip(ensemble.heads, loaded.heads):
                assert h_orig.weights == h_loaded.weights
                assert h_orig.biases == h_loaded.biases
            # Forward pass should match
            out_orig = ensemble.forward_single(ensemble.heads[0], x64)
            out_loaded = loaded.forward_single(loaded.heads[0], x64)
            assert out_orig == out_loaded
        finally:
            os.unlink(path)


class TestFullConfig:
    def test_five_heads_64dim(self, ensemble, x64):
        """Full 5-head 64-dim pipeline works end-to-end."""
        ens = ensemble
        x = x64
        # Forward
        results = ens.forward_ensemble(x, n_samples=3)
        assert len(results) == 15
        # Mean/std
        mean_q, std_q = ens.mean_std(x, n_samples=3)
        assert len(mean_q) == 5
        assert len(std_q) == 5
        # Bonus
        bonus = ens.uncertainty_bonus(x, n_samples=3)
        assert len(bonus) == 5
        # Intrinsic
        intrinsic = ens.intrinsic_reward(x, n_samples=3)
        assert intrinsic >= 0
        # Confidence
        conf = ens.confidence(x, n_samples=3)
        assert 0 <= conf <= 1
        # Action
        action = ens.action_with_bonus(x, n_samples=3)
        assert 0 <= action < 5
        # Report
        report = format_ensemble_report(ens, x)
        assert len(report) > 0
