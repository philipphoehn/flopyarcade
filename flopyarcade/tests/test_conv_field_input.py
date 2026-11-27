"""Tests for flopyarcade.conv_field_input (A3: ConvNet full-field input)."""

import numpy as np
import pytest
import importlib

from flopyarcade.conv_field_input import (
    FieldReshaper,
    build_conv_model,
    ConvFieldPipeline,
)

try:
    import tensorflow  # noqa
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


# ---------------------------------------------------------------------------
# FieldReshaper
# ---------------------------------------------------------------------------

class TestFieldReshaper:
    def test_defaults(self):
        r = FieldReshaper(nrow=4, ncol=6)
        assert r.nrow == 4
        assert r.ncol == 6
        assert r.nchannels == 1
        assert r.expected_flat == 24

    def test_spatial_shape(self):
        r = FieldReshaper(nrow=3, ncol=5, nchannels=2)
        assert r.spatial_shape == (3, 5, 2)

    def test_invalid_nrow(self):
        with pytest.raises(ValueError, match="nrow"):
            FieldReshaper(nrow=0, ncol=4)

    def test_invalid_ncol(self):
        with pytest.raises(ValueError, match="ncol"):
            FieldReshaper(nrow=4, ncol=-1)

    def test_invalid_nchannels(self):
        with pytest.raises(ValueError, match="nchannels"):
            FieldReshaper(nrow=4, ncol=4, nchannels=0)

    def test_reshape_single(self):
        r = FieldReshaper(nrow=3, ncol=4, nchannels=1)
        flat = np.arange(12, dtype=np.float64)
        out = r.reshape(flat)
        assert out.shape == (3, 4, 1)
        assert out[0, 0, 0] == 0.0
        assert out[2, 3, 0] == 11.0

    def test_reshape_multichannel(self):
        r = FieldReshaper(nrow=2, ncol=3, nchannels=3)
        flat = np.ones(18)
        out = r.reshape(flat)
        assert out.shape == (2, 3, 3)
        assert (out == 1.0).all()

    def test_reshape_wrong_size(self):
        r = FieldReshaper(nrow=3, ncol=4)
        with pytest.raises(ValueError, match="expected 12"):
            r.reshape(np.ones(10))

    def test_batch(self):
        r = FieldReshaper(nrow=2, ncol=3)
        batch = np.arange(12, dtype=np.float64).reshape(2, 6)
        out = r.batch(batch)
        assert out.shape == (2, 2, 3, 1)

    def test_batch_single_vec(self):
        r = FieldReshaper(nrow=2, ncol=3)
        flat = np.arange(6, dtype=np.float64)
        out = r.batch(flat)
        assert out.shape == (1, 2, 3, 1)

    def test_batch_wrong_shape(self):
        r = FieldReshaper(nrow=2, ncol=3)
        with pytest.raises(ValueError, match="expected"):
            r.batch(np.ones((4, 99)))

    def test_repr(self):
        r = FieldReshaper(nrow=5, ncol=7, nchannels=2)
        s = repr(r)
        assert "nrow=5" in s
        assert "ncol=7" in s
        assert "nchannels=2" in s


# ---------------------------------------------------------------------------
# build_conv_model
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TF, reason='tensorflow not installed')


class TestBuildConvModel:
    @pytest.fixture(scope="class")
    def model_discrete(self):
        return build_conv_model(nrow=8, ncol=8, nchannels=1, action_size=5,
                                action_type="discrete", seed=42)

    @pytest.fixture(scope="class")
    def model_continuous(self):
        return build_conv_model(nrow=8, ncol=8, nchannels=1, action_size=3,
                                action_type="continuous", seed=42)

    def test_input_shape(self, model_discrete):
        inp = model_discrete.input_shape
        # Keras input_shape is (None, H, W, C) for batch
        assert inp[1:] == (8, 8, 1)

    def test_output_shape_discrete(self, model_discrete):
        assert model_discrete.output_shape[-1] == 5

    def test_output_shape_continuous(self, model_continuous):
        assert model_continuous.output_shape[-1] == 3

    def test_forward_single(self, model_discrete):
        x = np.random.randn(1, 8, 8, 1).astype(np.float32)
        out = model_discrete.predict(x, verbose=0)
        assert out.shape == (1, 5)

    def test_forward_batch(self, model_discrete):
        x = np.random.randn(4, 8, 8, 1).astype(np.float32)
        out = model_discrete.predict(x, verbose=0)
        assert out.shape == (4, 5)

    def test_continuous_sigmoid_range(self, model_continuous):
        x = np.random.randn(2, 8, 8, 1).astype(np.float32)
        out = model_continuous.predict(x, verbose=0)
        # sigmoid output should be in [0, 1]
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_invalid_action_type(self):
        with pytest.raises(ValueError, match="unknown action_type"):
            build_conv_model(nrow=4, ncol=4, nchannels=1,
                             action_size=2, action_type="bogus")

    def test_reproducible_seed(self):
        m1 = build_conv_model(4, 4, 1, 3, "discrete", seed=99)
        m2 = build_conv_model(4, 4, 1, 3, "discrete", seed=99)
        w1 = m1.layers[0].get_weights()
        w2 = m2.layers[0].get_weights()
        assert np.allclose(w1[0], w2[0])


# ---------------------------------------------------------------------------
# ConvFieldPipeline
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TF, reason='tensorflow not installed')
class TestConvFieldPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        return ConvFieldPipeline(
            nrow=8, ncol=8, nchannels=1,
            action_size=5, action_type="discrete", seed=42,
        )

    def test_reshaper_attached(self, pipeline):
        assert isinstance(pipeline.reshaper, FieldReshaper)
        assert pipeline.reshaper.nrow == 8
        assert pipeline.reshaper.ncol == 8

    def test_forward(self, pipeline):
        flat = np.random.randn(64).astype(np.float64)
        out = pipeline.forward(flat)
        assert out.shape == (1, 5)

    def test_forward_batch(self, pipeline):
        flat = np.random.randn(3, 64).astype(np.float64)
        out = pipeline.forward_batch(flat)
        assert out.shape == (3, 5)

    def test_repr(self, pipeline):
        s = repr(pipeline)
        assert "ConvFieldPipeline" in s
        assert "nrow=8" in s
        assert "ncol=8" in s

    def test_custom_model(self):
        from flopyarcade.conv_field_input import build_conv_model
        m = build_conv_model(4, 4, 2, 3, "discrete", seed=7)
        p = ConvFieldPipeline(nrow=4, ncol=4, nchannels=2,
                              action_size=3, model=m)
        assert p.model is m

    def test_wrong_flat_size(self, pipeline):
        with pytest.raises(ValueError, match="expected 64"):
            pipeline.forward(np.ones(32))
