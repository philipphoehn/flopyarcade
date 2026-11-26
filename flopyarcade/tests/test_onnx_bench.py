"""Tests for flopyarcade.onnx_bench (T4)."""

import json
import math
import os
import tempfile

import pytest

from flopyarcade.onnx_bench import (
    InferenceBenchmark,
    ModelSpec,
    WeightExport,
    export_weights_tflite,
    forward_pass_numpy,
    generate_random_weights,
    inference_speedup,
    onnx_export_stubs,
)


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------

def test_modelspec_defaults():
    spec = ModelSpec(name="test_model")
    assert spec.name == "test_model"
    assert spec.input_dim == 64
    assert spec.hidden_dims == [128, 128]
    assert spec.output_dim == 5
    assert spec.framework == "tensorflow"


# ---------------------------------------------------------------------------
# WeightExport
# ---------------------------------------------------------------------------

def test_weightexport_creation():
    we = WeightExport(
        model_name="m1",
        framework="tensorflow",
        weights=[1.0, 2.0],
        biases=[0.1, 0.2],
        layer_sizes=[64, 128, 5],
        timestamp="2025-01-01T00:00:00",
    )
    assert we.model_name == "m1"
    assert we.framework == "tensorflow"
    assert we.weights == [1.0, 2.0]
    assert we.biases == [0.1, 0.2]
    assert we.layer_sizes == [64, 128, 5]
    assert we.timestamp == "2025-01-01T00:00:00"


# ---------------------------------------------------------------------------
# export_weights_tflite
# ---------------------------------------------------------------------------

def test_export_weights_tflite_structure():
    spec = ModelSpec(name="tflite_model", input_dim=32, hidden_dims=[64], output_dim=10)
    w, b = generate_random_weights(spec)
    result = export_weights_tflite(spec, w, b)
    assert isinstance(result, WeightExport)
    assert result.model_name == "tflite_model"
    assert result.framework == "tensorflow"
    assert result.layer_sizes == [32, 64, 10]
    assert len(result.weights) == len(w)
    assert len(result.biases) == len(b)
    assert result.timestamp  # non-empty


# ---------------------------------------------------------------------------
# onnx_export_stubs
# ---------------------------------------------------------------------------

def test_onnx_export_stubs_required_keys():
    spec = ModelSpec(name="onnx_test")
    stub = onnx_export_stubs(spec)
    assert "ir_version" in stub
    assert "opset_version" in stub
    assert "graph" in stub
    assert "nodes" in stub["graph"]
    assert "inputs" in stub["graph"]
    assert "outputs" in stub["graph"]


def test_onnx_export_stubs_graph_nodes_nonempty():
    spec = ModelSpec(name="onnx_test", input_dim=16, hidden_dims=[32, 32], output_dim=4)
    stub = onnx_export_stubs(spec)
    nodes = stub["graph"]["nodes"]
    assert len(nodes) > 0
    for node in nodes:
        assert "op" in node
        assert "inputs" in node
        assert "outputs" in node


# ---------------------------------------------------------------------------
# InferenceBenchmark init
# ---------------------------------------------------------------------------

def test_inferencebenchmark_init():
    spec = ModelSpec(name="bench_model")
    bench = InferenceBenchmark(spec, n_iterations=50, warmup=5)
    assert bench.spec is spec
    assert bench.n_iterations == 50
    assert bench.warmup == 5


# ---------------------------------------------------------------------------
# benchmark_numpy
# ---------------------------------------------------------------------------

def test_benchmark_numpy_returns_correct_keys():
    spec = ModelSpec(name="bench", input_dim=16, hidden_dims=[32], output_dim=4)
    w, b = generate_random_weights(spec)
    bench = InferenceBenchmark(spec, n_iterations=20, warmup=3)
    result = bench.benchmark_numpy(w, b)
    expected_keys = {"framework", "mean_ms", "std_ms", "min_ms", "max_ms",
                     "n_iterations", "input_dim", "output_dim"}
    assert expected_keys.issubset(result.keys())


def test_benchmark_numpy_mean_ms_positive():
    spec = ModelSpec(name="bench", input_dim=16, hidden_dims=[32], output_dim=4)
    w, b = generate_random_weights(spec)
    bench = InferenceBenchmark(spec, n_iterations=20, warmup=3)
    result = bench.benchmark_numpy(w, b)
    assert result["mean_ms"] > 0
    assert result["min_ms"] > 0
    assert result["max_ms"] >= result["min_ms"]


def test_benchmark_numpy_n_iterations_respected():
    spec = ModelSpec(name="bench", input_dim=16, hidden_dims=[32], output_dim=4)
    w, b = generate_random_weights(spec)
    bench = InferenceBenchmark(spec, n_iterations=77, warmup=2)
    result = bench.benchmark_numpy(w, b)
    assert result["n_iterations"] == 77


# ---------------------------------------------------------------------------
# benchmark_comparison
# ---------------------------------------------------------------------------

def test_benchmark_comparison_structure():
    spec = ModelSpec(name="bench")
    bench = InferenceBenchmark(spec)
    fake_results = {
        "numpy": {"mean_ms": 1.5, "std_ms": 0.1},
        "pytorch": {"mean_ms": 3.0, "std_ms": 0.2},
    }
    comp = bench.benchmark_comparison(fake_results)
    assert "numpy" in comp
    assert "pytorch" in comp
    for fw in comp:
        assert "mean_ms" in comp[fw]
        assert "speedup_vs_slowest" in comp[fw]
        assert "rank" in comp[fw]


def test_benchmark_comparison_speedup_calculation():
    spec = ModelSpec(name="bench")
    bench = InferenceBenchmark(spec)
    fake_results = {
        "fast": {"mean_ms": 1.0},
        "slow": {"mean_ms": 4.0},
    }
    comp = bench.benchmark_comparison(fake_results)
    # "fast" (1.0ms) is 4x faster than slowest → speedup = 4.0
    assert comp["fast"]["speedup_vs_slowest"] == pytest.approx(4.0, rel=1e-3)
    # "slow" (4.0ms) IS the slowest → speedup = 1.0
    assert comp["slow"]["speedup_vs_slowest"] == pytest.approx(1.0, rel=1e-3)
    # ranks: 1 = fastest
    assert comp["fast"]["rank"] == 1
    assert comp["slow"]["rank"] == 2


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def test_report_contains_framework_names():
    spec = ModelSpec(name="bench")
    bench = InferenceBenchmark(spec)
    fake_results = {
        "numpy": {"mean_ms": 1.0, "std_ms": 0.1},
        "tensorflow": {"mean_ms": 2.0, "std_ms": 0.2},
    }
    report = bench.report(fake_results)
    assert "numpy" in report
    assert "tensorflow" in report
    assert "Framework" in report


# ---------------------------------------------------------------------------
# save / load results
# ---------------------------------------------------------------------------

def test_save_load_results_roundtrip():
    spec = ModelSpec(name="bench")
    bench = InferenceBenchmark(spec)
    results = {
        "numpy": {"mean_ms": 1.23, "std_ms": 0.45, "min_ms": 1.0, "max_ms": 1.5,
                  "n_iterations": 100, "input_dim": 64, "output_dim": 5},
        "pytorch": {"mean_ms": 0.98, "std_ms": 0.11, "min_ms": 0.9, "max_ms": 1.2,
                    "n_iterations": 100, "input_dim": 64, "output_dim": 5},
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        bench.save_results(results, path)
        loaded = InferenceBenchmark.load_results(path)
        assert loaded == results
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# generate_random_weights
# ---------------------------------------------------------------------------

def test_generate_random_weights_correct_length():
    spec = ModelSpec(name="w", input_dim=32, hidden_dims=[64, 64], output_dim=10)
    w, b = generate_random_weights(spec)
    layer_sizes = [32, 64, 64, 10]
    expected_w = 32*64 + 64*64 + 64*10
    expected_b = 64 + 64 + 10
    assert len(w) == expected_w
    assert len(b) == expected_b


# ---------------------------------------------------------------------------
# forward_pass_numpy
# ---------------------------------------------------------------------------

def test_forward_pass_numpy_output_length():
    spec = ModelSpec(name="fp", input_dim=16, hidden_dims=[32], output_dim=7)
    w, b = generate_random_weights(spec)
    x = [0.5] * 16
    layer_sizes = [16, 32, 7]
    out = forward_pass_numpy(x, w, b, layer_sizes)
    assert len(out) == 7


def test_forward_pass_numpy_softmax_sums_to_one():
    spec = ModelSpec(name="fp", input_dim=16, hidden_dims=[32], output_dim=7)
    w, b = generate_random_weights(spec)
    x = [0.5] * 16
    layer_sizes = [16, 32, 7]
    out = forward_pass_numpy(x, w, b, layer_sizes)
    assert sum(out) == pytest.approx(1.0, rel=1e-6)
    # all values should be in [0, 1]
    assert all(0.0 <= v <= 1.0 for v in out)


# ---------------------------------------------------------------------------
# inference_speedup
# ---------------------------------------------------------------------------

def test_inference_speedup_calculation():
    assert inference_speedup(2.0, 8.0) == pytest.approx(4.0)
    assert inference_speedup(1.0, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        inference_speedup(0.0, 5.0)


# ---------------------------------------------------------------------------
# 3-layer model (64-128-128-5)
# ---------------------------------------------------------------------------

def test_three_layer_model_works():
    spec = ModelSpec(name="3layer", input_dim=64, hidden_dims=[128, 128], output_dim=5)
    w, b = generate_random_weights(spec, seed=123)
    x = [0.1] * 64
    layer_sizes = [64, 128, 128, 5]
    out = forward_pass_numpy(x, w, b, layer_sizes)
    assert len(out) == 5
    assert sum(out) == pytest.approx(1.0, rel=1e-6)

    # also verify ONNX stub and benchmark work
    stub = onnx_export_stubs(spec)
    assert len(stub["graph"]["nodes"]) > 0

    bench = InferenceBenchmark(spec, n_iterations=10, warmup=2)
    result = bench.benchmark_numpy(w, b)
    assert result["mean_ms"] > 0
