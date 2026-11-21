"""T4: ONNX export stub + inference benchmark.

Provides structural ONNX export representation, numpy forward-pass
benchmarking, and cross-framework comparison utilities.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification for a simple fully-connected model."""
    name: str
    input_dim: int = 64
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    output_dim: int = 5
    framework: str = "tensorflow"


@dataclass
class WeightExport:
    """Serialised weights for a 2-layer FC model."""
    model_name: str
    framework: str
    weights: list[float]
    biases: list[float]
    layer_sizes: list[int]
    timestamp: str


# ---------------------------------------------------------------------------
# Weight export (TFLite-style serialisation)
# ---------------------------------------------------------------------------

def export_weights_tflite(
    spec: ModelSpec,
    weights: list[float],
    biases: list[float],
) -> WeightExport:
    """Serialise a simple 2-layer FC model's weights into a WeightExport."""
    layer_sizes = [spec.input_dim, spec.hidden_dims[0], spec.output_dim]
    return WeightExport(
        model_name=spec.name,
        framework=spec.framework,
        weights=list(weights),
        biases=list(biases),
        layer_sizes=layer_sizes,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# ONNX export stub (structural, not a real .onnx file)
# ---------------------------------------------------------------------------

def onnx_export_stubs(spec: ModelSpec) -> dict:
    """Return a dict representing what an ONNX export would look like.

    This is a structural stub for pipeline testing, not a valid ONNX binary.
    """
    # Build a chain: input -> [Linear+ReLU]*n_hidden -> Linear -> Softmax
    nodes: list[dict] = []
    input_shape = ["batch", spec.input_dim]

    # Track tensor names through the graph
    current = "input"
    dims = [spec.input_dim] + spec.hidden_dims + [spec.output_dim]

    for i in range(len(dims) - 1):
        w_name = f"W_{i}"
        b_name = f"B_{i}"
        out_name = f"out_{i}"
        nodes.append({
            "op": "MatMul",
            "inputs": [current, w_name],
            "outputs": [out_name],
        })
        if i < len(dims) - 2:  # all except last are ReLU
            relu_out = f"relu_{i}"
            nodes.append({
                "op": "Add",
                "inputs": [out_name, b_name],
                "outputs": [f"bias_{i}"],
            })
            nodes.append({
                "op": "Relu",
                "inputs": [f"bias_{i}"],
                "outputs": [relu_out],
            })
            current = relu_out
        else:
            nodes.append({
                "op": "Add",
                "inputs": [out_name, b_name],
                "outputs": [f"logits"],
            })
            nodes.append({
                "op": "Softmax",
                "inputs": ["logits"],
                "outputs": ["output"],
            })
            current = "output"

    return {
        "ir_version": 8,
        "opset_version": 17,
        "graph": {
            "nodes": nodes,
            "inputs": [{"name": "input", "shape": input_shape}],
            "outputs": [{"name": "output", "shape": ["batch", spec.output_dim]}],
        },
    }


# ---------------------------------------------------------------------------
# Numpy forward pass
# ---------------------------------------------------------------------------

def forward_pass_numpy(
    x: list[float],
    weights: list[float],
    biases: list[float],
    layer_sizes: list[int],
) -> list[float]:
    """Forward pass: linear -> ReLU -> ... -> linear -> softmax.

    Parameters
    ----------
    x : input vector of length layer_sizes[0]
    weights : flattened weights; each layer i has layer_sizes[i]*layer_sizes[i+1] entries
    biases : flattened biases; each layer i has layer_sizes[i+1] entries
    layer_sizes : e.g. [64, 128, 5]

    Returns softmax output of length layer_sizes[-1].
    """
    a = np.array(x, dtype=np.float64)
    w_offset = 0
    b_offset = 0
    n_layers = len(layer_sizes) - 1

    for i in range(n_layers):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        W = np.array(weights[w_offset: w_offset + in_dim * out_dim], dtype=np.float64).reshape(in_dim, out_dim)
        b = np.array(biases[b_offset: b_offset + out_dim], dtype=np.float64)
        w_offset += in_dim * out_dim
        b_offset += out_dim

        a = a @ W + b
        if i < n_layers - 1:
            a = np.maximum(a, 0.0)  # ReLU

    # Softmax
    exp_a = np.exp(a - np.max(a))
    a = exp_a / exp_a.sum()
    return a.tolist()


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

class InferenceBenchmark:
    """Benchmark forward-pass latency and compare frameworks."""

    def __init__(self, spec: ModelSpec, n_iterations: int = 100, warmup: int = 10):
        self.spec = spec
        self.n_iterations = n_iterations
        self.warmup = warmup

    def benchmark_numpy(self, weights: list[float], biases: list[float]) -> dict:
        """Time n_iterations of numpy forward pass. Returns timing dict."""
        layer_sizes = [self.spec.input_dim] + self.spec.hidden_dims + [self.spec.output_dim]
        x = [0.0] * self.spec.input_dim

        # warmup
        for _ in range(self.warmup):
            forward_pass_numpy(x, weights, biases, layer_sizes)

        times_ms: list[float] = []
        for _ in range(self.n_iterations):
            t0 = time.perf_counter_ns()
            forward_pass_numpy(x, weights, biases, layer_sizes)
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1_000_000)  # ns -> ms

        return {
            "framework": "numpy",
            "mean_ms": statistics.mean(times_ms),
            "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "n_iterations": self.n_iterations,
            "input_dim": self.spec.input_dim,
            "output_dim": self.spec.output_dim,
        }

    def benchmark_comparison(self, results: dict[str, dict]) -> dict:
        """Compare multiple framework results.

        Returns {framework: {mean_ms, speedup_vs_slowest, rank}}
        """
        if not results:
            return {}

        mean_ms = {fw: r["mean_ms"] for fw, r in results.items()}
        fastest = min(mean_ms, key=mean_ms.get)
        slowest = max(mean_ms, key=mean_ms.get)
        slowest_ms = mean_ms[slowest]

        comparison: dict[str, dict] = {}
        for fw in mean_ms:
            comparison[fw] = {
                "mean_ms": mean_ms[fw],
                "speedup_vs_slowest": round(slowest_ms / mean_ms[fw], 4) if mean_ms[fw] > 0 else float("inf"),
                "rank": 0,  # placeholder
            }

        # Assign ranks (1 = fastest)
        sorted_fws = sorted(mean_ms, key=mean_ms.get)
        for rank, fw in enumerate(sorted_fws, start=1):
            comparison[fw]["rank"] = rank

        return comparison

    def report(self, results: dict[str, dict]) -> str:
        """Formatted comparison table."""
        comp = self.benchmark_comparison(results)
        lines = [
            f"{'Framework':<15} {'Mean (ms)':>12} {'Speedup':>10} {'Rank':>6}",
            "-" * 48,
        ]
        for fw in sorted(comp, key=lambda f: comp[f]["rank"]):
            c = comp[fw]
            lines.append(
                f"{fw:<15} {c['mean_ms']:>12.6f} {c['speedup_vs_slowest']:>10.3f}x {c['rank']:>6}"
            )
        return "\n".join(lines)

    def save_results(self, results: dict[str, dict], path: str) -> None:
        """Save benchmark results to JSON."""
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    @classmethod
    def load_results(cls, path: str) -> dict:
        """Load benchmark results from JSON."""
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_random_weights(spec: ModelSpec, seed: int = 42) -> tuple[list[float], list[float]]:
    """Generate random weights and biases matching spec dimensions."""
    rng = np.random.default_rng(seed)
    layer_sizes = [spec.input_dim] + spec.hidden_dims + [spec.output_dim]
    weights: list[float] = []
    biases: list[float] = []

    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        # Xavier init scale
        scale = math.sqrt(2.0 / in_dim)
        weights.extend(rng.normal(0, scale, size=in_dim * out_dim).tolist())
        biases.extend(np.zeros(out_dim).tolist())

    return weights, biases


def inference_speedup(target_ms: float, baseline_ms: float) -> float:
    """Return speedup ratio (baseline / target)."""
    if target_ms <= 0:
        raise ValueError("target_ms must be positive")
    return baseline_ms / target_ms
