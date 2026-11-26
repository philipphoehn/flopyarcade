"""P8: Checkpoint comparison tool - side-by-side policy table + KL divergence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PolicyWeights:
    name: str
    weights: List[float]
    biases: List[float]
    layer_sizes: List[int]
    timestamp: str = ""


def load_policy_weights(path: str) -> PolicyWeights:
    """Read a policy from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return PolicyWeights(
        name=data["name"],
        weights=data["weights"],
        biases=data["biases"],
        layer_sizes=data["layer_sizes"],
        timestamp=data.get("timestamp", ""),
    )


def save_policy_weights(policy: PolicyWeights, path: str) -> None:
    """Write a policy to a JSON file."""
    data = {
        "name": policy.name,
        "weights": policy.weights,
        "biases": policy.biases,
        "layer_sizes": policy.layer_sizes,
        "timestamp": policy.timestamp,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _forward(
    weights: np.ndarray,
    biases: np.ndarray,
    layer_sizes: List[int],
    x: np.ndarray,
) -> np.ndarray:
    """Simple feedforward: linear layers, softmax applied by caller."""
    n_layers = len(layer_sizes) - 1
    w_off = 0
    b_off = 0
    h = x
    for i in range(n_layers):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        W = weights[w_off : w_off + fan_in * fan_out].reshape(fan_in, fan_out)
        b = biases[b_off : b_off + fan_out]
        w_off += fan_in * fan_out
        b_off += fan_out
        h = h @ W + b
    return h


def _softmax(z: np.ndarray) -> np.ndarray:
    z_max = z.max(axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / e.sum(axis=-1, keepdims=True)


def kl_divergence_p(
    policy_a: PolicyWeights,
    policy_b: PolicyWeights,
    n_samples: int = 1000,
    seed: int = 42,
) -> float:
    """Approximate KL(P_a || P_b) via Monte-Carlo softmax sampling.

    Samples random normalised inputs, computes softmax outputs for each
    policy, and accumulates KL(P||Q) = sum(p * log(p/q)).
    """
    rng = np.random.default_rng(seed)
    input_dim = policy_a.layer_sizes[0]

    w_a = np.asarray(policy_a.weights, dtype=np.float64)
    b_a = np.asarray(policy_a.biases, dtype=np.float64)
    w_b = np.asarray(policy_b.weights, dtype=np.float64)
    b_b = np.asarray(policy_b.biases, dtype=np.float64)

    total_kl = 0.0
    for _ in range(n_samples):
        x = rng.standard_normal(input_dim)
        norm = np.linalg.norm(x)
        if norm > 1e-12:
            x = x / norm
        else:
            x = np.ones(input_dim) / math.sqrt(input_dim)

        out_a = _forward(w_a, b_a, policy_a.layer_sizes, x)
        out_b = _forward(w_b, b_b, policy_b.layer_sizes, x)
        p = _softmax(out_a)
        q = _softmax(out_b)

        eps = 1e-15
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        total_kl += float(np.sum(p_safe * np.log(p_safe / q_safe)))

    return total_kl / n_samples


def _js_divergence_mc(
    policy_a: PolicyWeights,
    policy_b: PolicyWeights,
    n_samples: int = 1000,
    seed: int = 42,
) -> float:
    """Approximate Jensen-Shannon divergence via Monte-Carlo."""
    rng = np.random.default_rng(seed)
    input_dim = policy_a.layer_sizes[0]

    w_a = np.asarray(policy_a.weights, dtype=np.float64)
    b_a = np.asarray(policy_a.biases, dtype=np.float64)
    w_b = np.asarray(policy_b.weights, dtype=np.float64)
    b_b = np.asarray(policy_b.biases, dtype=np.float64)

    js_total = 0.0
    for _ in range(n_samples):
        x = rng.standard_normal(input_dim)
        norm = np.linalg.norm(x)
        if norm > 1e-12:
            x = x / norm
        else:
            x = np.ones(input_dim) / math.sqrt(input_dim)

        out_a = _forward(w_a, b_a, policy_a.layer_sizes, x)
        out_b = _forward(w_b, b_b, policy_b.layer_sizes, x)
        p = _softmax(out_a)
        q = _softmax(out_b)
        m = 0.5 * (p + q)

        eps = 1e-15
        p_s = np.clip(p, eps, 1.0)
        q_s = np.clip(q, eps, 1.0)
        m_s = np.clip(m, eps, 1.0)

        kl_pm = float(np.sum(p_s * np.log(p_s / m_s)))
        kl_qm = float(np.sum(q_s * np.log(q_s / m_s)))
        js_total += 0.5 * kl_pm + 0.5 * kl_qm

    return js_total / n_samples


def policy_distance(
    policy_a: PolicyWeights,
    policy_b: PolicyWeights,
) -> Dict[str, float]:
    """Compute multiple distance metrics between two policies."""
    kl_ab = kl_divergence_p(policy_a, policy_b)
    kl_ba = kl_divergence_p(policy_b, policy_a)
    js_divergence = _js_divergence_mc(policy_a, policy_b)

    w_a = np.asarray(policy_a.weights, dtype=np.float64)
    w_b = np.asarray(policy_b.weights, dtype=np.float64)
    n = min(len(w_a), len(w_b))

    if n > 0:
        diff = w_b[:n] - w_a[:n]
        max_weight_diff = float(np.max(np.abs(diff)))
        mean_weight_diff = float(np.mean(np.abs(diff)))
        relative_weight_diff = mean_weight_diff / (float(np.mean(np.abs(w_a[:n]))) + 1e-8)
    else:
        max_weight_diff = 0.0
        mean_weight_diff = 0.0
        relative_weight_diff = 0.0

    return {
        "kl_ab": kl_ab,
        "kl_ba": kl_ba,
        "js_divergence": js_divergence,
        "max_weight_diff": max_weight_diff,
        "mean_weight_diff": mean_weight_diff,
        "relative_weight_diff": relative_weight_diff,
    }


def _build_weight_layer_map(layer_sizes: List[int]) -> List[int]:
    """Map each flat weight index to its layer number."""
    mapping: List[int] = []
    for i in range(len(layer_sizes) - 1):
        count = layer_sizes[i] * layer_sizes[i + 1]
        mapping.extend([i] * count)
    return mapping


def side_by_side_table(
    policy_a: PolicyWeights,
    policy_b: PolicyWeights,
    top_n: int = 10,
) -> List[Dict]:
    """Return top-N most-changed weights sorted by |diff| descending."""
    w_a = np.asarray(policy_a.weights, dtype=np.float64)
    w_b = np.asarray(policy_b.weights, dtype=np.float64)
    n = min(len(w_a), len(w_b))
    if n == 0:
        return []

    diff = w_b[:n] - w_a[:n]
    abs_diff = np.abs(diff)
    layer_map = _build_weight_layer_map(policy_a.layer_sizes)

    top_indices = np.argsort(abs_diff)[::-1][:top_n]

    result: List[Dict] = []
    for idx in top_indices:
        idx = int(idx)
        val_a = float(w_a[idx])
        val_b = float(w_b[idx])
        d = float(diff[idx])
        pct = d / (abs(val_a) + 1e-8) * 100.0
        layer = layer_map[idx] if idx < len(layer_map) else -1
        result.append(
            {
                "index": idx,
                "layer": layer,
                "value_a": val_a,
                "value_b": val_b,
                "diff": d,
                "pct_change": pct,
            }
        )
    return result


def layer_wise_diff(
    policy_a: PolicyWeights,
    policy_b: PolicyWeights,
) -> List[Dict]:
    """Per-layer statistics of weight differences."""
    w_a = np.asarray(policy_a.weights, dtype=np.float64)
    w_b = np.asarray(policy_b.weights, dtype=np.float64)

    layer_sizes = policy_a.layer_sizes
    n_layers = len(layer_sizes) - 1

    result: List[Dict] = []
    w_idx = 0
    for i in range(n_layers):
        count = layer_sizes[i] * layer_sizes[i + 1]
        a_slice = w_a[w_idx : w_idx + count]
        b_slice = w_b[w_idx : w_idx + count]
        diff = b_slice - a_slice

        result.append(
            {
                "layer": i,
                "n_params": count,
                "mean_diff": float(np.mean(np.abs(diff))),
                "max_diff": float(np.max(np.abs(diff))),
                "rms": float(np.sqrt(np.mean(diff**2))),
            }
        )
        w_idx += count
    return result


def format_comparison_report(
    comp: Dict[str, float],
    table: List[Dict],
    layers: List[Dict],
) -> str:
    """Format a human-readable comparison report."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("CHECKPOINT COMPARISON REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Divergence Metrics:")
    lines.append(f"  KL(A||B):            {comp['kl_ab']:.6f}")
    lines.append(f"  KL(B||A):            {comp['kl_ba']:.6f}")
    lines.append(f"  JS Divergence:       {comp['js_divergence']:.6f}")
    lines.append("")
    lines.append("Weight Differences:")
    lines.append(f"  Max |diff|:          {comp['max_weight_diff']:.6f}")
    lines.append(f"  Mean |diff|:         {comp['mean_weight_diff']:.6f}")
    lines.append(f"  Relative diff:       {comp['relative_weight_diff']:.6f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("Layer-wise Statistics:")
    lines.append("-" * 60)
    for l in layers:
        lines.append(
            f"  Layer {l['layer']}: {l['n_params']} params | "
            f"mean_diff={l['mean_diff']:.6f} max_diff={l['max_diff']:.6f} rms={l['rms']:.6f}"
        )
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"Top {len(table)} Most-Changed Weights:")
    lines.append("-" * 60)
    lines.append(
        f"  {'Idx':>5} {'Layer':>5} {'Val_A':>12} {'Val_B':>12} {'Diff':>12} {'Pct%':>10}"
    )
    for t in table:
        lines.append(
            f"  {t['index']:>5} {t['layer']:>5} {t['value_a']:>12.6f} "
            f"{t['value_b']:>12.6f} {t['diff']:>12.6f} {t['pct_change']:>10.2f}"
        )
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
