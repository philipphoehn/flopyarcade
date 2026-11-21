"""Head/velocity field snapshots: record grid arrays every N steps → .npz (ROADMAP P5).

Provides lightweight recording of hydraulic head and velocity fields across
time steps, export to NumPy .npz archives, and analytical helpers for
temporal-change and field-comparison diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FieldSnapshot:
    """A single recorded field state."""

    step: int
    head: list[list[float]]
    velocity_x: list[list[float]] | None = None
    velocity_y: list[list[float]] | None = None


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class FieldRecorder:
    """Record head/velocity fields every *record_every* steps and persist to .npz."""

    def __init__(
        self,
        envtype: str,
        grid_shape: tuple[int, int],
        record_every: int = 1,
    ) -> None:
        self.envtype = envtype
        self.grid_shape = grid_shape
        self.record_every = record_every
        self._snapshots: list[FieldSnapshot] = []
        self._timestamp: str = datetime.now(timezone.utc).isoformat()

    # -- recording -----------------------------------------------------------

    def record(
        self,
        step: int,
        head: list[list[float]],
        vx: list[list[float]] | None = None,
        vy: list[list[float]] | None = None,
    ) -> None:
        """Record a snapshot if *step* is a multiple of *record_every*."""
        if step % self.record_every == 0:
            self._snapshots.append(
                FieldSnapshot(step=step, head=head, velocity_x=vx, velocity_y=vy)
            )

    # -- accessors -----------------------------------------------------------

    def snapshots(self) -> list[FieldSnapshot]:
        return list(self._snapshots)

    def n_snapshots(self) -> int:
        return len(self._snapshots)

    # -- persistence ---------------------------------------------------------

    def save_npz(self, path: str) -> None:
        """Write all recorded snapshots to a .npz file.

        Keys:
            heads       (N, H, W) float64
            vxs         (N, H, W) float64  (or empty (0,) if none)
            vys         (N, H, W) float64  (or empty (0,) if none)
            steps       (N,)       int64
            envtype     str
            grid_shape  list[int]
            record_every int
            timestamp   str
        """
        n = len(self._snapshots)
        h, w = self.grid_shape

        heads = np.array([s.head for s in self._snapshots], dtype=np.float64) if n else np.zeros((0, h, w), dtype=np.float64)

        has_vx = all(s.velocity_x is not None for s in self._snapshots) and n > 0
        has_vy = all(s.velocity_y is not None for s in self._snapshots) and n > 0

        vxs = np.array([s.velocity_x for s in self._snapshots], dtype=np.float64) if has_vx else np.zeros((0,), dtype=np.float64)
        vys = np.array([s.velocity_y for s in self._snapshots], dtype=np.float64) if has_vy else np.zeros((0,), dtype=np.float64)

        steps = np.array([s.step for s in self._snapshots], dtype=np.int64) if n else np.zeros((0,), dtype=np.int64)

        np.savez(
            path,
            heads=heads,
            vxs=vxs,
            vys=vys,
            steps=steps,
            envtype=np.array(self.envtype),
            grid_shape=np.array(list(self.grid_shape)),
            record_every=np.array(self.record_every, dtype=np.int64),
            timestamp=np.array(self._timestamp),
        )

    @classmethod
    def load_npz(cls, path: str) -> FieldRecorder:
        """Load a .npz file previously written by :meth:`save_npz`."""
        data = np.load(path, allow_pickle=True)

        envtype = str(data["envtype"])
        grid_shape = tuple(int(x) for x in data["grid_shape"])
        record_every = int(data["record_every"])

        recorder = cls(
            envtype=envtype,
            grid_shape=grid_shape,
            record_every=record_every,
        )
        recorder._timestamp = str(data["timestamp"])

        heads = data["heads"]
        vxs = data["vxs"]
        vys = data["vys"]
        steps = data["steps"]

        n = len(steps)
        has_vx = vxs.size > 0 and vxs.ndim == 3
        has_vy = vys.size > 0 and vys.ndim == 3

        for i in range(n):
            head = heads[i].tolist()
            vx = vxs[i].tolist() if has_vx else None
            vy = vys[i].tolist() if has_vy else None
            recorder._snapshots.append(
                FieldSnapshot(
                    step=int(steps[i]),
                    head=head,
                    velocity_x=vx,
                    velocity_y=vy,
                )
            )

        return recorder

    # -- summary -------------------------------------------------------------

    def summary(self) -> dict:
        """Return aggregate statistics over all recorded head fields."""
        n = len(self._snapshots)
        if n == 0:
            return {
                "n_snapshots": 0,
                "grid_shape": list(self.grid_shape),
                "record_every": self.record_every,
                "head_min": 0.0,
                "head_max": 0.0,
                "head_mean_overall": 0.0,
            }
        all_heads = np.array([s.head for s in self._snapshots], dtype=np.float64)
        return {
            "n_snapshots": n,
            "grid_shape": list(self.grid_shape),
            "record_every": self.record_every,
            "head_min": float(all_heads.min()),
            "head_max": float(all_heads.max()),
            "head_mean_overall": float(all_heads.mean()),
        }


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------

def compute_field_stats(snapshot: FieldSnapshot) -> dict:
    """Compute per-snapshot field statistics."""
    h = np.array(snapshot.head, dtype=np.float64)
    result: dict = {
        "head_mean": float(h.mean()),
        "head_std": float(h.std()),
        "head_min": float(h.min()),
        "head_max": float(h.max()),
    }
    if snapshot.velocity_x is not None:
        result["vx_mean"] = float(np.mean(snapshot.velocity_x))
    if snapshot.velocity_y is not None:
        result["vy_mean"] = float(np.mean(snapshot.velocity_y))
    return result


def temporal_change(recorder: FieldRecorder) -> dict:
    """Compute temporal-change statistics across consecutive snapshots."""
    snaps = recorder.snapshots()
    n = len(snaps)

    if n < 2:
        return {
            "n_transitions": 0,
            "mean_abs_change": 0.0,
            "max_abs_change": 0.0,
            "cells_most_changed": [],
        }

    changes: list[np.ndarray] = []
    for i in range(n - 1):
        diff = np.abs(
            np.array(snaps[i + 1].head, dtype=np.float64)
            - np.array(snaps[i].head, dtype=np.float64)
        )
        changes.append(diff)

    stacked = np.stack(changes, axis=0)  # (T-1, H, W)
    mean_abs_change = float(stacked.mean())
    max_abs_change = float(stacked.max())

    # cells_most_changed: top 5 by cumulative absolute change across transitions
    cumulative = stacked.sum(axis=0)  # (H, W)
    h, w = cumulative.shape
    flat = cumulative.flatten()
    k = min(5, flat.size)
    top_idx = np.argpartition(flat, -k)[-k:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    cells_most_changed = [(int(i // w), int(i % w)) for i in top_idx]

    return {
        "n_transitions": n - 1,
        "mean_abs_change": mean_abs_change,
        "max_abs_change": max_abs_change,
        "cells_most_changed": cells_most_changed,
    }


def compare_fields(snap_a: FieldSnapshot, snap_b: FieldSnapshot) -> dict:
    """Compare two snapshots field-wise."""
    a = np.array(snap_a.head, dtype=np.float64)
    b = np.array(snap_b.head, dtype=np.float64)
    abs_diff = np.abs(a - b)

    # relative diff: |a-b| / max(|a|, |b|, eps) to avoid div-by-zero
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    rel_diff = abs_diff / denom

    return {
        "abs_diff_max": float(abs_diff.max()),
        "abs_diff_mean": float(abs_diff.mean()),
        "rel_diff_mean": float(rel_diff.mean()),
    }


def format_field_report(recorder: FieldRecorder) -> str:
    """Return a human-readable summary report string."""
    s = recorder.summary()
    lines = [
        "Field Snapshot Report",
        "=====================",
        f"Environment type : {recorder.envtype}",
        f"Grid shape       : {s['grid_shape']}",
        f"Record every     : {s['record_every']} step(s)",
        f"Snapshots        : {s['n_snapshots']}",
        f"Head min         : {s['head_min']:.6f}",
        f"Head max         : {s['head_max']:.6f}",
        f"Head mean        : {s['head_mean_overall']:.6f}",
    ]

    tc = temporal_change(recorder)
    if tc["n_transitions"] > 0:
        lines.append("---")
        lines.append("Temporal change")
        lines.append(f"  Transitions      : {tc['n_transitions']}")
        lines.append(f"  Mean |Δh|       : {tc['mean_abs_change']:.6f}")
        lines.append(f"  Max  |Δh|       : {tc['max_abs_change']:.6f}")
        if tc["cells_most_changed"]:
            lines.append(f"  Top-5 cells     : {tc['cells_most_changed']}")

    return "\n".join(lines)

