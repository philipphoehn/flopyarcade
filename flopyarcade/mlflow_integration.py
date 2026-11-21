"""
MLflow / TensorBoard integration for FloPyArcade.

Logs metrics, params, artifacts, and tags.
Works without mlflow installed (JSON fallback).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# TrackingConfig
# ---------------------------------------------------------------------------

@dataclass
class TrackingConfig:
    experiment_name: str = 'flopyarcade'
    run_name: str = ''
    tracking_uri: str = 'file:./mlruns'
    tensorboard_dir: str = './tb_logs'


# ---------------------------------------------------------------------------
# MetricRecorder
# ---------------------------------------------------------------------------

class MetricRecorder:
    """In-memory recorder for params, metrics, artifacts, and tags."""

    def __init__(self, config: TrackingConfig | None = None):
        self.config = config or TrackingConfig()
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, list[tuple[int, float]]] = {}
        self._artifacts: list[str] = []
        self._tags: dict[str, str] = {}

    def log_param(self, key: str, value: Any) -> None:
        self._params[key] = value

    def log_metric(self, step: int, key: str, value: float) -> None:
        self._metrics.setdefault(key, []).append((step, value))

    def log_artifact(self, path: str) -> None:
        self._artifacts.append(path)

    def set_tag(self, key: str, value: str) -> None:
        self._tags[key] = value

    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def metrics(self) -> dict[str, list[tuple[int, float]]]:
        return {k: list(v) for k, v in self._metrics.items()}

    def artifacts(self) -> list[str]:
        return list(self._artifacts)

    def tags(self) -> dict[str, str]:
        return dict(self._tags)


# ---------------------------------------------------------------------------
# MLflowLogger
# ---------------------------------------------------------------------------

class MLflowLogger:
    """Logger that works with or without mlflow installed.

    Falls back to pure JSON output when mlflow is unavailable.
    """

    def __init__(self, recorder: MetricRecorder):
        self.recorder = recorder
        self._run_id: str | None = None
        self._status: str = 'CREATED'
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._mlflow_run = None
        self._mlflow_client = None

    def start_run(self) -> str:
        self._run_id = str(uuid.uuid4())
        self._started_at = time.time()
        self._status = 'RUNNING'

        # Try mlflow if available
        try:
            import mlflow  # noqa: F401
            mlflow.set_experiment(self.recorder.config.experiment_name)
            run_name = self.recorder.config.run_name or self._run_id
            mlflow.start_run(run_name=run_name)
            self._mlflow_run = mlflow.active_run()
            self._run_id = self._mlflow_run.info.run_id
        except ImportError:
            pass
        except Exception:
            # mlflow present but misconfigured – fall back silently
            self._mlflow_run = None

        # Log params we already have
        for k, v in self.recorder.params().items():
            self._safe_log_param(k, v)

        return self._run_id

    def end_run(self, status: str = 'FINISHED') -> None:
        self._status = status
        self._ended_at = time.time()
        self.flush()
        if self._mlflow_run is not None:
            try:
                import mlflow
                mlflow.end_run(status=status)
            except Exception:
                pass

    def flush(self) -> None:
        """Write any pending data to mlflow or no-op in fallback mode."""
        if self._mlflow_run is not None:
            try:
                import mlflow
                for k, v in self.recorder.params().items():
                    mlflow.log_param(k, v)
                for key, points in self.recorder.metrics().items():
                    for step, val in points:
                        mlflow.log_metric(key, val, step)
                for path in self.recorder.artifacts():
                    mlflow.log_artifact(path)
                for k, v in self.recorder.tags().items():
                    mlflow.set_tag(k, v)
            except Exception:
                pass

    def export_json(self, path: str) -> None:
        data = {
            'run_id': self._run_id,
            'params': self.recorder.params(),
            'metrics': self.recorder.metrics(),
            'artifacts': self.recorder.artifacts(),
            'tags': self.recorder.tags(),
            'status': self._status,
            'timestamps': {
                'started_at': self._started_at,
                'ended_at': self._ended_at,
            },
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def export_tensorboard(self, path: str) -> None:
        """Write TB-compatible tab-separated text: step\tkey\tvalue."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        lines: list[str] = []
        for key, points in sorted(self.recorder.metrics().items()):
            for step, val in points:
                lines.append(f"{step}\t{key}\t{val}")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
            if lines:
                f.write('\n')

    def summary(self) -> dict:
        return {
            'run_id': self._run_id,
            'n_params': len(self.recorder.params()),
            'n_metrics': sum(len(v) for v in self.recorder.metrics().values()),
            'n_artifacts': len(self.recorder.artifacts()),
            'status': self._status,
        }

    def _safe_log_param(self, key: str, value: Any) -> None:
        if self._mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_param(key, value)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# TensorBoardWriter
# ---------------------------------------------------------------------------

class TensorBoardWriter:
    """Writes scalar data in a simple tab-separated format per tag."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._buffer: dict[str, list[tuple[int, float]]] = {}

    def write_scalar(self, tag: str, value: float, step: int) -> None:
        self._buffer.setdefault(tag, []).append((step, value))

    def flush(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        for tag, points in self._buffer.items():
            safe_tag = tag.replace('/', '_').replace(' ', '_')
            path = os.path.join(self.log_dir, f"{safe_tag}.txt")
            with open(path, 'w') as f:
                for step, val in points:
                    f.write(f"{step}\t{tag}\t{val}\n")

    def n_entries(self) -> int:
        return sum(len(v) for v in self._buffer.values())

    def summary(self) -> dict:
        return {
            'n_tags': len(self._buffer),
            'n_entries': self.n_entries(),
            'log_dir': self.log_dir,
        }


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

def compare_runs(run_a: dict, run_b: dict) -> dict:
    """Compare two exported run JSONs.

    Returns dict with keys: param_diffs, metric_diffs, winner.
    """
    params_a = run_a.get('params', {})
    params_b = run_b.get('params', {})

    param_diffs: dict[str, dict] = {}
    all_param_keys = set(params_a) | set(params_b)
    for k in sorted(all_param_keys):
        va = params_a.get(k)
        vb = params_b.get(k)
        if va != vb:
            param_diffs[k] = {'a': va, 'b': vb}

    metrics_a = run_a.get('metrics', {})
    metrics_b = run_b.get('metrics', {})
    all_metric_keys = set(metrics_a) | set(metrics_b)

    metric_diffs: dict[str, dict] = {}
    for k in sorted(all_metric_keys):
        vals_a = [v for _, v in metrics_a.get(k, [])]
        vals_b = [v for _, v in metrics_b.get(k, [])]
        mean_a = sum(vals_a) / len(vals_a) if vals_a else 0.0
        mean_b = sum(vals_b) / len(vals_b) if vals_b else 0.0
        metric_diffs[k] = {
            'mean_a': mean_a,
            'mean_b': mean_b,
            'diff': mean_b - mean_a,
        }

    # Determine winner by final reward if present
    winner = None
    final_reward_a = _final_metric(metrics_a, 'reward')
    final_reward_b = _final_metric(metrics_b, 'reward')
    if final_reward_a is not None and final_reward_b is not None:
        winner = 'a' if final_reward_a >= final_reward_b else 'b'
    elif final_reward_a is not None:
        winner = 'a'
    elif final_reward_b is not None:
        winner = 'b'

    return {
        'param_diffs': param_diffs,
        'metric_diffs': metric_diffs,
        'winner': winner,
    }


def _final_metric(metrics: dict, key: str) -> float | None:
    points = metrics.get(key, [])
    if not points:
        return None
    return points[-1][1]


# ---------------------------------------------------------------------------
# format_run_report
# ---------------------------------------------------------------------------

def format_run_report(recorder: MetricRecorder) -> str:
    """Human-readable text summary of a recorder's contents."""
    lines: list[str] = []
    lines.append('=== FloPyArcade Run Report ===')
    lines.append(f"Experiment: {recorder.config.experiment_name}")
    lines.append(f"Run name:   {recorder.config.run_name or '(default)'}")
    lines.append('')

    params = recorder.params()
    lines.append(f'Params ({len(params)}):')
    if params:
        for k, v in sorted(params.items()):
            lines.append(f'  {k}: {v}')
    else:
        lines.append('  (none)')
    lines.append('')

    metrics = recorder.metrics()
    n_points = sum(len(v) for v in metrics.values())
    lines.append(f'Metrics ({len(metrics)} keys, {n_points} points):')
    if metrics:
        for k in sorted(metrics):
            pts = metrics[k]
            vals = [v for _, v in pts]
            lines.append(f'  {k}: steps 0-{pts[-1][0]}, mean={sum(vals)/len(vals):.4f}')
    else:
        lines.append('  (none)')
    lines.append('')

    artifacts = recorder.artifacts()
    lines.append(f'Artifacts ({len(artifacts)}):')
    if artifacts:
        for a in artifacts:
            lines.append(f'  {a}')
    else:
        lines.append('  (none)')
    lines.append('')

    tags = recorder.tags()
    lines.append(f'Tags ({len(tags)}):')
    if tags:
        for k, v in sorted(tags.items()):
            lines.append(f'  {k}: {v}')
    else:
        lines.append('  (none)')
    lines.append('')
    lines.append('=== End of Report ===')

    return '\n'.join(lines)
