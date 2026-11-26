"""Training curve logging with TensorBoard-compatible JSON output.

Provides StepMetric, TrainingCurveLogger, compare_curves, and
format_curve_report for recording and inspecting loss / reward / epsilon
trajectories over training steps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class StepMetric:
    """A single training-step record."""
    step: int
    loss: Optional[float] = None
    reward: Optional[float] = None
    epsilon: Optional[float] = None
    extra: Optional[dict] = None


class TrainingCurveLogger:
    """Collect per-step metrics and export them in multiple formats."""

    def __init__(self, run_name: str = 'default', experiment: str = 'flopyarcade'):
        self.run_name = run_name
        self.experiment = experiment
        self._metrics: list[StepMetric] = []

    def record(self, step: int, loss=None, reward=None, epsilon=None, extra=None) -> None:
        """Append a StepMetric."""
        self._metrics.append(StepMetric(
            step=step, loss=loss, reward=reward, epsilon=epsilon, extra=extra
        ))

    def steps(self) -> list[StepMetric]:
        return list(self._metrics)

    def n_steps(self) -> int:
        return len(self._metrics)

    def loss_history(self) -> list[float]:
        return [m.loss for m in self._metrics if m.loss is not None]

    def reward_history(self) -> list[float]:
        return [m.reward for m in self._metrics if m.reward is not None]

    def epsilon_history(self) -> list[float]:
        return [m.epsilon for m in self._metrics if m.epsilon is not None]

    def best_reward(self) -> float:
        rh = self.reward_history()
        if not rh:
            raise ValueError('No reward data recorded.')
        return max(rh)

    def worst_loss(self) -> float:
        lh = self.loss_history()
        if not lh:
            raise ValueError('No loss data recorded.')
        return max(lh)

    @staticmethod
    def moving_average(history: list[float], window: int = 10) -> list[float]:
        """Trailing moving average. Returns list of same length as history."""
        if not history:
            return []
        out: list[float] = []
        for i in range(len(history)):
            start = max(0, i - window + 1)
            ws = history[start:i + 1]
            out.append(sum(ws) / len(ws))
        return out

    def save_json(self, path: str) -> None:
        """Write TensorBoard-compatible JSON."""
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        loss_entries = [{'step': m.step, 'value': m.loss} for m in self._metrics if m.loss is not None]
        reward_entries = [{'step': m.step, 'value': m.reward} for m in self._metrics if m.reward is not None]
        epsilon_entries = [{'step': m.step, 'value': m.epsilon} for m in self._metrics if m.epsilon is not None]
        extra_dict: dict[str, list] = {}
        for m in self._metrics:
            if m.extra:
                for k, v in m.extra.items():
                    extra_dict.setdefault(k, []).append({'step': m.step, 'value': v})
        payload = {
            'experiment': self.experiment,
            'run_name': self.run_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'scalars': {
                'loss': loss_entries,
                'reward': reward_entries,
                'epsilon': epsilon_entries,
                'extra': extra_dict,
            },
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> 'TrainingCurveLogger':
        """Reconstruct a TrainingCurveLogger from a saved JSON file."""
        with open(path, 'r') as f:
            payload = json.load(f)
        logger = cls(
            run_name=payload.get('run_name', 'default'),
            experiment=payload.get('experiment', 'flopyarcade'),
        )
        scalars = payload.get('scalars', {})
        loss_map = {e['step']: e['value'] for e in scalars.get('loss', [])}
        reward_map = {e['step']: e['value'] for e in scalars.get('reward', [])}
        epsilon_map = {e['step']: e['value'] for e in scalars.get('epsilon', [])}
        extra_scalars = scalars.get('extra', {})
        extra_maps: dict[int, dict] = {}
        for k, entries in extra_scalars.items():
            for e in entries:
                extra_maps.setdefault(e['step'], {})[k] = e['value']
        all_steps = sorted(set(loss_map) | set(reward_map) | set(epsilon_map) | set(extra_maps))
        for s in all_steps:
            logger.record(
                step=s,
                loss=loss_map.get(s),
                reward=reward_map.get(s),
                epsilon=epsilon_map.get(s),
                extra=extra_maps.get(s),
            )
        return logger

    def save_tensorboard_text(self, path: str) -> None:
        """Write TensorBoard text-summary format: step\tmetric\tvalue."""
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        with open(path, 'w') as f:
            for m in self._metrics:
                if m.loss is not None:
                    f.write(str(m.step) + '\t' + 'loss' + '\t' + str(m.loss) + '\n')
                if m.reward is not None:
                    f.write(str(m.step) + '\t' + 'reward' + '\t' + str(m.reward) + '\n')
                if m.epsilon is not None:
                    f.write(str(m.step) + '\t' + 'epsilon' + '\t' + str(m.epsilon) + '\n')
                if m.extra:
                    for k, v in m.extra.items():
                        f.write(str(m.step) + '\t' + k + '\t' + str(v) + '\n')

    def summary(self) -> dict:
        """Return a compact dict summary of the training curve."""
        lh = self.loss_history()
        rh = self.reward_history()
        eh = self.epsilon_history()
        result: dict = {
            'n_steps': self.n_steps(),
            'best_reward': max(rh) if rh else None,
            'worst_loss': max(lh) if lh else None,
            'final_epsilon': eh[-1] if eh else None,
            'loss_trend': None,
        }
        if len(lh) >= 10:
            first_avg = sum(lh[:10]) / 10
            last_avg = sum(lh[-10:]) / 10
            result['loss_trend'] = {'first_10_avg': first_avg, 'last_10_avg': last_avg}
        elif lh:
            avg = sum(lh) / len(lh)
            result['loss_trend'] = {'first_10_avg': avg, 'last_10_avg': avg}
        return result


def compare_curves(logger_a: TrainingCurveLogger, logger_b: TrainingCurveLogger) -> dict:
    """Compare two training curves and return winner / diffs."""
    a_best = logger_a.best_reward()
    b_best = logger_b.best_reward()
    a_loss = logger_a.worst_loss()
    b_loss = logger_b.worst_loss()
    winner = 'a' if a_best > b_best else ('b' if b_best > a_best else 'tie')
    return {
        'a_best': a_best,
        'b_best': b_best,
        'winner': winner,
        'reward_diff': a_best - b_best,
        'loss_diff': a_loss - b_loss,
    }


def format_curve_report(logger: TrainingCurveLogger) -> str:
    """Human-readable text report of a training curve."""
    s = logger.summary()
    lines = [
        f'Training Curve Report: {logger.experiment}/{logger.run_name}',
        f"  Steps recorded : {s['n_steps']}",
        f"  Best reward    : {s['best_reward']}",
        f"  Worst loss     : {s['worst_loss']}",
        f"  Final epsilon  : {s['final_epsilon']}",
    ]
    if s['loss_trend']:
        lt = s['loss_trend']
        lines.append(f"  Loss trend     : first-10 avg={lt['first_10_avg']:.6f}  last-10 avg={lt['last_10_avg']:.6f}")
    if logger.n_steps() > 0:
        lines.append('  Last 5 steps:')
        for m in logger.steps()[-5:]:
            parts = [f'    step {m.step}']
            if m.loss is not None:
                parts.append(f'loss={m.loss:.6f}')
            if m.reward is not None:
                parts.append(f'reward={m.reward:.6f}')
            if m.epsilon is not None:
                parts.append(f'eps={m.epsilon:.6f}')
            lines.append('  '.join(parts))
    return '\n'.join(lines)
