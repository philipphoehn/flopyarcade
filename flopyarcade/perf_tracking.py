"""B23: Wall-clock vs sim-steps dual-axis performance tracking.

Tracks ms/step and steps/s per environment type, flags drift > threshold
(10% by default) between baseline and latest sample, and provides
comparison utilities across commits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PerfSample:
    """A single performance measurement snapshot."""

    timestamp: str
    envtype: str
    ms_per_step: float
    steps_per_sec: float
    n_episodes: int
    git_sha: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PerfSample:
        return cls(
            timestamp=d["timestamp"],
            envtype=d["envtype"],
            ms_per_step=d["ms_per_step"],
            steps_per_sec=d["steps_per_sec"],
            n_episodes=d["n_episodes"],
            git_sha=d.get("git_sha"),
        )


class PerfTracker:
    """Collects PerfSamples and detects performance drift.

    Parameters
    ----------
    drift_threshold_pct : float
        Percentage change in ms_per_step above which drift is flagged.
        Default 10.0 (i.e. >10 % slower than baseline).
    """

    def __init__(self, drift_threshold_pct: float = 10.0) -> None:
        self.drift_threshold_pct = drift_threshold_pct
        self._samples: list[PerfSample] = []

    def record(
        self,
        envtype: str,
        ms_per_step: float,
        n_episodes: int,
        git_sha: Optional[str] = None,
    ) -> PerfSample:
        """Record a new performance sample and append to the sample list."""
        steps_per_sec = 1000.0 / ms_per_step if ms_per_step > 0 else float("inf")
        sample = PerfSample(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            envtype=envtype,
            ms_per_step=ms_per_step,
            steps_per_sec=steps_per_sec,
            n_episodes=n_episodes,
            git_sha=git_sha,
        )
        self._samples.append(sample)
        return sample

    def get_baseline(self, envtype: str) -> PerfSample:
        """Return the first recorded sample for *envtype*."""
        for s in self._samples:
            if s.envtype == envtype:
                return s
        raise KeyError(f"No samples recorded for envtype '{envtype}'")

    def check_drift(self, envtype: str) -> dict:
        """Compare latest sample to baseline for *envtype*.

        Returns a dict with keys:
            baseline_ms, current_ms, drift_pct, flag
        """
        env_samples = [s for s in self._samples if s.envtype == envtype]
        if len(env_samples) < 2:
            baseline_ms = env_samples[0].ms_per_step if env_samples else 0.0
            return {
                "baseline_ms": baseline_ms,
                "current_ms": baseline_ms,
                "drift_pct": 0.0,
                "flag": False,
            }
        baseline = env_samples[0]
        current = env_samples[-1]
        if baseline.ms_per_step == 0:
            drift_pct = 0.0
        else:
            drift_pct = (current.ms_per_step - baseline.ms_per_step) / baseline.ms_per_step * 100.0
        flag = abs(drift_pct) > self.drift_threshold_pct
        return {
            "baseline_ms": baseline.ms_per_step,
            "current_ms": current.ms_per_step,
            "drift_pct": drift_pct,
            "flag": flag,
        }

    def all_drifts(self) -> dict[str, dict]:
        """Return drift info for every tracked envtype."""
        envtypes = {s.envtype for s in self._samples}
        return {et: self.check_drift(et) for et in sorted(envtypes)}

    def summary(self) -> dict:
        """High-level summary of the tracker state."""
        envtypes = {s.envtype for s in self._samples}
        drifts = self.all_drifts()
        max_drift_pct = max((abs(d["drift_pct"]) for d in drifts.values()), default=0.0)
        flagged_count = sum(1 for d in drifts.values() if d["flag"])
        return {
            "n_samples": len(self._samples),
            "envtypes_tracked": len(envtypes),
            "max_drift_pct": max_drift_pct,
            "flagged_count": flagged_count,
        }

    def save(self, path: str) -> None:
        """Serialize all samples as a JSON list to *path*."""
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in self._samples], f, indent=2)

    @classmethod
    def load(cls, path: str, drift_threshold_pct: float = 10.0) -> PerfTracker:
        """Load a PerfTracker from a JSON file saved via :meth:`save`."""
        tracker = cls(drift_threshold_pct=drift_threshold_pct)
        with open(path, "r") as f:
            data = json.load(f)
        tracker._samples = [PerfSample.from_dict(d) for d in data]
        return tracker


def format_perf_report(tracker: PerfTracker) -> str:
    """Render a human-readable text table of all drifts."""
    lines = []
    lines.append(f"{'envtype':<20} {'baseline_ms':>12} {'current_ms':>12} {'drift_pct':>10} {'flag':>6}")
    lines.append("-" * 64)
    for et, d in tracker.all_drifts().items():
        flag_str = "FLAG" if d["flag"] else "ok"
        lines.append(
            f"{et:<20} {d['baseline_ms']:>12.4f} {d['current_ms']:>12.4f} "
            f"{d['drift_pct']:>+10.2f}% {flag_str:>6}"
        )
    s = tracker.summary()
    lines.append("-" * 64)
    lines.append(
        f"Samples: {s['n_samples']}  |  Envtypes: {s['envtypes_tracked']}  |  "
        f"Max drift: {s['max_drift_pct']:.2f}%  |  Flagged: {s['flagged_count']}"
    )
    return "\n".join(lines)


def compare_perf(
    samples_a: list[PerfSample],
    samples_b: list[PerfSample],
) -> dict[str, float]:
    """Compare two sets of samples, returning pct change in ms_per_step per envtype.

    For each envtype present in *both* lists, uses the first sample from each.
    Result: ``(b.ms - a.ms) / a.ms * 100``.
    """
    result: dict[str, float] = {}
    a_by_env: dict[str, PerfSample] = {}
    for s in samples_a:
        if s.envtype not in a_by_env:
            a_by_env[s.envtype] = s
    b_by_env: dict[str, PerfSample] = {}
    for s in samples_b:
        if s.envtype not in b_by_env:
            b_by_env[s.envtype] = s
    for envtype in a_by_env:
        if envtype in b_by_env and a_by_env[envtype].ms_per_step != 0:
            a_ms = a_by_env[envtype].ms_per_step
            b_ms = b_by_env[envtype].ms_per_step
            result[envtype] = (b_ms - a_ms) / a_ms * 100.0
    return result
