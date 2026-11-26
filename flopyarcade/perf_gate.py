"""Q3: Performance regression gate – assert ms/step < 2× baseline; fail CI on >50% slowdown.

Provides dataclasses for baselines and measurements, a PerfGate class for
comparisons and threshold logic, and helper functions for CI integration,
report formatting, auto-baseline updates, and regression history.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional


# ─── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PerfBaseline:
    """A stored performance baseline for one environment type."""

    envtype: str
    baseline_ms_per_step: float
    recorded_at: str
    git_sha: Optional[str] = None


@dataclass
class PerfMeasurement:
    """A single performance measurement run."""

    envtype: str
    ms_per_step: float
    n_episodes: int
    timestamp: str


# ─── Gate logic ────────────────────────────────────────────────────────────────

class PerfGate:
    """Performance regression gate.

    Compares measurements against baselines and classifies results as
    pass / warn / fail based on ratio thresholds.

    Parameters
    ----------
    baselines : dict[str, PerfBaseline]
        Mapping of envtype → PerfBaseline.
    max_ratio : float
        Maximum allowed ratio (measured / baseline) before failure.
        Default 2.0 (i.e. 2× baseline).
    fail_threshold_pct : float
        Percentage slowdown that triggers failure. Default 50.0.
    """

    def __init__(
        self,
        baselines: dict[str, PerfBaseline],
        max_ratio: float = 2.0,
        fail_threshold_pct: float = 50.0,
    ) -> None:
        self._baselines = dict(baselines)
        self.max_ratio = max_ratio
        self.fail_threshold_pct = fail_threshold_pct

    # ── Core checks ──

    def check(self, measurement: PerfMeasurement) -> dict:
        """Compare one measurement against its baseline.

        Returns a dict with keys:
            envtype, baseline_ms, measured_ms, ratio, slowdown_pct, passed, action
        """
        baseline = self._baselines.get(measurement.envtype)
        if baseline is None:
            return {
                "envtype": measurement.envtype,
                "baseline_ms": None,
                "measured_ms": measurement.ms_per_step,
                "ratio": None,
                "slowdown_pct": None,
                "passed": True,
                "action": "pass",
                "note": "no baseline; skipping",
            }

        baseline_ms = baseline.baseline_ms_per_step
        measured_ms = measurement.ms_per_step

        if baseline_ms <= 0:
            ratio = 1.0
            slowdown_pct = 0.0
        else:
            ratio = measured_ms / baseline_ms
            slowdown_pct = (measured_ms - baseline_ms) / baseline_ms * 100.0

        passed = ratio <= self.max_ratio

        if ratio <= 1.5:
            action = "pass"
        elif ratio <= 2.0:
            action = "warn"
        else:
            action = "fail"

        return {
            "envtype": measurement.envtype,
            "baseline_ms": baseline_ms,
            "measured_ms": measured_ms,
            "ratio": round(ratio, 4),
            "slowdown_pct": round(slowdown_pct, 2),
            "passed": passed,
            "action": action,
        }

    def check_all(self, measurements: list[PerfMeasurement]) -> list[dict]:
        """Run check() on every measurement and return the list of result dicts."""
        return [self.check(m) for m in measurements]

    # ── Aggregations ──

    def gate_passed(self, results: list[dict]) -> bool:
        """True if no result has action == 'fail'."""
        return all(r.get("action") != "fail" for r in results)

    def worst_regression(self, results: list[dict]) -> dict:
        """Return the result dict with the highest slowdown_pct."""
        valid = [r for r in results if r.get("slowdown_pct") is not None]
        if not valid:
            return results[0] if results else {}
        return max(valid, key=lambda r: r["slowdown_pct"])

    # ── Baseline management ──

    def set_baseline(
        self,
        envtype: str,
        ms_per_step: float,
        git_sha: Optional[str] = None,
    ) -> None:
        """Create or update the baseline for *envtype*."""
        self._baselines[envtype] = PerfBaseline(
            envtype=envtype,
            baseline_ms_per_step=ms_per_step,
            recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            git_sha=git_sha,
        )

    def baselines(self) -> dict[str, PerfBaseline]:
        """Return a copy of the baselines dict."""
        return dict(self._baselines)

    # ── Persistence ──

    def save_baselines(self, path: str) -> None:
        """Serialize baselines to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "max_ratio": self.max_ratio,
            "fail_threshold_pct": self.fail_threshold_pct,
            "baselines": {
                k: asdict(v) for k, v in self._baselines.items()
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")

    @classmethod
    def load_baselines(cls, path: str) -> PerfGate:
        """Load a PerfGate from a JSON file written by save_baselines."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        baselines = {
            k: PerfBaseline(**v) for k, v in payload["baselines"].items()
        }
        return cls(
            baselines=baselines,
            max_ratio=payload.get("max_ratio", 2.0),
            fail_threshold_pct=payload.get("fail_threshold_pct", 50.0),
        )


# ─── CI helpers ────────────────────────────────────────────────────────────────

def ci_gate_check(
    gate: PerfGate,
    measurements: list[PerfMeasurement],
) -> dict:
    """Run the full CI gate pipeline.

    Returns a dict with keys: passed, results, worst, summary.
    """
    results = gate.check_all(measurements)
    passed = gate.gate_passed(results)
    worst = gate.worst_regression(results) if results else {}

    n_pass = sum(1 for r in results if r["action"] == "pass")
    n_warn = sum(1 for r in results if r["action"] == "warn")
    n_fail = sum(1 for r in results if r["action"] == "fail")
    summary = (
        f"Q3 PerfGate: {n_pass} pass, {n_warn} warn, {n_fail} fail"
        f" | gate {'PASSED' if passed else 'FAILED'}"
    )

    return {
        "passed": passed,
        "results": results,
        "worst": worst,
        "summary": summary,
    }


def format_gate_report(results: list[dict]) -> str:
    """Format a human-readable text table of gate results.

    Columns: envtype | baseline | measured | ratio | slowdown% | status
    """
    if not results:
        return "Q3 PerfGate: no measurements."

    header = f"{'envtype':<16} {'baseline':>10} {'measured':>10} {'ratio':>8} {'slowdown%':>12} {'status':>6}"
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        envtype = r["envtype"]
        baseline = r.get("baseline_ms")
        measured = r.get("measured_ms")
        ratio = r.get("ratio")
        slowdown = r.get("slowdown_pct")
        action = r["action"].upper()

        b_str = f"{baseline:.2f}ms" if baseline is not None else "N/A"
        m_str = f"{measured:.2f}ms" if measured is not None else "N/A"
        r_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        s_str = f"{slowdown:+.1f}%" if slowdown is not None else "N/A"

        lines.append(f"{envtype:<16} {b_str:>10} {m_str:>10} {r_str:>8} {s_str:>12} {action:>6}")

    return "\n".join(lines)


def auto_update_baseline(
    gate: PerfGate,
    measurements: list[PerfMeasurement],
    only_if_better: bool = True,
) -> list[str]:
    """Update baselines where the new measurement is faster (lower ms).

    Returns a list of envtypes that were updated.
    """
    updated: list[str] = []
    for m in measurements:
        existing = gate._baselines.get(m.envtype)
        if existing is None:
            gate.set_baseline(m.envtype, m.ms_per_step)
            updated.append(m.envtype)
            continue
        if only_if_better:
            if m.ms_per_step < existing.baseline_ms_per_step:
                gate.set_baseline(m.envtype, m.ms_per_step)
                updated.append(m.envtype)
        else:
            gate.set_baseline(m.envtype, m.ms_per_step)
            updated.append(m.envtype)
    return updated


def regression_history(path: str) -> list[dict]:
    """Read a JSONL file of past gate checks.

    Each line should be a JSON object with at least a 'timestamp' key.
    Returns a list of dicts (one per line). Empty list if file missing.
    """
    if not os.path.isfile(path):
        return []
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
