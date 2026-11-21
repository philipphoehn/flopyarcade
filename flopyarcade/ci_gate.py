"""CI benchmark gate – detect score regressions (ROADMAP B24).

Fails a CI run if any environment score drops more than a configurable
percentage (default 5 %) relative to the stored baseline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ─── Data model ────────────────────────────────────────────────────────────────

@dataclass
class BaselineEntry:
    """A single baseline record for one environment type."""
    envtype: str
    score: float
    timestamp: str
    source: str = "ci"


# ─── Baseline JSON (simple {envtype: score}) ─────────────────────────────────

def load_baseline(path: str) -> dict[str, float]:
    """Read a baseline JSON file of the form ``{envtype: score}``.

    Returns an empty dict if the file does not exist.
    """
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): float(v) for k, v in data.items()}


def save_baseline(scores: dict[str, float], path: str) -> None:
    """Write ``scores`` as a JSON object ``{envtype: score}`` to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scores, fh, indent=2, sort_keys=True)
        fh.write("\n")


# ─── Gate logic ────────────────────────────────────────────────────────────────

def check_gate(
    current_scores: dict[str, float],
    baseline_scores: dict[str, float],
    threshold_pct: float = 5.0,
) -> dict[str, Any]:
    """Compare *current_scores* against *baseline_scores*.

    Parameters
    ----------
    current_scores:
        Scores produced by the current CI run.
    baseline_scores:
        Previously recorded baseline scores.
    threshold_pct:
        Maximum allowed percentage drop before a violation is reported.

    Returns
    -------
    dict with keys:
        ``pass`` – True when no violations exist.
        ``violations`` – list of dicts for envtypes that dropped too much.
        ``improvements`` – list of dicts for envtypes that improved.
        ``new_envtypes`` – envtypes present only in current.
        ``missing_envtypes`` – envtypes present only in baseline.
    """
    violations: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    current_keys = set(current_scores.keys())
    baseline_keys = set(baseline_scores.keys())

    new_envtypes = sorted(current_keys - baseline_keys)
    missing_envtypes = sorted(baseline_keys - current_keys)

    for envtype in sorted(current_keys & baseline_keys):
        cur = current_scores[envtype]
        base = baseline_scores[envtype]

        if base == 0.0:
            continue  # avoid division-by-zero; a zero baseline is not a meaningful reference

        drop_pct = (base - cur) / base * 100.0
        gain_pct = (cur - base) / base * 100.0

        # Violation: current dropped more than threshold below baseline
        if cur < base * (1.0 - threshold_pct / 100.0):
            violations.append({
                "envtype": envtype,
                "baseline": base,
                "current": cur,
                "drop_pct": round(drop_pct, 4),
            })
        elif cur > base:
            improvements.append({
                "envtype": envtype,
                "baseline": base,
                "current": cur,
                "gain_pct": round(gain_pct, 4),
            })

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "improvements": improvements,
        "new_envtypes": new_envtypes,
        "missing_envtypes": missing_envtypes,
    }


# ─── Human-readable report ────────────────────────────────────────────────────

def format_gate_report(result: dict[str, Any]) -> str:
    """Return a human-readable multi-line report string."""
    lines: list[str] = []
    status = "PASS" if result["pass"] else "FAIL"
    lines.append(f"=== CI Benchmark Gate: {status} ===")
    lines.append("")

    if result["violations"]:
        lines.append(f"Violations ({len(result['violations'])}):")
        for v in result["violations"]:
            lines.append(
                f"  [{v['envtype']}] baseline={v['baseline']:.4f}  "
                f"current={v['current']:.4f}  drop={v['drop_pct']:.2f}%"
            )
        lines.append("")

    if result["improvements"]:
        lines.append(f"Improvements ({len(result['improvements'])}):")
        for i in result["improvements"]:
            lines.append(
                f"  [{i['envtype']}] baseline={i['baseline']:.4f}  "
                f"current={i['current']:.4f}  gain=+{i['gain_pct']:.2f}%"
            )
        lines.append("")

    if result["new_envtypes"]:
        lines.append(f"New envtypes: {', '.join(result['new_envtypes'])}")

    if result["missing_envtypes"]:
        lines.append(f"Missing envtypes: {', '.join(result['missing_envtypes'])}")

    lines.append("")
    return "\n".join(lines)


# ─── JSONL history helpers ────────────────────────────────────────────────────

def update_history(history_path: str, scores: dict[str, float]) -> None:
    """Append one JSONL line ``{timestamp, scores: {envtype: score}}`` to *history_path*."""
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": {str(k): float(v) for k, v in scores.items()},
    }
    with open(history_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def get_latest_baseline(history_path: str) -> dict[str, float]:
    """Return the scores dict from the **last** line of the JSONL history file.

    Returns ``{}`` if the file does not exist or is empty.
    """
    if not os.path.isfile(history_path):
        return {}
    last_line: str | None = None
    with open(history_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if last_line is None:
        return {}
    data = json.loads(last_line)
    return {str(k): float(v) for k, v in data.get("scores", {}).items()}


def run_gate(
    current: dict[str, float],
    history_path: str,
    threshold_pct: float = 5.0,
) -> dict[str, Any]:
    """Convenience: load latest baseline from *history_path* and run ``check_gate``."""
    baseline = get_latest_baseline(history_path)
    return check_gate(current, baseline, threshold_pct=threshold_pct)
