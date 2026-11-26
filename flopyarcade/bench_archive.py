"""B26: Longitudinal benchmark archive (append-only JSONL).

Every CI run appends a single JSON line to the archive file.
Provides per-envtype history, regression detection, trend analysis,
and archive summary.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class ArchiveEntry:
    """A single benchmark archive entry."""
    timestamp: str
    git_sha: str
    scores: dict[str, float]
    metadata: dict | None = None


def append_entry(path: str, entry: ArchiveEntry) -> None:
    """Append one JSON line representing *entry* to the archive at *path*."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    record = {
        "timestamp": entry.timestamp,
        "git_sha": entry.git_sha,
        "scores": entry.scores,
        "metadata": entry.metadata,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_archive(path: str) -> list[ArchiveEntry]:
    """Read all lines from the archive file. Returns [] if file missing."""
    if not os.path.isfile(path):
        return []
    entries: list[ArchiveEntry] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append(ArchiveEntry(
                timestamp=obj["timestamp"],
                git_sha=obj["git_sha"],
                scores=obj["scores"],
                metadata=obj.get("metadata"),
            ))
    return entries


def get_score_history(path: str, envtype: str) -> list[dict]:
    """Return [{timestamp, score, git_sha}] for a single envtype."""
    entries = read_archive(path)
    result = []
    for e in entries:
        if envtype in e.scores:
            result.append({
                "timestamp": e.timestamp,
                "score": e.scores[envtype],
                "git_sha": e.git_sha,
            })
    return result


def detect_regression(path: str, envtype: str, threshold_pct: float = 5.0) -> dict:
    """Detect regression between the last two entries for *envtype*.

    Returns {latest, previous, change_pct, regressed}.
    change_pct = (latest - previous) / previous * 100.
    regressed is True when change_pct < -threshold_pct.
    """
    history = get_score_history(path, envtype)
    if len(history) < 2:
        return {
            "latest": history[0]["score"] if history else None,
            "previous": None,
            "change_pct": None,
            "regressed": False,
        }
    latest = history[-1]["score"]
    previous = history[-2]["score"]
    if previous == 0:
        change_pct = 0.0 if latest == 0 else float("inf")
    else:
        change_pct = (latest - previous) / abs(previous) * 100.0
    regressed = change_pct < -threshold_pct
    return {
        "latest": latest,
        "previous": previous,
        "change_pct": change_pct,
        "regressed": regressed,
    }


def trend_scores(path: str, envtype: str, window: int = 5) -> dict:
    """Simple linear regression on the last *window* entries for *envtype*.

    Returns {slope, intercept, n_points, predicted_next}.
    slope > 0 means scores are improving (positive trend).
    slope < 0 means scores are declining (negative trend).
    """
    history = get_score_history(path, envtype)
    if len(history) == 0:
        return {"slope": 0.0, "intercept": 0.0, "n_points": 0, "predicted_next": 0.0}
    points = history[-window:]
    n = len(points)
    if n == 1:
        return {"slope": 0.0, "intercept": points[0]["score"], "n_points": 1, "predicted_next": points[0]["score"]}
    xs = list(range(n))
    ys = [p["score"] for p in points]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = numerator / denominator if denominator != 0 else 0.0
    intercept = y_mean - slope * x_mean
    predicted_next = slope * n + intercept
    return {
        "slope": slope,
        "intercept": intercept,
        "n_points": n,
        "predicted_next": predicted_next,
    }


def archive_summary(path: str) -> dict:
    """Return {n_entries, envtypes, first_timestamp, last_timestamp, n_regressions}."""
    entries = read_archive(path)
    if not entries:
        return {
            "n_entries": 0,
            "envtypes": [],
            "first_timestamp": None,
            "last_timestamp": None,
            "n_regressions": 0,
        }
    envtypes: set[str] = set()
    for e in entries:
        envtypes.update(e.scores.keys())
    first_timestamp = entries[0].timestamp
    last_timestamp = entries[-1].timestamp
    n_regressions = 0
    for et in sorted(envtypes):
        reg = detect_regression(path, et)
        if reg["regressed"]:
            n_regressions += 1
    return {
        "n_entries": len(entries),
        "envtypes": sorted(envtypes),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "n_regressions": n_regressions,
    }


def format_archive_report(path: str) -> str:
    """Return a human-readable text summary of the archive."""
    summary = archive_summary(path)
    lines: list[str] = []
    lines.append("=" * 50)
    lines.append("  Benchmark Archive Report")
    lines.append("=" * 50)
    lines.append(f"  Entries:          {summary['n_entries']}")
    lines.append(f"  Environment types: {', '.join(summary['envtypes']) or 'N/A'}")
    lines.append(f"  First run:        {summary['first_timestamp'] or 'N/A'}")
    lines.append(f"  Last run:         {summary['last_timestamp'] or 'N/A'}")
    lines.append(f"  Regressions:      {summary['n_regressions']}")
    lines.append("-" * 50)
    for et in summary["envtypes"]:
        reg = detect_regression(path, et)
        trend = trend_scores(path, et)
        status = "REGRESSED" if reg["regressed"] else "OK"
        lines.append(f"  [{et}]")
        lines.append(f"    Latest:      {reg['latest']}")
        lines.append(f"    Previous:    {reg['previous']}")
        if reg["change_pct"] is not None:
            lines.append(f"    Change:      {reg['change_pct']:.2f}%")
        else:
            lines.append("    Change:      N/A")
        lines.append(f"    Status:      {status}")
        lines.append(f"    Trend slope: {trend['slope']:.4f} ({trend['n_points']} pts)")
        lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)
