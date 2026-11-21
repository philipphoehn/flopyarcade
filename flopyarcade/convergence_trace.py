"""MODFLOW convergence trace parser and analyzer (ROADMAP P2).

Parses MODFLOW lst-style logs to extract per-step convergence metrics:
iterations, condition number, residual, convergence status, and elapsed time.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceStep:
    """A single time-step convergence record."""

    step: int
    n_iterations: int
    condition_number: float
    residual: float
    converged: bool
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Structured format patterns
_RE_STEP = re.compile(r"^\s*STEP\s+(\d+)", re.IGNORECASE)
_RE_ITERATIONS = re.compile(r"^\s*Iterations:\s*(\d+)", re.IGNORECASE)
_RE_COND = re.compile(r"^\s*Condition\s+Number:\s*([\d.eE+\-]+)", re.IGNORECASE)
_RE_RESIDUAL = re.compile(r"^\s*Residual:\s*([\d.eE+\-]+)", re.IGNORECASE)
_RE_CONVERGED = re.compile(r"^\s*Converged:\s*(Yes|No)", re.IGNORECASE)
_RE_ELAPSED = re.compile(r"^\s*Elapsed:\s*([\d.]+)\s*ms", re.IGNORECASE)

# Numeric tab/space-separated format: iterations  cond_num  residual  YES/NO  elapsed_ms
_RE_NUMERIC = re.compile(
    r"^\s*(\d+)\s+"
    r"(\d+\.?\d*[eE][+\-]\d+|\d+\.?\d*)\s+"
    r"(\d+\.?\d*[eE][+\-]\d+|\d+\.?\d*)\s+"
    r"(YES|NO|Yes|No|yes|no)\s+"
    r"([\d.]+)\s*$"
)


def _try_parse_numeric(line: str) -> tuple[int, float, float, bool, float] | None:
    """Attempt to parse a numeric-space-separated line.

    Returns (iterations, cond, residual, converged, elapsed_ms) or None.
    """
    m = _RE_NUMERIC.match(line)
    if not m:
        return None
    iterations = int(m.group(1))
    condition_number = float(m.group(2))
    residual = float(m.group(3))
    converged = m.group(4).upper() == "YES"
    elapsed_ms = float(m.group(5))
    return iterations, condition_number, residual, converged, elapsed_ms


def parse_convergence_log(text: str) -> list[ConvergenceStep]:
    """Parse a MODFLOW lst-style convergence log.

    Supports two formats:

    1. Structured::

        STEP 1
          Iterations: 12
          Condition Number: 1.5e10
          Residual: 3.2e-8
          Converged: Yes
          Elapsed: 45.2 ms

    2. Numeric (tab/space separated)::

          12  1.5e+10  3.2e-08  YES  45.2

    Malformed lines are silently skipped.
    Returns an empty list for empty/whitespace-only input.
    """
    if not text or not text.strip():
        return []

    steps: list[ConvergenceStep] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # --- Structured format: STEP N header ---
        m = _RE_STEP.match(line)
        if m:
            step_num = int(m.group(1))
            # Collect fields from following indented lines
            iters: int | None = None
            cond: float | None = None
            resid: float | None = None
            converged: bool | None = None
            elapsed: float | None = None
            j = i + 1
            while j < n:
                sub = lines[j]
                mi = _RE_ITERATIONS.match(sub)
                mc = _RE_COND.match(sub)
                mr = _RE_RESIDUAL.match(sub)
                mv = _RE_CONVERGED.match(sub)
                me = _RE_ELAPSED.match(sub)
                if mi:
                    iters = int(mi.group(1))
                elif mc:
                    cond = float(mc.group(1))
                elif mr:
                    resid = float(mr.group(1))
                elif mv:
                    converged = mv.group(1).lower() == "yes"
                elif me:
                    elapsed = float(me.group(1))
                else:
                    # Non-field line ends the block
                    break
                j += 1
            if iters is not None:
                steps.append(
                    ConvergenceStep(
                        step=step_num,
                        n_iterations=iters,
                        condition_number=cond if cond is not None else 0.0,
                        residual=resid if resid is not None else 0.0,
                        converged=converged if converged is not None else True,
                        elapsed_ms=elapsed if elapsed is not None else 0.0,
                    )
                )
                i = j
                continue

        # --- Numeric format ---
        parsed = _try_parse_numeric(line)
        if parsed:
            iters, cond, resid, converged, elapsed = parsed
            steps.append(
                ConvergenceStep(
                    step=len(steps) + 1,
                    n_iterations=iters,
                    condition_number=cond,
                    residual=resid,
                    converged=converged,
                    elapsed_ms=elapsed,
                )
            )

        i += 1

    return steps


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ConvergenceAnalyzer:
    """Compute aggregate convergence metrics over a list of steps."""

    def __init__(self, steps: list[ConvergenceStep]):
        self.steps = steps

    def max_iterations(self) -> int:
        """Return the maximum iteration count across all steps."""
        if not self.steps:
            return 0
        return max(s.n_iterations for s in self.steps)

    def avg_iterations(self) -> float:
        """Return the mean iteration count."""
        if not self.steps:
            return 0.0
        return sum(s.n_iterations for s in self.steps) / len(self.steps)

    def max_condition_number(self) -> float:
        """Return the maximum condition number."""
        if not self.steps:
            return 0.0
        return max(s.condition_number for s in self.steps)

    def worst_step(self) -> ConvergenceStep:
        """Return the step with the highest iteration count."""
        if not self.steps:
            raise ValueError("No steps to analyze")
        return max(self.steps, key=lambda s: s.n_iterations)

    def convergence_rate(self) -> float:
        """Return the fraction of steps that converged (0.0–1.0)."""
        if not self.steps:
            return 0.0
        converged = sum(1 for s in self.steps if s.converged)
        return converged / len(self.steps)

    def is_stable(self, threshold: int = 100) -> bool:
        """Return True if no step exceeds *threshold* iterations."""
        return all(s.n_iterations <= threshold for s in self.steps)

    def trend(self) -> dict:
        """Compare average iterations of first vs last 10 steps.

        Returns a dict with keys: first_avg_10, last_avg_10, improving.
        """
        n = len(self.steps)
        if n == 0:
            return {"first_avg_10": 0.0, "last_avg_10": 0.0, "improving": True}

        k = min(10, n)
        first_slice = self.steps[:k]
        last_slice = self.steps[-k:]
        first_avg = sum(s.n_iterations for s in first_slice) / k
        last_avg = sum(s.n_iterations for s in last_slice) / k
        return {
            "first_avg_10": first_avg,
            "last_avg_10": last_avg,
            "improving": last_avg <= first_avg,
        }

    def summary(self) -> dict:
        """Return a dict of all key metrics."""
        return {
            "n_steps": len(self.steps),
            "max_iterations": self.max_iterations(),
            "avg_iterations": self.avg_iterations(),
            "max_condition_number": self.max_condition_number(),
            "worst_step": asdict(self.worst_step()) if self.steps else None,
            "convergence_rate": self.convergence_rate(),
            "is_stable": self.is_stable(),
            "trend": self.trend(),
        }

    # -- persistence --

    def save(self, path: str | Path) -> None:
        """Serialize to JSON at *path*."""
        data = {
            "steps": [asdict(s) for s in self.steps],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ConvergenceAnalyzer:
        """Load from a JSON file saved by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        steps = [ConvergenceStep(**s) for s in data["steps"]]
        return cls(steps)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_convergence_report(analyzer: ConvergenceAnalyzer) -> str:
    """Render a human-readable text table of convergence metrics."""
    lines: list[str] = []
    header = f"{'Step':>6} | {'Iter':>6} | {'Cond#':>12} | {'Residual':>12} | {'Conv':>5} | {'ms':>8}"
    lines.append("=" * len(header))
    lines.append("  MODFLOW Convergence Report")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for s in analyzer.steps:
        lines.append(
            f"{s.step:>6} | {s.n_iterations:>6} | {s.condition_number:>12.3e} "
            f"| {s.residual:>12.3e} | {'Yes' if s.converged else 'No':>5} | {s.elapsed_ms:>8.1f}"
        )

    lines.append("-" * len(header))
    sm = analyzer.summary()
    lines.append(f"  Steps: {sm['n_steps']}  |  Max iter: {sm['max_iterations']}  |  "
                 f"Avg iter: {sm['avg_iterations']:.1f}  |  "
                 f"Convergence rate: {sm['convergence_rate']:.1%}")
    lines.append(f"  Stable (≤100 iter): {'Yes' if sm['is_stable'] else 'No'}")
    tr = sm["trend"]
    lines.append(f"  Trend: first-10 avg {tr['first_avg_10']:.1f} → last-10 avg {tr['last_avg_10']:.1f} "
                 f"({'improving' if tr['improving'] else 'worsening'})")
    lines.append("=" * len(header))
    return "\n".join(lines)
