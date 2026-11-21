"""V7: Well-activation timeline Gantt.

Per-step pumping rate per well; stacked bar and Gantt chart data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class WellStep:
    """A single (step, well) entry in the pumping timeline."""
    step: int
    well_id: int
    pumping_rate: float
    active: bool


class WellTimeline:
    """Track per-step pumping rates for a set of wells."""

    def __init__(self, n_wells: int, n_steps: int) -> None:
        self.n_wells = n_wells
        self.n_steps = n_steps
        # rates[well_id][step]
        self._rates: list[list[float]] = [[0.0] * n_steps for _ in range(n_wells)]

    def set_rate(self, step: int, well_id: int, rate: float) -> None:
        """Set pumping rate; auto-sets active = (rate != 0)."""
        if not (0 <= step < self.n_steps):
            raise IndexError(f"step {step} out of range [0, {self.n_steps})")
        if not (0 <= well_id < self.n_wells):
            raise IndexError(f"well_id {well_id} out of range [0, {self.n_wells})")
        self._rates[well_id][step] = rate

    def get_rate(self, step: int, well_id: int) -> float:
        """Return the pumping rate for a given step and well."""
        return self._rates[well_id][step]

    def well_timeline(self, well_id: int) -> list[WellStep]:
        """Return all WellStep entries for one well."""
        return [
            WellStep(
                step=s,
                well_id=well_id,
                pumping_rate=self._rates[well_id][s],
                active=self._rates[well_id][s] != 0.0,
            )
            for s in range(self.n_steps)
        ]

    def total_pumpage(self, well_id: int) -> float:
        """Sum of |rate| over all steps for a well."""
        return sum(abs(r) for r in self._rates[well_id])

    def active_fraction(self, well_id: int) -> float:
        """Fraction of steps where rate != 0."""
        active = sum(1 for r in self._rates[well_id] if r != 0.0)
        return active / self.n_steps

    def gantt_data(self) -> dict[str, Any]:
        """Return Gantt-chart-ready data.

        Keys: wells (list of ids), steps (list of 0..n-1),
              rates (per-well list of per-step rates),
              active (per-well list of per-step booleans).
        """
        wells = list(range(self.n_wells))
        steps = list(range(self.n_steps))
        rates = [list(self._rates[w]) for w in range(self.n_wells)]
        active = [[r != 0.0 for r in self._rates[w]] for w in range(self.n_wells)]
        return {"wells": wells, "steps": steps, "rates": rates, "active": active}

    def stacked_bar_data(self) -> dict[str, Any]:
        """Return stacked-bar-chart-ready data.

        Keys: steps, per_well (dict well_id -> [rates]), total (sum per step).
        """
        steps = list(range(self.n_steps))
        per_well: dict[int, list[float]] = {}
        for w in range(self.n_wells):
            per_well[w] = list(self._rates[w])
        total = [sum(self._rates[w][s] for w in range(self.n_wells)) for s in range(self.n_steps)]
        return {"steps": steps, "per_well": per_well, "total": total}

    def peak_step(self, well_id: int) -> dict[str, Any]:
        """Return {step, rate} of the step with max |rate| for a well."""
        best_step = 0
        best_abs = -1.0
        for s in range(self.n_steps):
            val = abs(self._rates[well_id][s])
            if val > best_abs:
                best_abs = val
                best_step = s
        return {"step": best_step, "rate": self._rates[well_id][best_step]}

    def duty_cycle(self, well_id: int) -> float:
        """Alias for active_fraction."""
        return self.active_fraction(well_id)

    def save_json(self, path: str) -> None:
        """Serialize to JSON file."""
        data = {
            "n_wells": self.n_wells,
            "n_steps": self.n_steps,
            "rates": self._rates,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> WellTimeline:
        """Deserialize from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        tl = cls(n_wells=data["n_wells"], n_steps=data["n_steps"])
        tl._rates = data["rates"]
        return tl


def compare_timelines(tl_a: WellTimeline, tl_b: WellTimeline) -> dict[str, Any]:
    """Compare two timelines per-well.

    Returns: {wells: [{well_id, pumpage_a, pumpage_b, diff, faster}]}
    """
    wells = []
    for w in range(tl_a.n_wells):
        pa = tl_a.total_pumpage(w)
        pb = tl_b.total_pumpage(w)
        diff = pa - pb
        faster = "a" if pa > pb else ("b" if pb > pa else "equal")
        wells.append({
            "well_id": w,
            "pumpage_a": pa,
            "pumpage_b": pb,
            "diff": diff,
            "faster": faster,
        })
    return {"wells": wells}


def format_gantt_report(timeline: WellTimeline) -> str:
    """Return a text table: well | total pumpage | active% | peak step | peak rate."""
    header = f"{'well':>6} | {'total pumpage':>14} | {'active%':>8} | {'peak step':>9} | {'peak rate':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for w in range(timeline.n_wells):
        pumpage = timeline.total_pumpage(w)
        frac = timeline.active_fraction(w)
        peak = timeline.peak_step(w)
        lines.append(
            f"{w:>6} | {pumpage:>14.4f} | {frac * 100:>7.1f}% | {peak['step']:>9} | {peak['rate']:>10.4f}"
        )
    return "\n".join(lines)
