"""Hydraulic head change heatmap with flow arrows.

ROADMAP V2: Δh per well, annotated with flow arrows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CellChange:
    """A single cell's hydraulic head change."""
    row: int
    col: int
    delta_h: float
    flow_direction: Optional[Tuple[float, float]] = None


class HeadChangeGrid:
    """Grid of hydraulic head changes with gradient and flow computation."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self._cells: dict[tuple[int, int], CellChange] = {}

    def set_change(self, row: int, col: int, delta_h: float, flow_dir=None):
        """Record a head change for one cell."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(
                f"Cell ({row}, {col}) out of bounds for {self.rows}x{self.cols} grid"
            )
        self._cells[(row, col)] = CellChange(
            row=row, col=col, delta_h=delta_h, flow_direction=flow_dir
        )

    def cell(self, row: int, col: int) -> CellChange:
        """Return CellChange for (row, col); zero-valued if unset."""
        key = (row, col)
        if key not in self._cells:
            return CellChange(row=row, col=col, delta_h=0.0, flow_direction=None)
        return self._cells[key]

    def _all_deltas(self) -> list[float]:
        return [c.delta_h for c in self._cells.values()]

    def max_change(self) -> CellChange:
        """Cell with largest |delta_h|."""
        if not self._cells:
            raise ValueError("No cells set")
        return max(self._cells.values(), key=lambda c: abs(c.delta_h))

    def min_change(self) -> CellChange:
        """Cell with smallest |delta_h| among set cells."""
        if not self._cells:
            raise ValueError("No cells set")
        return min(self._cells.values(), key=lambda c: abs(c.delta_h))

    def mean_change(self) -> float:
        """Mean delta_h over all set cells."""
        if not self._cells:
            return 0.0
        return sum(self._all_deltas()) / len(self._cells)

    def rms_change(self) -> float:
        """Root-mean-square of delta_h over all set cells."""
        if not self._cells:
            return 0.0
        n = len(self._cells)
        return math.sqrt(sum(d ** 2 for d in self._all_deltas()) / n)

    def heatmap_data(self) -> list[list[float]]:
        """Return 2D list of delta_h values ready for imshow."""
        grid = []
        for r in range(self.rows):
            row_data = [self.cell(r, c).delta_h for c in range(self.cols)]
            grid.append(row_data)
        return grid

    def gradient_field(self) -> list[list[tuple[float, float]]]:
        """Compute ∇h via finite differences (central interior, one-sided edges).

        Returns grid[r][c] = (dh_dr, dh_dc).
        """
        data = self.heatmap_data()
        grad: list[list[tuple[float, float]]] = []
        for r in range(self.rows):
            row_grad: list[tuple[float, float]] = []
            for c in range(self.cols):
                # dh/dr
                if self.rows == 1:
                    dh_dr = 0.0
                elif r == 0:
                    dh_dr = data[1][c] - data[0][c]
                elif r == self.rows - 1:
                    dh_dr = data[r][c] - data[r - 1][c]
                else:
                    dh_dr = (data[r + 1][c] - data[r - 1][c]) / 2.0

                # dh/dc
                if self.cols == 1:
                    dh_dc = 0.0
                elif c == 0:
                    dh_dc = data[r][1] - data[r][0]
                elif c == self.cols - 1:
                    dh_dc = data[r][c] - data[r][c - 1]
                else:
                    dh_dc = (data[r][c + 1] - data[r][c - 1]) / 2.0

                row_grad.append((dh_dr, dh_dc))
            grad.append(row_grad)
        return grad

    def flow_arrows(self, max_arrow_cells: int = 20) -> list[dict]:
        """Return top-N cells by |∇h|.

        Each entry: {row, col, direction: (dx, dy) unit vector, magnitude}.
        """
        grad = self.gradient_field()
        candidates = []
        for r in range(self.rows):
            for c in range(self.cols):
                dh_dr, dh_dc = grad[r][c]
                magnitude = math.sqrt(dh_dr ** 2 + dh_dc ** 2)
                if magnitude > 0:
                    direction = (dh_dr / magnitude, dh_dc / magnitude)
                else:
                    direction = (0.0, 0.0)
                candidates.append(
                    {
                        "row": r,
                        "col": c,
                        "direction": direction,
                        "magnitude": magnitude,
                    }
                )
        candidates.sort(key=lambda x: x["magnitude"], reverse=True)
        return candidates[:max_arrow_cells]

    def annotation_text(self, cell: CellChange) -> str:
        """Formatted annotation, e.g. 'Δh=+0.52 → (1.0,0.0)'."""
        sign = "+" if cell.delta_h >= 0 else ""
        text = f"Δh={sign}{cell.delta_h:.2f}"
        if cell.flow_direction is not None:
            dx, dy = cell.flow_direction
            text += f" → ({dx:.1f},{dy:.1f})"
        return text

    def save_json(self, path: str):
        """Serialise grid to JSON."""
        data = {
            "rows": self.rows,
            "cols": self.cols,
            "cells": [
                {
                    "row": c.row,
                    "col": c.col,
                    "delta_h": c.delta_h,
                    "flow_direction": list(c.flow_direction) if c.flow_direction else None,
                }
                for c in self._cells.values()
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> HeadChangeGrid:
        """Deserialise grid from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        grid = cls(data["rows"], data["cols"])
        for cell_data in data["cells"]:
            fd = (
                tuple(cell_data["flow_direction"])
                if cell_data["flow_direction"]
                else None
            )
            grid.set_change(
                cell_data["row"], cell_data["col"], cell_data["delta_h"], fd
            )
        return grid


def compare_grids(grid_a: HeadChangeGrid, grid_b: HeadChangeGrid) -> dict:
    """Return {mean_diff, max_cell_diff, rms_diff} between two same-size grids."""
    if grid_a.rows != grid_b.rows or grid_a.cols != grid_b.cols:
        raise ValueError("Grids must have the same dimensions")
    diffs = [
        grid_a.cell(r, c).delta_h - grid_b.cell(r, c).delta_h
        for r in range(grid_a.rows)
        for c in range(grid_a.cols)
    ]
    n = len(diffs)
    mean_diff = sum(diffs) / n if n else 0.0
    max_cell_diff = max(abs(d) for d in diffs) if diffs else 0.0
    rms_diff = math.sqrt(sum(d ** 2 for d in diffs) / n) if n else 0.0
    return {
        "mean_diff": mean_diff,
        "max_cell_diff": max_cell_diff,
        "rms_diff": rms_diff,
    }


def format_head_report(grid: HeadChangeGrid) -> str:
    """Human-readable text summary with max/min cells and flow arrows."""
    lines: list[str] = []
    lines.append(f"Head Change Report: {grid.rows}x{grid.cols} grid")
    lines.append("=" * 40)

    if not grid._cells:
        lines.append("No changes recorded.")
        return "\n".join(lines)

    max_c = grid.max_change()
    min_c = grid.min_change()
    lines.append(
        f"Max change: row={max_c.row}, col={max_c.col}, Δh={max_c.delta_h:+.4f}"
    )
    lines.append(
        f"Min change: row={min_c.row}, col={min_c.col}, Δh={min_c.delta_h:+.4f}"
    )
    lines.append(f"Mean change: {grid.mean_change():+.4f}")
    lines.append(f"RMS change: {grid.rms_change():.4f}")

    arrows = grid.flow_arrows(max_arrow_cells=5)
    if arrows:
        lines.append("")
        lines.append("Top flow arrows:")
        for a in arrows:
            lines.append(
                f"  ({a['row']},{a['col']}) "
                f"dir=({a['direction'][0]:+.3f},{a['direction'][1]:+.3f}) "
                f"mag={a['magnitude']:.4f}"
            )
    return "\n".join(lines)
