"""S2: Heterogeneous aquifer environment with layered K and lens geometry.

Env type: 8s-d
Provides contrast ratio, effective K (harmonic mean), capture-zone efficiency,
simplified Darcy flow path, and comparison vs homogeneous baseline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KLayer:
    """A horizontal layer of constant hydraulic conductivity."""
    name: str
    k_value: float
    row_start: int  # inclusive
    row_end: int    # inclusive


@dataclass
class LensGeometry:
    """A circular lens region that multiplies the base K."""
    center: tuple[int, int]  # (row, col)
    radius: int
    k_multiplier: float


@dataclass
class AquiferConfig:
    """Configuration for a heterogeneous aquifer grid."""
    rows: int = 10
    cols: int = 10
    layers: list[KLayer] = field(default_factory=list)
    lens: Optional[LensGeometry] = None
    base_k: float = 1.0
    envtype: str = '8s-d'


class HeterogeneousAquifer:
    """Heterogeneous aquifer model with layered K and optional lens geometry."""

    def __init__(self, config: AquiferConfig):
        self.config = config

    # ── Grid access ──────────────────────────────────────────────────────

    def k_grid(self) -> list[list[float]]:
        """Compute K value for each cell: layer K × lens multiplier (if within radius)."""
        grid = []
        for r in range(self.config.rows):
            row_vals = []
            for c in range(self.config.cols):
                row_vals.append(self.k_at(r, c))
            grid.append(row_vals)
        return grid

    def k_at(self, row: int, col: int) -> float:
        """Return hydraulic conductivity at a specific cell."""
        k = self._layer_k(row)
        if self.config.lens is not None:
            lr, lc = self.config.lens.center
            dist = math.sqrt((row - lr) ** 2 + (col - lc) ** 2)
            if dist <= self.config.lens.radius:
                k *= self.config.lens.k_multiplier
        return k

    def _layer_k(self, row: int) -> float:
        """Return the layer-based K for a given row."""
        for layer in self.config.layers:
            if layer.row_start <= row <= layer.row_end:
                return layer.k_value
        return self.config.base_k

    # ── Properties ───────────────────────────────────────────────────────

    def n_layers(self) -> int:
        return len(self.config.layers)

    def has_lens(self) -> bool:
        return self.config.lens is not None

    def contrast_ratio(self) -> float:
        """max(K)/min(K) over all cells."""
        grid = self.k_grid()
        all_vals = [v for row in grid for v in row]
        mn = min(all_vals)
        mx = max(all_vals)
        if mn == 0:
            return float('inf')
        return mx / mn

    def lens_cells(self) -> list[tuple[int, int]]:
        """Cells within lens radius (Euclidean)."""
        if self.config.lens is None:
            return []
        lr, lc = self.config.lens.center
        rad = self.config.lens.radius
        cells = []
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                if math.sqrt((r - lr) ** 2 + (c - lc) ** 2) <= rad:
                    cells.append((r, c))
        return cells

    def lens_fraction(self) -> float:
        total = self.config.rows * self.config.cols
        return len(self.lens_cells()) / total

    # ── Aggregates ───────────────────────────────────────────────────────

    def homogeneous_equivalent(self) -> float:
        """Arithmetic mean of K grid."""
        grid = self.k_grid()
        all_vals = [v for row in grid for v in row]
        return sum(all_vals) / len(all_vals)

    def effective_k(self) -> float:
        """Harmonic mean (for flow): 1 / (mean(1/K))."""
        grid = self.k_grid()
        all_vals = [v for row in grid for v in row]
        inv_sum = sum(1.0 / v for v in all_vals)
        return 1.0 / (inv_sum / len(all_vals))

    def transmissivity(self, thickness: float = 1.0) -> list[list[float]]:
        """K × thickness per cell."""
        return [[k * thickness for k in row] for row in self.k_grid()]

    # ── Analysis ─────────────────────────────────────────────────────────

    def capture_zone_efficiency(
        self,
        capture_cells: list[tuple[int, int]],
        source_pos: tuple[int, int],
    ) -> dict:
        """Capture-zone efficiency heuristic."""
        n = len(capture_cells)
        if n == 0:
            return {
                'n_capture_cells': 0,
                'mean_k_capture': 0.0,
                'mean_k_global': self.homogeneous_equivalent(),
                'ratio': 1.0,
                'efficiency_estimate': 1.0,
            }
        ks = [self.k_at(r, c) for r, c in capture_cells]
        mean_k_capture = sum(ks) / n
        mean_k_global = self.homogeneous_equivalent()
        ratio = mean_k_capture / mean_k_global if mean_k_global != 0 else 1.0
        efficiency_estimate = ratio ** 0.5
        return {
            'n_capture_cells': n,
            'mean_k_capture': mean_k_capture,
            'mean_k_global': mean_k_global,
            'ratio': ratio,
            'efficiency_estimate': efficiency_estimate,
        }

    def flow_direction_estimate(
        self,
        source: tuple[int, int],
        sink: tuple[int, int],
    ) -> list[dict]:
        """Simplified Darcy flow path: at each step move toward sink weighted by local K.

        Returns list of {row, col, k, step}.
        """
        path = []
        r, c = source
        max_steps = self.config.rows * self.config.cols * 2
        for step in range(max_steps):
            k_here = self.k_at(r, c)
            path.append({'row': r, 'col': c, 'k': k_here, 'step': step})
            if (r, c) == sink:
                break
            # Candidate moves: 8-neighbourhood weighted by K and proximity to sink
            candidates = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.config.rows and 0 <= nc < self.config.cols:
                        k_candidate = self.k_at(nr, nc)
                        dist_to_sink = math.sqrt((nr - sink[0]) ** 2 + (nc - sink[1]) ** 2)
                        weight = k_candidate / (dist_to_sink + 1e-9)
                        candidates.append((weight, nr, nc))
            if not candidates:
                break
            candidates.sort(key=lambda x: -x[0])
            best = candidates[0]
            r, c = best[1], best[2]
        return path

    def compare_homogeneous(self) -> dict:
        """Comparison metrics vs homogeneous baseline."""
        return {
            'het_mean_k': self.homogeneous_equivalent(),
            'het_effective_k': self.effective_k(),
            'homo_k': self.config.base_k,
            'contrast': self.contrast_ratio(),
            'n_layers': self.n_layers(),
            'has_lens': self.has_lens(),
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save_config(self, path: str) -> None:
        """Save config as JSON."""
        data = {
            'rows': self.config.rows,
            'cols': self.config.cols,
            'layers': [
                {'name': l.name, 'k_value': l.k_value,
                 'row_start': l.row_start, 'row_end': l.row_end}
                for l in self.config.layers
            ],
            'lens': (
                {'center': list(self.config.lens.center),
                 'radius': self.config.lens.radius,
                 'k_multiplier': self.config.lens.k_multiplier}
                if self.config.lens else None
            ),
            'base_k': self.config.base_k,
            'envtype': self.config.envtype,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> HeterogeneousAquifer:
        """Load config from JSON and construct HeterogeneousAquifer."""
        with open(path, 'r') as f:
            data = json.load(f)
        layers = [
            KLayer(
                name=d['name'], k_value=d['k_value'],
                row_start=d['row_start'], row_end=d['row_end'],
            )
            for d in data.get('layers', [])
        ]
        lens = None
        if data.get('lens') is not None:
            ld = data['lens']
            lens = LensGeometry(
                center=tuple(ld['center']),
                radius=ld['radius'],
                k_multiplier=ld['k_multiplier'],
            )
        config = AquiferConfig(
            rows=data['rows'],
            cols=data['cols'],
            layers=layers,
            lens=lens,
            base_k=data.get('base_k', 1.0),
            envtype=data.get('envtype', '8s-d'),
        )
        return cls(config)


# ── Config generators ──────────────────────────────────────────────────────────

def generate_layered_config(rows: int = 10, cols: int = 10, n_layers: int = 3) -> AquiferConfig:
    """Generate config with n_layers equal-width layers, K = [1, 5, 0.2, ...]."""
    k_values = [1.0, 5.0, 0.2, 10.0, 0.5]
    layer_height = rows // n_layers
    layers = []
    for i in range(n_layers):
        rs = i * layer_height
        re = (i + 1) * layer_height - 1 if i < n_layers - 1 else rows - 1
        layers.append(KLayer(
            name=f'layer_{i}',
            k_value=k_values[i % len(k_values)],
            row_start=rs,
            row_end=re,
        ))
    return AquiferConfig(rows=rows, cols=cols, layers=layers)


def generate_lens_config(
    rows: int = 10,
    cols: int = 10,
    lens_center: tuple[int, int] = (5, 5),
    radius: int = 3,
    k_mult: float = 10.0,
) -> AquiferConfig:
    """Generate config with a single lens (no layers)."""
    return AquiferConfig(
        rows=rows,
        cols=cols,
        lens=LensGeometry(center=lens_center, radius=radius, k_multiplier=k_mult),
    )


def generate_combined_config(rows: int = 10, cols: int = 10) -> AquiferConfig:
    """Generate config with both 3 layers and a lens."""
    layered = generate_layered_config(rows=rows, cols=cols, n_layers=3)
    layered.lens = LensGeometry(center=(rows // 2, cols // 2), radius=3, k_multiplier=10.0)
    return layered


# ── Reporting ──────────────────────────────────────────────────────────────────

def format_aquifer_report(aq: HeterogeneousAquifer) -> str:
    """Produce a human-readable text report."""
    lines = []
    lines.append(f"envtype: {aq.config.envtype}")
    lines.append(f"grid: {aq.config.rows} x {aq.config.cols}")
    lines.append(f"n_layers: {aq.n_layers()}")
    lines.append(f"has_lens: {aq.has_lens()}")
    lines.append(f"contrast_ratio: {aq.contrast_ratio():.4f}")
    lines.append(f"homogeneous_equivalent (arithmetic mean K): {aq.homogeneous_equivalent():.4f}")
    lines.append(f"effective_k (harmonic mean): {aq.effective_k():.4f}")
    if aq.has_lens():
        lines.append(f"lens_center: {aq.config.lens.center}")
        lines.append(f"lens_radius: {aq.config.lens.radius}")
        lines.append(f"lens_k_multiplier: {aq.config.lens.k_multiplier}")
        lines.append(f"lens_cells: {len(aq.lens_cells())}")
        lines.append(f"lens_fraction: {aq.lens_fraction():.4f}")
    return '\n'.join(lines)
