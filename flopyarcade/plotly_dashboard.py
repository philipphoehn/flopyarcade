"""Interactive Plotly dashboard for FloPyArcade visualisations.

ROADMAP V3: HDF5 → 3D scatter + time slider + well selector + head contour.

This module builds self-contained Plotly figure dicts (pure Python)
and can export them as standalone HTML files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DashboardConfig:
    """Configuration for the Plotly dashboard."""

    title: str = "FloPyArcade Dashboard"
    n_frames: int = 5
    grid_shape: tuple[int, int] = (5, 5)
    well_positions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class FrameData:
    """A single time-step snapshot of the simulation."""

    t: int
    heads: list[list[float]]
    particle_positions: Optional[list[list[float]]] = None
    active_wells: Optional[list[int]] = None


class DashboardBuilder:
    """Builds Plotly figure dicts for interactive dashboards."""

    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self._frames: list[FrameData] = []

    def add_frame(
        self,
        t: int,
        heads: list[list[float]],
        particle_positions: Optional[list[list[float]]] = None,
        active_wells: Optional[list[int]] = None,
    ) -> None:
        """Append a FrameData snapshot to the builder."""
        frame = FrameData(
            t=t,
            heads=heads,
            particle_positions=particle_positions,
            active_wells=active_wells,
        )
        self._frames.append(frame)

    def frames(self) -> list[FrameData]:
        """Return all accumulated frames."""
        return list(self._frames)

    # ------------------------------------------------------------------
    # Figure builders
    # ------------------------------------------------------------------

    def build_3d_scatter(self) -> dict:
        """Return a Plotly figure dict: 3D scatter of particle positions
        over time with a slider for frame selection."""
        data = []
        frame_objs = []

        for i, frame in enumerate(self._frames):
            if frame.particle_positions is None:
                continue
            x = [p[0] for p in frame.particle_positions]
            y = [p[1] for p in frame.particle_positions]
            z = [p[2] if len(p) > 2 else 0.0 for p in frame.particle_positions]
            trace_name = f"frame_{frame.t}"
            trace = {
                "type": "scatter3d",
                "x": x,
                "y": y,
                "z": z,
                "mode": "markers",
                "name": trace_name,
                "frame": {"name": trace_name},
                "marker": {"size": 5, "color": "royalblue"},
            }
            data.append(trace)
            frame_objs.append({"name": trace_name, "data": [trace]})

        # Slider steps
        steps = []
        for i, frame in enumerate(self._frames):
            if frame.particle_positions is None:
                continue
            trace_name = f"frame_{frame.t}"
            steps.append({
                "method": "animate",
                "label": f"t={frame.t}",
                "value": trace_name,
                "array": [],
            })

        layout = {
            "title": self.config.title,
            "scene": {
                "xaxis": {"title": "Row"},
                "yaxis": {"title": "Col"},
                "zaxis": {"title": "Head"},
            },
            "sliders": [
                {
                    "steps": steps,
                    "activecurrent": True,
                    "x": 0.1,
                    "y": 0.05,
                    "len": 0.8,
                }
            ],
            "updatemenus": [
                {
                    "type": "buttons",
                    "buttons": [
                        {"label": "Play", "method": "animate", "args": [None]},
                        {"label": "Pause", "method": "animate", "args": ["paused"]},
                    ],
                    "x": 0.75,
                    "y": 0.05,
                }
            ],
            "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
        }

        return {"data": data, "layout": layout, "frames": frame_objs}

    def build_head_contour(self) -> dict:
        """Return a Plotly figure dict: 2D contour of the head field per
        frame with a slider."""
        data = []
        frame_objs = []

        for i, frame in enumerate(self._frames):
            rows = len(frame.heads)
            cols = len(frame.heads[0]) if rows else 0
            x = list(range(cols))
            y = list(range(rows))
            trace_name = f"contour_t{frame.t}"
            trace = {
                "type": "contour",
                "x": x,
                "y": y,
                "z": frame.heads,
                "name": trace_name,
                "colorscale": "Viridis",
                "ncontours": 20,
            }
            data.append(trace)
            frame_objs.append({"name": trace_name, "data": [trace]})

        steps = []
        for i, frame in enumerate(self._frames):
            trace_name = f"contour_t{frame.t}"
            steps.append({
                "method": "animate",
                "label": f"t={frame.t}",
                "value": trace_name,
                "array": [],
            })

        layout = {
            "title": f"{self.config.title} – Head Contour",
            "xaxis": {"title": "Column"},
            "yaxis": {"title": "Row"},
            "sliders": [
                {
                    "steps": steps,
                    "activecurrent": True,
                    "x": 0.1,
                    "y": 0.05,
                    "len": 0.8,
                }
            ],
            "updatemenus": [
                {
                    "type": "buttons",
                    "buttons": [
                        {"label": "Play", "method": "animate", "args": [None]},
                        {"label": "Pause", "method": "animate", "args": ["paused"]},
                    ],
                    "x": 0.75,
                    "y": 0.05,
                }
            ],
            "margin": {"l": 60, "r": 40, "t": 50, "b": 60},
        }

        return {"data": data, "layout": layout, "frames": frame_objs}

    def build_well_selector(self) -> dict:
        """Return the updatemenus structure for well filtering."""
        wells = self.config.well_positions
        buttons = []

        # "All wells" button
        buttons.append({
            "label": "All wells",
            "method": "update",
            "args": [{"visible": [True] * max(len(wells), 1)}, None],
        })

        for idx, (r, c) in enumerate(wells):
            visible = [False] * len(wells)
            visible[idx] = True
            buttons.append({
                "label": f"Well {idx} ({r},{c})",
                "method": "update",
                "args": [{"visible": visible}, None],
            })

        return {
            "updatemenus": [
                {
                    "type": "buttons",
                    "buttons": buttons,
                    "x": 0.05,
                    "y": 1.15,
                    "xanchor": "left",
                }
            ]
        }

    def build_combined(self) -> dict:
        """Return a full dashboard figure combining all traces."""
        scatter = self.build_3d_scatter()
        contour = self.build_head_contour()
        well_sel = self.build_well_selector()

        data = scatter.get("data", []) + contour.get("data", [])
        frames = scatter.get("frames", []) + contour.get("frames", [])

        layout = {
            "title": self.config.title,
            "scene": {
                "xaxis": {"title": "Row"},
                "yaxis": {"title": "Col"},
                "zaxis": {"title": "Head"},
            },
            "xaxis": {"title": "Column"},
            "yaxis": {"title": "Row"},
        }

        # Merge sliders from contour (2D) as primary
        if "sliders" in contour.get("layout", {}):
            layout["sliders"] = contour["layout"]["sliders"]
        elif "sliders" in scatter.get("layout", {}):
            layout["sliders"] = scatter["layout"]["sliders"]

        # Combine updatemenus: well selector + play/pause
        updatemenus = []
        if "updatemenus" in well_sel:
            updatemenus.extend(well_sel["updatemenus"])
        if "updatemenus" in scatter.get("layout", {}):
            updatemenus.extend(scatter["layout"]["updatemenus"])
        if updatemenus:
            layout["updatemenus"] = updatemenus

        layout.setdefault("margin", {"l": 0, "r": 0, "t": 50, "b": 0})

        return {"data": data, "layout": layout, "frames": frames}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_html(self, path: str) -> None:
        """Write a self-contained HTML file with embedded Plotly."""
        figure = self.build_combined()
        figure_json = json.dumps(figure)

        html = (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n"
            "</head>\n"
            "<body>\n"
            "<div id='plot'></div>\n"
            f"<script>Plotly.newPlot('plot', {figure_json});</script>\n"
            "</body>\n"
            "</html>\n"
        )

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)

    def to_json(self) -> dict:
        """Return a JSON-serializable figure dict."""
        return self.build_combined()


# ----------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------


def build_minimal_dashboard() -> DashboardBuilder:
    """Create a small demo dashboard: 5×5 grid, 3 wells, 5 frames.

    Useful for testing and quick demos.
    """
    config = DashboardConfig(
        title="FloPyArcade Minimal Dashboard",
        n_frames=5,
        grid_shape=(5, 5),
        well_positions=[(1, 1), (3, 2), (2, 4)],
    )
    builder = DashboardBuilder(config)

    for t in range(5):
        # Synthetic head field: radial gradient shifting over time
        heads = []
        for r in range(5):
            row = []
            for c in range(5):
                val = 10.0 - 0.5 * ((r - 2) ** 2 + (c - 2) ** 2) + 0.3 * t
                row.append(round(val, 4))
            heads.append(row)

        # Synthetic particle positions (3 particles drifting)
        particles = []
        for w_idx, (wr, wc) in enumerate(config.well_positions):
            px = wr + 0.5 * t
            py = wc + 0.3 * t
            pz = 5.0 - 0.2 * t
            particles.append([px, py, pz])

        active = list(range(len(config.well_positions)))

        builder.add_frame(t, heads, particle_positions=particles, active_wells=active)

    return builder


def format_dashboard_summary(builder: DashboardBuilder) -> str:
    """Return a human-readable summary of the dashboard."""
    cfg = builder.config
    n = len(builder.frames())
    wells = ", ".join(f"({r},{c})" for r, c in cfg.well_positions)
    return (
        f"Dashboard: {cfg.title}\n"
        f"  Grid: {cfg.grid_shape[0]}x{cfg.grid_shape[1]}\n"
        f"  Frames: {n}\n"
        f"  Wells: {wells}\n"
        f"  Status: ready"
    )
