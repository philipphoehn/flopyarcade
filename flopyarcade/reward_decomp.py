"""Reward decomposition: per-component per-episode tracking (ROADMAP P7).

Records individual reward components each episode, computes dominance,
exports structured data for stacked-bar visualisation, and supports
comparison between two reward decompositions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RewardComponent:
    """A single reward component for one episode."""

    name: str
    value: float
    weight: float = 1.0


@dataclass
class EpisodeReward:
    """All reward components for a single episode."""

    episode: int
    components: list[RewardComponent] = field(default_factory=list)
    total: float = 0.0


# ---------------------------------------------------------------------------
# Core decomposer
# ---------------------------------------------------------------------------

class RewardDecomposer:
    """Accumulate per-episode reward component data and derive analytics."""

    def __init__(self, envtype: str, component_names: list[str]) -> None:
        self.envtype: str = envtype
        self.component_names: list[str] = list(component_names)
        self._episodes: list[EpisodeReward] = []

    # -- recording ----------------------------------------------------------

    def record_episode(self, episode: int, component_values: dict[str, float]) -> EpisodeReward:
        """Record one episode's component values and compute the weighted total."""
        components: list[RewardComponent] = []
        total = 0.0
        for name in self.component_names:
            value = component_values.get(name, 0.0)
            weight = 1.0
            components.append(RewardComponent(name=name, value=value, weight=weight))
            total += value * weight
        ep = EpisodeReward(episode=episode, components=components, total=total)
        self._episodes.append(ep)
        return ep

    # -- accessors ----------------------------------------------------------

    def episodes(self) -> list[EpisodeReward]:
        """Return all recorded episodes."""
        return list(self._episodes)

    def n_episodes(self) -> int:
        """Number of recorded episodes."""
        return len(self._episodes)

    # -- analytics ----------------------------------------------------------

    def component_means(self) -> dict[str, float]:
        """Mean value per component across all episodes."""
        if not self._episodes:
            return {name: 0.0 for name in self.component_names}
        means: dict[str, float] = {}
        for name in self.component_names:
            values = [ep.total * 0 + sum(c.value for c in ep.components if c.name == name) for ep in self._episodes]
            means[name] = sum(values) / len(values)
        return means

    def component_dominance(self) -> dict[str, float]:
        """Fraction of total (absolute) reward contributed by each component.

        Returns zero for all components if total reward is zero.
        """
        means = self.component_means()
        total_abs = sum(abs(v) for v in means.values())
        if total_abs == 0.0:
            return {name: 0.0 for name in self.component_names}
        return {name: abs(means[name]) / total_abs for name in self.component_names}

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialisable dictionary representation."""
        return {
            "envtype": self.envtype,
            "component_names": self.component_names,
            "episodes": [
                {
                    "episode": ep.episode,
                    "total": ep.total,
                    "components": [
                        {"name": c.name, "value": c.value, "weight": c.weight}
                        for c in ep.components
                    ],
                }
                for ep in self._episodes
            ],
        }

    def save_json(self, path: str) -> None:
        """Write the full decomposition to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load_json(cls, path: str) -> RewardDecomposer:
        """Reconstruct a RewardDecomposer from a JSON file."""
        data = json.loads(Path(path).read_text())
        decomp = cls(envtype=data["envtype"], component_names=data["component_names"])
        for ep_data in data.get("episodes", []):
            components = [
                RewardComponent(name=c["name"], value=c["value"], weight=c.get("weight", 1.0))
                for c in ep_data["components"]
            ]
            ep = EpisodeReward(episode=ep_data["episode"], components=components, total=ep_data["total"])
            decomp._episodes.append(ep)
        return decomp

    # -- summary ------------------------------------------------------------

    def summary(self) -> dict:
        """High-level summary statistics."""
        means = self.component_means()
        n = len(self.component_names)
        if n == 0:
            return {
                "n_episodes": self.n_episodes(),
                "n_components": 0,
                "total_mean": 0.0,
                "most_important_component": None,
                "least_important_component": None,
            }
        dominance = self.component_dominance()
        most = max(dominance, key=dominance.get) if dominance else None
        least = min(dominance, key=dominance.get) if dominance else None
        total_mean = sum(means.values())
        return {
            "n_episodes": self.n_episodes(),
            "n_components": n,
            "total_mean": total_mean,
            "most_important_component": most,
            "least_important_component": least,
        }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def stacked_bar_data(decomposer: RewardDecomposer) -> dict:
    """Return data shaped for a matplotlib stacked bar chart.

    Returns
    -------
    dict with keys:
        episodes   : list[int]  – x-axis labels
        components : dict[str, list[float]]  – one value-list per component
        totals     : list[float]
    """
    episodes = [ep.episode for ep in decomposer._episodes]
    components: dict[str, list[float]] = {name: [] for name in decomposer.component_names}
    for ep in decomposer._episodes:
        for name in decomposer.component_names:
            val = next((c.value for c in ep.components if c.name == name), 0.0)
            components[name].append(val)
    totals = [ep.total for ep in decomposer._episodes]
    return {"episodes": episodes, "components": components, "totals": totals}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_decompositions(decomp_a: RewardDecomposer, decomp_b: RewardDecomposer) -> dict:
    """Per-component mean difference (A - B).

    Returns a dict mapping component name -> mean_a - mean_b.
    """
    means_a = decomp_a.component_means()
    means_b = decomp_b.component_means()
    all_names = list(dict.fromkeys(decomp_a.component_names + decomp_b.component_names))
    return {name: means_a.get(name, 0.0) - means_b.get(name, 0.0) for name in all_names}


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def format_decomp_report(decomposer: RewardDecomposer) -> str:
    """Human-readable text summary of the decomposition."""
    lines: list[str] = []
    lines.append(f"Reward Decomposition – {decomposer.envtype}")
    lines.append(f"Episodes: {decomposer.n_episodes()}")
    lines.append(f"Components: {', '.join(decomposer.component_names)}")
    lines.append("")
    lines.append("Component means:")
    means = decomposer.component_means()
    for name in decomposer.component_names:
        lines.append(f"  {name:.<24s} {means[name]:>12.4f}")
    lines.append("")
    lines.append("Dominance (fraction of |total|):")
    dominance = decomposer.component_dominance()
    for name in decomposer.component_names:
        lines.append(f"  {name:.<24s} {dominance[name]:>12.4f}")
    lines.append("")
    s = decomposer.summary()
    lines.append(f"Total mean: {s['total_mean']:.4f}")
    if s["most_important_component"] is not None:
        lines.append(f"Most important component:  {s['most_important_component']}")
    if s["least_important_component"] is not None:
        lines.append(f"Least important component: {s['least_important_component']}")
    return "\n".join(lines)
