"""C19: Save test game dataset for evaluation.

Creates a structured, versioned dataset of game episodes that can be used
to evaluate and compare agents reproducibly.

Usage::

    from flopyarcade.test_dataset import (
        EpisodeRecord, DatasetConfig, TestDataset,
        generate_synthetic_dataset, evaluate_agent, format_dataset_report
    )

    ds = generate_synthetic_dataset()
    print(format_dataset_report(ds))
    ds.save_json("dataset.json")
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VALID_TERMINAL_STATES = ("capture", "fail", "timeout")


@dataclass
class EpisodeRecord:
    """A single recorded game episode."""

    episode_id: int
    envtype: str
    seed: int
    n_steps: int
    total_reward: float
    terminal_state: str  # capture | fail | timeout
    action_sequence: Optional[List[int]] = None
    reward_sequence: Optional[List[float]] = None
    obs_initial: Optional[List[float]] = None
    obs_final: Optional[List[float]] = None
    git_sha: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        if self.terminal_state not in VALID_TERMINAL_STATES:
            raise ValueError(
                f"terminal_state must be one of {VALID_TERMINAL_STATES}, "
                f"got {self.terminal_state!r}"
            )
        if self.n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {self.n_steps}")


@dataclass
class DatasetConfig:
    """Configuration for a test dataset."""

    name: str = "flopyarcade-eval"
    version: str = "1.0.0"
    description: str = ""
    n_episodes_target: int = 50
    envtypes: List[str] = field(
        default_factory=lambda: ["1s-d", "2s-d", "3s-d", "1s-c", "2s-c", "3s-c"]
    )
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestDataset:
    """Structured, versioned collection of game episodes for evaluation."""

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.config = config or DatasetConfig()
        self._episodes: List[EpisodeRecord] = []

    # -- core accessors ----------------------------------------------------

    def add_episode(self, episode: EpisodeRecord) -> None:
        self._episodes.append(episode)

    def episodes(self) -> List[EpisodeRecord]:
        return list(self._episodes)

    def n_episodes(self) -> int:
        return len(self._episodes)

    def n_envtypes(self) -> int:
        return len({ep.envtype for ep in self._episodes})

    # -- filtering ---------------------------------------------------------

    def episodes_by_envtype(self, envtype: str) -> List[EpisodeRecord]:
        return [ep for ep in self._episodes if ep.envtype == envtype]

    def episodes_by_seed(self, seed: int) -> List[EpisodeRecord]:
        return [ep for ep in self._episodes if ep.seed == seed]

    def episodes_by_outcome(self, outcome: str) -> List[EpisodeRecord]:
        return [ep for ep in self._episodes if ep.terminal_state == outcome]

    # -- statistics --------------------------------------------------------

    def capture_rate(self) -> float:
        if not self._episodes:
            return 0.0
        captures = sum(1 for ep in self._episodes if ep.terminal_state == "capture")
        return captures / len(self._episodes)

    def mean_reward(self, envtype: Optional[str] = None) -> float:
        eps = self._episodes
        if envtype is not None:
            eps = [ep for ep in eps if ep.envtype == envtype]
        if not eps:
            return 0.0
        return float(np.mean([ep.total_reward for ep in eps]))

    def reward_stats(self, envtype: Optional[str] = None) -> Dict[str, float]:
        eps = self._episodes
        if envtype is not None:
            eps = [ep for ep in eps if ep.envtype == envtype]
        if not eps:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        rewards = np.array([ep.total_reward for ep in eps])
        return {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "n": len(eps),
        }

    def duration_stats(self) -> Dict[str, float]:
        if not self._episodes:
            return {"mean_steps": 0.0, "std_steps": 0.0, "min_steps": 0, "max_steps": 0}
        steps = np.array([ep.n_steps for ep in self._episodes])
        return {
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "min_steps": int(np.min(steps)),
            "max_steps": int(np.max(steps)),
        }

    # -- splitting ---------------------------------------------------------

    def split(self, train_frac: float = 0.8) -> Tuple[List[EpisodeRecord], List[EpisodeRecord]]:
        if not 0.0 < train_frac < 1.0:
            raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(self._episodes))
        split_idx = int(len(self._episodes) * train_frac)
        train_idx = set(indices[:split_idx].tolist())
        train = [ep for i, ep in enumerate(self._episodes) if i in train_idx]
        test = [ep for i, ep in enumerate(self._episodes) if i not in train_idx]
        return train, test

    def stratified_split(self, train_frac: float = 0.8) -> Tuple[List[EpisodeRecord], List[EpisodeRecord]]:
        if not 0.0 < train_frac < 1.0:
            raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
        envtypes = sorted({ep.envtype for ep in self._episodes})
        train: List[EpisodeRecord] = []
        test: List[EpisodeRecord] = []
        for et in envtypes:
            group = [ep for ep in self._episodes if ep.envtype == et]
            split_idx = max(1, int(len(group) * train_frac)) if len(group) > 1 else len(group)
            if len(group) <= 1:
                train.extend(group)
            else:
                train.extend(group[:split_idx])
                test.extend(group[split_idx:])
        return train, test

    # -- serialization -----------------------------------------------------

    def checksum(self) -> str:
        canonical = self._canonical_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _canonical_json(self) -> str:
        data = {
            "config": asdict(self.config),
            "episodes": [asdict(ep) for ep in self._episodes],
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def save_json(self, path: str) -> str:
        data = {
            "config": asdict(self.config),
            "episodes": [asdict(ep) for ep in self._episodes],
            "stats": {
                "n_episodes": self.n_episodes(),
                "n_envtypes": self.n_envtypes(),
                "capture_rate": self.capture_rate(),
                "mean_reward": self.mean_reward(),
            },
            "checksum": self.checksum(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path

    @classmethod
    def load_json(cls, path: str) -> "TestDataset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = DatasetConfig(**data["config"])
        ds = cls(config)
        for ep_data in data["episodes"]:
            ep = EpisodeRecord(**ep_data)
            ds._episodes.append(ep)
        return ds

    # -- validation --------------------------------------------------------

    def validate(self) -> List[str]:
        issues: List[str] = []
        if self.n_episodes() == 0:
            issues.append("Dataset is empty")
        if self.n_episodes() < self.config.n_episodes_target:
            issues.append(
                f"Episode count {self.n_episodes()} below target {self.config.n_episodes_target}"
            )
        seen_ids: set = set()
        for i, ep in enumerate(self._episodes):
            if ep.terminal_state not in VALID_TERMINAL_STATES:
                issues.append(f"Episode {i}: invalid terminal_state {ep.terminal_state!r}")
            if ep.episode_id in seen_ids:
                issues.append(f"Episode {i}: duplicate episode_id {ep.episode_id}")
            seen_ids.add(ep.episode_id)
            if ep.n_steps < 0:
                issues.append(f"Episode {i}: negative n_steps {ep.n_steps}")
        return issues

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "version": self.config.version,
            "n_episodes": self.n_episodes(),
            "n_envtypes": self.n_envtypes(),
            "capture_rate": round(self.capture_rate(), 4),
            "mean_reward": round(self.mean_reward(), 4),
            "checksum": self.checksum()[:16] + "...",
        }


def generate_synthetic_dataset(
    config: Optional[DatasetConfig] = None, seed: int = 42
) -> TestDataset:
    """Generate a realistic synthetic dataset with varied rewards, outcomes, and envtypes."""
    cfg = config or DatasetConfig()
    rng = np.random.RandomState(seed)
    ds = TestDataset(cfg)

    outcomes = ["capture", "fail", "timeout"]
    outcome_weights = [0.4, 0.35, 0.25]

    for i in range(cfg.n_episodes_target):
        envtype = cfg.envtypes[i % len(cfg.envtypes)]
        s = cfg.seeds[i % len(cfg.seeds)]
        outcome = rng.choice(outcomes, p=outcome_weights)

        # Realistic reward ranges per envtype
        base_reward = {"1s-d": 80, "2s-d": 120, "3s-d": 160,
                       "1s-c": 200, "2s-c": 280, "3s-c": 350}
        base = base_reward.get(envtype, 100)

        if outcome == "capture":
            reward = float(rng.uniform(base * 0.8, base * 1.5))
            n_steps = int(rng.uniform(50, 200))
        elif outcome == "fail":
            reward = float(rng.uniform(-base * 0.5, base * 0.3))
            n_steps = int(rng.uniform(20, 100))
        else:  # timeout
            reward = float(rng.uniform(-base * 0.2, base * 0.6))
            n_steps = int(rng.uniform(100, 500))

        ep = EpisodeRecord(
            episode_id=i,
            envtype=envtype,
            seed=s,
            n_steps=n_steps,
            total_reward=round(reward, 4),
            terminal_state=outcome,
        )
        ds.add_episode(ep)

    return ds


def evaluate_agent(
    dataset: TestDataset, scores_dict: Dict[str, float]
) -> Dict[str, Any]:
    """Compare per-envtype agent scores against dataset baseline.

    Args:
        dataset: The reference dataset.
        scores_dict: Mapping of envtype -> agent score.

    Returns:
        Dict with per-envtype comparison and overall stats.
    """
    results: Dict[str, Any] = {"envtypes": {}, "overall": {}}

    envtypes = sorted({ep.envtype for ep in dataset.episodes()})
    for et in envtypes:
        baseline = dataset.mean_reward(et)
        agent_score = scores_dict.get(et, 0.0)
        delta = agent_score - baseline
        results["envtypes"][et] = {
            "baseline": round(baseline, 4),
            "agent": round(agent_score, 4),
            "delta": round(delta, 4),
            "improved": delta > 0,
        }

    all_scores = [scores_dict.get(et, 0.0) for et in envtypes]
    all_baselines = [dataset.mean_reward(et) for et in envtypes]
    results["overall"] = {
        "mean_agent": round(float(np.mean(all_scores)), 4) if all_scores else 0.0,
        "mean_baseline": round(float(np.mean(all_baselines)), 4) if all_baselines else 0.0,
        "n_envtypes": len(envtypes),
        "n_improved": sum(
            1 for et in envtypes if scores_dict.get(et, 0.0) > dataset.mean_reward(et)
        ),
    }
    return results


def format_dataset_report(ds: TestDataset) -> str:
    """Generate a human-readable text summary table."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"  Dataset: {ds.config.name} v{ds.config.version}")
    lines.append(f"  Description: {ds.config.description or '(none)'}")
    lines.append("=" * 60)
    lines.append(f"  Episodes:    {ds.n_episodes()}")
    lines.append(f"  Envtypes:    {ds.n_envtypes()}")
    lines.append(f"  Capture:     {ds.capture_rate():.2%}")
    lines.append(f"  Mean Reward: {ds.mean_reward():.2f}")
    lines.append("")

    # Per-envtype breakdown
    lines.append("  Per-envtype breakdown:")
    lines.append(f"  {'Envtype':<10} {'N':>4} {'Mean':>10} {'Std':>8} {'Min':>10} {'Max':>10} {'CapRate':>8}")
    lines.append(f"  {'-'*10} {'-'*4} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    envtypes = sorted({ep.envtype for ep in ds.episodes()})
    for et in envtypes:
        eps = ds.episodes_by_envtype(et)
        if not eps:
            continue
        rewards = [ep.total_reward for ep in eps]
        caps = sum(1 for ep in eps if ep.terminal_state == "capture")
        lines.append(
            f"  {et:<10} {len(eps):>4} {np.mean(rewards):>10.2f} "
            f"{np.std(rewards):>8.2f} {np.min(rewards):>10.2f} "
            f"{np.max(rewards):>10.2f} {caps/len(eps):>8.2%}"
        )

    # Duration stats
    d = ds.duration_stats()
    lines.append("")
    lines.append(f"  Duration: mean={d['mean_steps']:.1f} steps, "
                 f"std={d['std_steps']:.1f}, range=[{d['min_steps']}, {d['max_steps']}]")
    lines.append("")
    lines.append(f"  Checksum:  {ds.checksum()[:32]}...")
    lines.append("=" * 60)
    return "\n".join(lines)


def dataset_diff(ds_a: TestDataset, ds_b: TestDataset) -> Dict[str, Any]:
    """Structural comparison between two datasets."""
    diff: Dict[str, Any] = {
        "name_diff": ds_a.config.name != ds_b.config.name,
        "version_diff": ds_a.config.version != ds_b.config.version,
        "n_episodes_a": ds_a.n_episodes(),
        "n_episodes_b": ds_b.n_episodes(),
        "n_episodes_diff": ds_a.n_episodes() - ds_b.n_episodes(),
        "envtypes_a": sorted({ep.envtype for ep in ds_a.episodes()}),
        "envtypes_b": sorted({ep.envtype for ep in ds_b.episodes()}),
        "envtypes_only_a": sorted(
            {ep.envtype for ep in ds_a.episodes()} - {ep.envtype for ep in ds_b.episodes()}
        ),
        "envtypes_only_b": sorted(
            {ep.envtype for ep in ds_b.episodes()} - {ep.envtype for ep in ds_a.episodes()}
        ),
        "capture_rate_a": round(ds_a.capture_rate(), 4),
        "capture_rate_b": round(ds_b.capture_rate(), 4),
        "mean_reward_a": round(ds_a.mean_reward(), 4),
        "mean_reward_b": round(ds_b.mean_reward(), 4),
        "checksum_a": ds_a.checksum()[:16],
        "checksum_b": ds_b.checksum()[:16],
        "identical": ds_a.checksum() == ds_b.checksum(),
    }
    return diff
