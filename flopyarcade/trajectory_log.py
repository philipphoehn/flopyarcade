"""Per-step trajectory logging with HDF5/JSON persistence (ROADMAP P1)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class StepRecord:
    """A single step in an episode trajectory."""

    step: int
    action: list[float]
    reward: float
    obs_sha: Optional[str] = None
    pos_2d: Optional[tuple[float, float]] = None
    head_mean: Optional[float] = None


class TrajectoryLogger:
    """Collects per-step records and persists them to HDF5 or JSON."""

    def __init__(self, envtype: str, seed: int, agent_type: str = "random"):
        self.envtype = envtype
        self.seed = seed
        self.agent_type = agent_type
        self.records: list[StepRecord] = []

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def log_step(
        self,
        step: int,
        action: list[float],
        reward: float,
        obs_sha: Optional[str] = None,
        pos_2d: Optional[tuple[float, float]] = None,
        head_mean: Optional[float] = None,
    ) -> None:
        """Append a single step record to the trajectory."""
        self.records.append(
            StepRecord(
                step=step,
                action=action,
                reward=reward,
                obs_sha=obs_sha,
                pos_2d=pos_2d,
                head_mean=head_mean,
            )
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_hdf5(self, path: str) -> str:
        """Write trajectory to an HDF5 file.

        If h5py is unavailable, falls back to JSON.
        Returns the actual path written.
        """
        try:
            import h5py
        except ImportError:
            return self.save_json(path)

        meta = {
            "envtype": self.envtype,
            "seed": self.seed,
            "agent_type": self.agent_type,
            "n_steps": len(self.records),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with h5py.File(path, "w") as f:
            # --- meta group ---
            mg = f.create_group("meta")
            mg.attrs["envtype"] = meta["envtype"]
            mg.attrs["seed"] = meta["seed"]
            mg.attrs["agent_type"] = meta["agent_type"]
            mg.attrs["n_steps"] = meta["n_steps"]
            mg.attrs["timestamp"] = meta["timestamp"]

            # --- steps group ---
            sg = f.create_group("steps")
            if self.records:
                steps = []
                for r in self.records:
                    rec = {
                        "step": r.step,
                        "action": r.action,
                        "reward": r.reward,
                        "obs_sha": r.obs_sha,
                        "pos_2d": r.pos_2d,
                        "head_mean": r.head_mean,
                    }
                    steps.append(rec)

                import numpy as np

                n = len(steps)
                # step (int), reward (float), head_mean (float or None)
                sg.create_dataset(
                    "step", data=np.array([s["step"] for s in steps], dtype=np.int64)
                )
                # action: variable length → store as list of JSON strings
                action_arr = np.array(
                    [json.dumps(s["action"]) for s in steps], dtype="S"
                )
                sg.create_dataset("action", data=action_arr)
                sg.create_dataset(
                    "reward", data=np.array([s["reward"] for s in steps], dtype=np.float64)
                )
                # obs_sha: optional string → JSON-encoded (null or string)
                obs_arr = np.array(
                    [json.dumps(s["obs_sha"]) for s in steps], dtype="S"
                )
                sg.create_dataset("obs_sha", data=obs_arr)
                # pos_2d: optional tuple → JSON-encoded
                pos_arr = np.array(
                    [json.dumps(list(s["pos_2d"]) if s["pos_2d"] else None) for s in steps],
                    dtype="S",
                )
                sg.create_dataset("pos_2d", data=pos_arr)
                # head_mean: optional float → NaN sentinel
                head_vals = np.array(
                    [
                        s["head_mean"] if s["head_mean"] is not None else np.nan
                        for s in steps
                    ],
                    dtype=np.float64,
                )
                sg.create_dataset("head_mean", data=head_vals)

        return path

    def save_json(self, path: str) -> str:
        """Write trajectory as a JSON list of StepRecord dicts."""
        data = [asdict(r) for r in self.records]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str) -> list[StepRecord]:
        """Load StepRecords from an HDF5 or JSON file."""
        if path.endswith(".h5") or path.endswith(".hdf5"):
            try:
                return cls._load_hdf5(path)
            except ImportError:
                pass
            except Exception:
                pass

        # Fallback: JSON
        return cls._load_json(path)

    @staticmethod
    def _load_hdf5(path: str) -> list[StepRecord]:
        import h5py
        import numpy as np

        records: list[StepRecord] = []
        with h5py.File(path, "r") as f:
            if "steps" not in f:
                return records
            sg = f["steps"]
            if "step" not in sg:
                return records
            n = sg["step"].shape[0]
            for i in range(n):
                step = int(sg["step"][i])
                action = json.loads(sg["action"][i].decode() if isinstance(sg["action"][i], bytes) else str(sg["action"][i]))
                reward = float(sg["reward"][i])
                obs_raw = sg["obs_sha"][i]
                obs_sha = json.loads(obs_raw.decode() if isinstance(obs_raw, bytes) else str(obs_raw))
                pos_raw = sg["pos_2d"][i]
                pos_list = json.loads(pos_raw.decode() if isinstance(pos_raw, bytes) else str(pos_raw))
                pos_2d = tuple(pos_list) if pos_list else None
                head_val = sg["head_mean"][i]
                head_mean = None if np.isnan(head_val) else float(head_val)

                records.append(
                    StepRecord(
                        step=step,
                        action=action,
                        reward=reward,
                        obs_sha=obs_sha,
                        pos_2d=pos_2d,
                        head_mean=head_mean,
                    )
                )
        return records

    @staticmethod
    def _load_json(path: str) -> list[StepRecord]:
        with open(path) as f:
            data = json.load(f)
        records: list[StepRecord] = []
        for item in data:
            pos_2d = tuple(item["pos_2d"]) if item.get("pos_2d") else None
            records.append(
                StepRecord(
                    step=item["step"],
                    action=item["action"],
                    reward=item["reward"],
                    obs_sha=item.get("obs_sha"),
                    pos_2d=pos_2d,
                    head_mean=item.get("head_mean"),
                )
            )
        return records

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """Compute aggregate statistics over the logged trajectory."""
        if not self.records:
            return {
                "n_steps": 0,
                "mean_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "total_reward": 0.0,
                "n_unique_actions": 0,
            }
        rewards = [r.reward for r in self.records]
        actions = [tuple(r.action) for r in self.records]
        return {
            "n_steps": len(self.records),
            "mean_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "total_reward": sum(rewards),
            "n_unique_actions": len(set(actions)),
        }

    def to_dataframe(self) -> list[dict]:
        """Return trajectory as a list of dicts (no pandas dependency)."""
        out: list[dict] = []
        for r in self.records:
            d: dict = {
                "step": r.step,
                "action": r.action,
                "reward": r.reward,
                "obs_sha": r.obs_sha,
                "pos_2d": list(r.pos_2d) if r.pos_2d else None,
                "head_mean": r.head_mean,
            }
            out.append(d)
        return out


# ---------------------------------------------------------------------- #
# Comparison
# ---------------------------------------------------------------------- #

def compare_trajectories(
    traj_a: list[StepRecord],
    traj_b: list[StepRecord],
) -> dict:
    """Compare two trajectories and return summary statistics."""
    len_a = len(traj_a)
    len_b = len(traj_b)
    mean_a = sum(r.reward for r in traj_a) / len_a if len_a else 0.0
    mean_b = sum(r.reward for r in traj_b) / len_b if len_b else 0.0

    # Action similarity over aligned (min-length) steps
    n = min(len_a, len_b)
    if n > 0:
        matches = sum(
            1 for i in range(n) if list(traj_a[i].action) == list(traj_b[i].action)
        )
        action_similarity = matches / n
    else:
        action_similarity = 0.0

    return {
        "len_a": len_a,
        "len_b": len_b,
        "mean_reward_a": mean_a,
        "mean_reward_b": mean_b,
        "reward_diff": mean_a - mean_b,
        "action_similarity": action_similarity,
    }
