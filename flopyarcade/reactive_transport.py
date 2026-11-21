"""Multi-component reactive transport environment (S1 - 7r-d).

Two-species first-order degradation:  parent -> daughter
kinetics:  daughter += parent * lambda * dt ;  parent *= (1 - lambda * dt),  dt = 1
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


# -- data containers -----------------------------------------------------------

@dataclass
class SpeciesState:
    """Single species descriptor."""
    name: str
    concentration: float
    parent: Optional[str] = None
    decay_constant: float = 0.0  # 1/time


@dataclass
class ReactiveParams:
    """Parameters for the two-species reactive transport env."""
    parent_name: str = "C"        # e.g. Carbon-14
    daughter_name: str = "N"      # e.g. Nitrogen-14
    decay_lambda: float = 0.001   # first-order decay constant 1/timestep
    initial_parent: float = 1.0
    initial_daughter: float = 0.0

    @property
    def half_life_steps(self) -> float:
        """Number of timesteps for the parent to halve."""
        return math.log(2) / self.decay_lambda


# -- environment ---------------------------------------------------------------

class ReactiveTransportEnv:
    """Grid-based two-species reactive transport with first-order kinetics."""

    def __init__(
        self,
        params: ReactiveParams,
        grid_shape: tuple[int, int] = (5, 5),
    ) -> None:
        self.params = params
        self.grid_shape = grid_shape
        self._parent: np.ndarray = np.full(grid_shape, params.initial_parent, dtype=np.float64)
        self._daughter: np.ndarray = np.full(grid_shape, params.initial_daughter, dtype=np.float64)
        self._initial_total: float = float(self._parent.sum() + self._daughter.sum())
        self._t: int = 0

    # -- field accessors -------------------------------------------------------

    def parent_field(self) -> list[list[float]]:
        return self._parent.tolist()

    def daughter_field(self) -> list[list[float]]:
        return self._daughter.tolist()

    # -- simulation step -------------------------------------------------------

    def step(self) -> dict:
        """Advance one timestep (dt = 1) of first-order kinetics."""
        lam = self.params.decay_lambda
        self._daughter += self._parent * lam
        self._parent *= (1.0 - lam)
        self._t += 1

        parent_total = float(self._parent.sum())
        daughter_total = float(self._daughter.sum())
        total = parent_total + daughter_total
        return {
            "t": self._t,
            "parent_total": parent_total,
            "daughter_total": daughter_total,
            "mass_conserved": abs(total - self._initial_total) < 1e-6,
        }

    # -- metrics ---------------------------------------------------------------

    def n_steps(self) -> int:
        return self._t

    def total_concentration(self) -> float:
        return float(self._parent.sum() + self._daughter.sum())

    def mass_balance_error(self) -> float:
        if self._initial_total == 0.0:
            return 0.0
        return abs(self.total_concentration() - self._initial_total) / self._initial_total

    def daughter_fraction(self) -> float:
        total = self.total_concentration()
        if total == 0.0:
            return 0.0
        return float(self._daughter.sum()) / total

    def activity_ratio(self) -> float:
        """Parent activity / daughter activity (with epsilon guard)."""
        lam = self.params.decay_lambda
        parent_act = lam * float(self._parent.sum())
        daughter_act = lam * float(self._daughter.sum()) + 1e-12
        return parent_act / daughter_act

    def half_life_reached(self) -> bool:
        return self.daughter_fraction() >= 0.5

    def equilibrium(self, tol: float = 0.01) -> bool:
        total = self.total_concentration()
        if total == 0.0:
            return True
        diff = abs(float(self._parent.sum()) - float(self._daughter.sum()))
        return diff < tol * total

    def capture_efficiency(self, capture_zone: list[tuple[int, int]]) -> float:
        """Fraction of total daughter concentration in the given cells."""
        total_daughter = float(self._daughter.sum())
        if total_daughter == 0.0:
            return 0.0
        zone_sum = 0.0
        for (r, c) in capture_zone:
            zone_sum += float(self._daughter[r, c])
        return zone_sum / total_daughter

    # -- lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        self._parent = np.full(self.grid_shape, self.params.initial_parent, dtype=np.float64)
        self._daughter = np.full(self.grid_shape, self.params.initial_daughter, dtype=np.float64)
        self._initial_total = float(self._parent.sum() + self._daughter.sum())
        self._t = 0

    # -- serialisation ---------------------------------------------------------

    def export_field(self, path: str) -> None:
        """Save parent+daughter grids as .npz file."""
        np.savez(path, parent=self._parent, daughter=self._daughter)

    def save_state(self, path: str) -> None:
        """Save full environment state as JSON."""
        state = {
            "params": asdict(self.params),
            "grid_shape": list(self.grid_shape),
            "t": self._t,
            "initial_total": self._initial_total,
            "parent": self._parent.tolist(),
            "daughter": self._daughter.tolist(),
        }
        with open(path, "w") as fh:
            json.dump(state, fh, indent=2)

    @classmethod
    def load_state(cls, path: str) -> ReactiveTransportEnv:
        """Load environment state from JSON file."""
        with open(path, "r") as fh:
            state = json.load(fh)
        params = ReactiveParams(**state["params"])
        env = cls(params, grid_shape=tuple(state["grid_shape"]))
        env._t = state["t"]
        env._initial_total = state["initial_total"]
        env._parent = np.array(state["parent"], dtype=np.float64)
        env._daughter = np.array(state["daughter"], dtype=np.float64)
        return env


# -- module-level helpers ------------------------------------------------------

def multi_step_simulation(
    params: ReactiveParams,
    n_steps: int,
    grid_shape: tuple[int, int] = (5, 5),
) -> list[dict]:
    """Run n_steps timesteps and return per-step records."""
    env = ReactiveTransportEnv(params, grid_shape)
    records: list[dict] = []
    for _ in range(n_steps):
        records.append(env.step())
    return records


def kinetic_curve(
    params: ReactiveParams,
    n_points: int = 100,
) -> list[tuple[float, float, float]]:
    """Return (t, parent_frac, daughter_frac) samples for plotting."""
    env = ReactiveTransportEnv(params, grid_shape=(1, 1))
    curve: list[tuple[float, float, float]] = []
    initial = env.total_concentration()
    for i in range(n_points + 1):
        p_frac = float(env._parent.sum()) / initial if initial else 0.0
        d_frac = float(env._daughter.sum()) / initial if initial else 0.0
        curve.append((float(i), p_frac, d_frac))
        if i < n_points:
            env.step()
    return curve


def compare_reactive(
    env_a: ReactiveTransportEnv,
    env_b: ReactiveTransportEnv,
) -> dict:
    """Compare two reactive transport environments."""
    a_df = env_a.daughter_fraction()
    b_df = env_b.daughter_fraction()
    return {
        "a_daughter_frac": a_df,
        "b_daughter_frac": b_df,
        "a_mass_err": env_a.mass_balance_error(),
        "b_mass_err": env_b.mass_balance_error(),
        "faster_decay": a_df > b_df,
    }


def format_reactive_report(env: ReactiveTransportEnv) -> str:
    """Human-readable text summary of the current state."""
    parent_total = float(env._parent.sum())
    daughter_total = float(env._daughter.sum())
    total = env.total_concentration()
    mbe = env.mass_balance_error()
    hl = env.half_life_reached()
    lines = [
        "Reactive Transport Report (7r-d)",
        "=" * 40,
        f"  Time step        : {env.n_steps()}",
        f"  Grid shape       : {env.grid_shape}",
        f"  Parent total     : {parent_total:.8f}",
        f"  Daughter total   : {daughter_total:.8f}",
        f"  Total conc.      : {total:.8f}",
        f"  Mass balance err : {mbe:.2e}",
        f"  Daughter fraction: {env.daughter_fraction():.6f}",
        f"  Half-life reached: {hl}",
        f"  Activity ratio   : {env.activity_ratio():.6f}",
        f"  Equilibrium      : {env.equilibrium()}",
    ]
    return "\n".join(lines)
