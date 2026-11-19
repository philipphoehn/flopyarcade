"""C14: Prognostic auxiliary particle with longer MODPATH tracking.

Provides a second, longer-horizon particle tracking simulation that gives
the agent predictive information about where the primary particle is
expected to travel over a future window.

The prognostic particle does NOT affect the reward or game state; it is
purely an auxiliary observation channel.

Usage::

    from flopyarcade.prognostic_particle import PrognosticParticle

    prognostic = PrognosticParticle(
        horizon_steps=10,
        modpath_runner=mp_runner,  # existing ModflowRunner instance
    )

    # Call after each step to get the extended trajectory
    trajectory = prognostic.track(current_coords, env_state)
    # trajectory['positions']: list of (x, y, z) tuples
    # trajectory['endpoint']: (x, y, z)
    # trajectory['will_capture']: bool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PrognosticResult:
    """Result of a prognostic particle tracking run.

    Attributes:
        positions: Sequence of (x, y, z) positions along the trajectory.
        endpoint: Final (x, y, z) position after the horizon.
        will_capture: Whether the particle reaches the capture zone.
        steps: Number of timesteps simulated.
        horizon_steps: Requested horizon length.
        confidence: Heuristic confidence in [0, 1] based on trajectory smoothness.
    """
    positions: List[Tuple[float, float, float]] = field(default_factory=list)
    endpoint: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    will_capture: bool = False
    steps: int = 0
    horizon_steps: int = 0
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Core prognostic tracking logic
# ---------------------------------------------------------------------------

class PrognosticParticle:
    """Longer-horizon particle tracking for predictive observations.

    This class wraps the existing MODPATH infrastructure to simulate
    N additional timesteps beyond the current game state, providing the
    agent with a forecast of the particle's future trajectory.

    Args:
        horizon_steps: Number of additional timesteps to simulate.
        modpath_runner: The existing ModflowRunner (or compatible object)
            that has particle tracking methods. May be None for
            standalone computation mode.
        capture_radius: Radius of the capture zone in model units.
        well_coords: (x, y, z) of the capture well. If None, read from
            modpath_runner.
    """

    def __init__(
        self,
        horizon_steps: int = 10,
        modpath_runner: Any = None,
        capture_radius: float = 5.0,
        well_coords: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.horizon_steps = max(1, horizon_steps)
        self.modpath_runner = modpath_runner
        self.capture_radius = capture_radius
        self.well_coords = well_coords

    @property
    def effective_well(self) -> Optional[Tuple[float, float, float]]:
        """Return the well coordinates, falling back to the runner."""
        if self.well_coords is not None:
            return self.well_coords
        if self.modpath_runner is not None and hasattr(self.modpath_runner, 'wellCoords'):
            return tuple(self.modpath_runner.wellCoords)
        return None

    def track(
        self,
        current_coords: Tuple[float, float, float],
        env_state: Optional[Dict[str, Any]] = None,
    ) -> PrognosticResult:
        """Run the prognostic tracking simulation.

        Args:
            current_coords: Current (x, y, z) particle position.
            env_state: Optional dict with 'velocity_field' (np.ndarray
                shape (n_steps, 3)) for standalone mode, or
                'runner' key to override the modpath_runner.

        Returns:
            PrognosticResult with trajectory, endpoint, and capture prediction.
        """
        if env_state and 'velocity_field' in env_state:
            return self._track_with_velocity(current_coords, env_state['velocity_field'])

        if self.modpath_runner is not None:
            return self._track_with_runner(current_coords)

        # Fallback: simple ballistic extrapolation
        return self._track_ballistic(current_coords)

    def _track_with_velocity(
        self,
        start: Tuple[float, float, float],
        velocity_field: np.ndarray,
    ) -> PrognosticResult:
        """Track using a pre-computed velocity field.

        Args:
            start: Starting (x, y, z).
            velocity_field: Array of shape (n_steps, 3) with velocities.

        Returns:
            PrognosticResult.
        """
        pos = np.array(start, dtype=float)
        positions = [tuple(pos)]

        n_steps = min(self.horizon_steps, velocity_field.shape[0])
        for t in range(n_steps):
            pos = pos + velocity_field[t]
            positions.append(tuple(pos))

        endpoint = positions[-1]
        will_capture = self._check_capture(endpoint)
        confidence = self._compute_confidence(positions)

        return PrognosticResult(
            positions=positions,
            endpoint=endpoint,
            will_capture=will_capture,
            steps=n_steps,
            horizon_steps=self.horizon_steps,
            confidence=confidence,
        )

    def _track_with_runner(self, start: Tuple[float, float, float]) -> PrognosticResult:
        """Track using the MODPATH runner (full simulation)."""
        runner = self.modpath_runner
        positions = [start]
        current = list(start)

        for step in range(self.horizon_steps):
            runner.particleCoords = list(current)
            runner.runMODPATH()
            runner.evaluateParticleTracking()
            current = list(runner.particleCoords)
            positions.append(tuple(current))

            # Early exit if particle exits domain or reaches well
            if self._check_capture(tuple(current)):
                break

        endpoint = positions[-1]
        will_capture = self._check_capture(endpoint)
        confidence = self._compute_confidence(positions)

        return PrognosticResult(
            positions=positions,
            endpoint=endpoint,
            will_capture=will_capture,
            steps=len(positions) - 1,
            horizon_steps=self.horizon_steps,
            confidence=confidence,
        )

    def _track_ballistic(
        self, start: Tuple[float, float, float]
    ) -> PrognosticResult:
        """Simple ballistic extrapolation (no runner available).

        Assumes constant velocity estimated from the starting direction
        toward the well (if provided) or zero velocity.
        """
        pos = np.array(start, dtype=float)
        positions = [tuple(pos)]

        well = self.effective_well
        if well is not None:
            direction = np.array(well) - pos
            dist = np.linalg.norm(direction)
            if dist > 1e-9:
                velocity = direction / dist * 1.0  # 1 unit per step
            else:
                velocity = np.zeros(3)
        else:
            velocity = np.zeros(3)

        for _ in range(self.horizon_steps):
            pos = pos + velocity
            positions.append(tuple(pos))

        endpoint = positions[-1]
        will_capture = self._check_capture(endpoint)
        confidence = 0.3  # Low confidence for ballistic mode

        return PrognosticResult(
            positions=positions,
            endpoint=endpoint,
            will_capture=will_capture,
            steps=self.horizon_steps,
            horizon_steps=self.horizon_steps,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_capture(self, position: Tuple[float, float, float]) -> bool:
        """Check if a position is within the capture zone."""
        well = self.effective_well
        if well is None:
            return False
        dist = np.linalg.norm(np.array(position) - np.array(well))
        return bool(dist <= self.capture_radius)

    def _compute_confidence(
        self, positions: List[Tuple[float, float, float]]
    ) -> float:
        """Compute a heuristic confidence score based on trajectory smoothness.

        A straight trajectory gets high confidence; a jagged one gets low.
        """
        if len(positions) < 3:
            return 0.5

        pts = np.array(positions)
        # Compute direction changes
        diffs = np.diff(pts, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        valid = norms > 1e-12
        if np.sum(valid) < 2:
            return 0.5

        unit_diffs = diffs[valid] / norms[valid, None]
        # Cosine similarity between consecutive direction vectors
        cosines = []
        for i in range(len(unit_diffs) - 1):
            c = np.dot(unit_diffs[i], unit_diffs[i + 1])
            cosines.append(c)

        if not cosines:
            return 0.5

        mean_cos = np.mean(cosines)
        # Map from [-1, 1] to [0, 1]: 1.0 = perfectly straight
        confidence = (mean_cos + 1.0) / 2.0
        return float(np.clip(confidence, 0.0, 1.0))

    def get_observation_vector(
        self, result: PrognosticResult
    ) -> List[float]:
        """Convert a PrognosticResult to a flat observation vector.

        Format: [endpoint_x, endpoint_y, endpoint_z,
                 will_capture, confidence, n_steps]

        Returns:
            List of 6 floats suitable for concatenation with state.
        """
        return [
            result.endpoint[0],
            result.endpoint[1],
            result.endpoint[2],
            float(result.will_capture),
            result.confidence,
            float(result.steps),
        ]
