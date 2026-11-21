"""Surrogate-assisted Bayesian optimisation for hyperparameter search.

Implements ROADMAP R1: GP surrogate on cheap runs selects top candidates
for expensive evaluation using Expected Improvement (EI) acquisition.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional, Union

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from scipy.stats import norm
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


@dataclass
class BayesOptResult:
    """Single observation result from Bayesian optimisation."""
    params: dict
    objective: float
    surrogate_pred: float
    surrogate_std: float
    acquisition_value: float
    iteration: int


def expected_improvement(
    mu: float,
    sigma: float,
    best_value: float,
    xi: float = 0.01,
) -> float:
    """Compute Expected Improvement (EI) for minimisation (lower is better).

    EI = (best - mu - xi) * Phi(z) + sigma * phi(z)
    where z = (best - mu - xi) / sigma

    Returns a non-negative value; ~0 when mu is well above best_value.
    """
    if _HAS_SKLEARN:
        if sigma < 1e-12:
            return max(0.0, best_value - mu - xi)
        z = (best_value - mu - xi) / sigma
        ei = (best_value - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return max(0.0, ei)
    else:
        # Fallback without scipy
        if sigma < 1e-12:
            return max(0.0, best_value - mu - xi)
        z = (best_value - mu - xi) / sigma
        # Approximate normal CDF and PDF
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        ei = (best_value - mu - xi) * cdf + sigma * pdf
        return max(0.0, ei)


def _is_continuous(spec) -> bool:
    """Check if a search space spec represents a continuous parameter."""
    return (isinstance(spec, (tuple, list)) and len(spec) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in spec))


class SurrogateBayesOptimizer:
    """Surrogate-assisted Bayesian optimiser with GP + EI acquisition.

    Parameters
    ----------
    search_space : dict
        Mapping of param_name -> (low, high) for continuous parameters,
        or param_name -> list_of_values for categorical parameters.
    n_cheap_runs : int
        Number of cheap exploratory runs in phase 1 (default 20).
    n_expensive_runs : int
        Number of expensive runs in phase 2 selected by EI (default 5).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        search_space: dict,
        n_cheap_runs: int = 20,
        n_expensive_runs: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self.search_space = search_space
        self.n_cheap_runs = n_cheap_runs
        self.n_expensive_runs = n_expensive_runs
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._history: list[BayesOptResult] = []
        self._iteration: int = 0
        self._param_names: list[str] = list(search_space.keys())

    # ------------------------------------------------------------------
    # Encoding / decoding helpers
    # ------------------------------------------------------------------
    def _encode(self, params: dict) -> np.ndarray:
        """Encode a params dict into a numeric feature vector in [0,1]."""
        features = []
        for name in self._param_names:
            spec = self.search_space[name]
            val = params[name]
            if _is_continuous(spec):
                low, high = spec
                features.append((val - low) / (high - low))
            else:
                values = list(spec)
                idx = values.index(val) if val in values else 0
                features.append(idx / max(len(values) - 1, 1))
        return np.array(features)

    def _decode(self, features: np.ndarray) -> dict:
        """Decode a numeric feature vector back into a params dict."""
        params = {}
        for i, name in enumerate(self._param_names):
            spec = self.search_space[name]
            if _is_continuous(spec):
                low, high = spec
                params[name] = low + features[i] * (high - low)
            else:
                values = list(spec)
                idx = int(round(features[i] * max(len(values) - 1, 1)))
                idx = max(0, min(idx, len(values) - 1))
                params[name] = values[idx]
        return params

    def _random_params(self) -> dict:
        """Generate random parameters within the search space."""
        params = {}
        for name, spec in self.search_space.items():
            if _is_continuous(spec):
                params[name] = self._rng.uniform(spec[0], spec[1])
            else:
                params[name] = self._rng.choice(list(spec))
        return params

    def _fit_gp(self):
        """Fit GP on current history. Returns (gp, y) or (None, None)."""
        if not _HAS_SKLEARN or len(self._history) < 2:
            return None, None
        X = np.array([self._encode(r.params) for r in self._history])
        y = np.array([r.objective for r in self._history])
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-10, normalize_y=True, random_state=self.seed
        )
        gp.fit(X, y)
        return gp, y

    # ------------------------------------------------------------------
    # Core optimisation API
    # ------------------------------------------------------------------
    def suggest_params(self) -> dict:
        """Suggest next parameters using GP surrogate + EI acquisition.

        Falls back to random search if sklearn is unavailable or there are
        fewer than 2 observations.
        """
        gp, y = self._fit_gp()
        if gp is None:
            return self._random_params()

        best_ei = -np.inf
        best_params = self._random_params()
        n_candidates = 200

        for _ in range(n_candidates):
            cand = self._random_params()
            x = self._encode(cand).reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = max(float(sigma[0]), 1e-12)
            ei = expected_improvement(float(mu[0]), sigma, float(y.min()))
            if ei > best_ei:
                best_ei = ei
                best_params = cand

        return best_params

    def observe(self, params: dict, objective: float) -> BayesOptResult:
        """Record an observation and return the result."""
        self._iteration += 1
        surr_pred = 0.0
        surr_std = 0.0
        acq_val = 0.0

        if _HAS_SKLEARN and len(self._history) >= 1:
            try:
                X = np.array([self._encode(r.params) for r in self._history])
                y = np.array([r.objective for r in self._history])
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
                gp = GaussianProcessRegressor(
                    kernel=kernel, alpha=1e-10, normalize_y=True, random_state=self.seed
                )
                gp.fit(X, y)
                x_new = self._encode(params).reshape(1, -1)
                mu, sigma = gp.predict(x_new, return_std=True)
                surr_pred = float(mu[0])
                surr_std = max(float(sigma[0]), 1e-12)
                acq_val = expected_improvement(surr_pred, surr_std, float(y.min()))
            except Exception:
                pass

        result = BayesOptResult(
            params=params,
            objective=objective,
            surrogate_pred=surr_pred,
            surrogate_std=surr_std,
            acquisition_value=acq_val,
            iteration=self._iteration,
        )
        self._history.append(result)
        return result

    def run_cheap_phase(self, objective_fn: Callable[[dict], float]) -> list[BayesOptResult]:
        """Evaluate objective_fn on n_cheap_runs randomly suggested params.

        This is the exploration phase: random sampling to build initial
        data for the GP surrogate.
        """
        results = []
        for _ in range(self.n_cheap_runs):
            params = self._random_params()
            obj = objective_fn(params)
            result = self.observe(params, obj)
            results.append(result)
        return results

    def run_expensive_phase(self, objective_fn: Callable[[dict], float]) -> list[BayesOptResult]:
        """Select n_expensive_runs most promising params via EI, then evaluate.

        Uses the GP fitted on all prior observations to identify
        candidates with highest Expected Improvement.
        """
        gp, y = self._fit_gp()
        if gp is None:
            candidates = [self._random_params() for _ in range(self.n_expensive_runs)]
        else:
            scored = []
            for _ in range(500):
                cand = self._random_params()
                x = self._encode(cand).reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                sigma = max(float(sigma[0]), 1e-12)
                ei = expected_improvement(float(mu[0]), sigma, float(y.min()))
                scored.append((ei, cand))
            scored.sort(key=lambda t: t[0], reverse=True)
            candidates = [c for _, c in scored[:self.n_expensive_runs]]

        results = []
        for params in candidates:
            obj = objective_fn(params)
            result = self.observe(params, obj)
            results.append(result)
        return results

    def get_best(self) -> BayesOptResult:
        """Return the best (lowest objective) observed result."""
        if not self._history:
            raise ValueError("No observations recorded yet.")
        return min(self._history, key=lambda r: r.objective)

    def get_history(self) -> list[BayesOptResult]:
        """Return full observation history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save_state(self, path: str) -> None:
        """Save optimiser state to a JSON file."""
        state = {
            "search_space": {
                k: (list(v) if _is_continuous(v) else list(v))
                for k, v in self.search_space.items()
            },
            "n_cheap_runs": self.n_cheap_runs,
            "n_expensive_runs": self.n_expensive_runs,
            "seed": self.seed,
            "iteration": self._iteration,
            "history": [asdict(r) for r in self._history],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @classmethod
    def load_state(cls, path: str) -> "SurrogateBayesOptimizer":
        """Load optimiser state from a JSON file."""
        with open(path, "r") as f:
            state = json.load(f)

        # Restore search_space: convert [low, high] back to tuple for continuous
        search_space = {}
        for k, v in state["search_space"].items():
            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                search_space[k] = (v[0], v[1])
            else:
                search_space[k] = v

        opt = cls(
            search_space=search_space,
            n_cheap_runs=state["n_cheap_runs"],
            n_expensive_runs=state["n_expensive_runs"],
            seed=state["seed"],
        )
        opt._iteration = state["iteration"]
        opt._history = [BayesOptResult(**r) for r in state["history"]]
        return opt


def format_bayes_opt_report(bo: SurrogateBayesOptimizer) -> str:
    """Generate a human-readable text summary of the optimisation run."""
    lines = []
    lines.append("=" * 60)
    lines.append("  Surrogate-Assisted Bayesian Optimisation Report")
    lines.append("=" * 60)
    lines.append("")

    lines.append("  Search space:")
    for name, spec in bo.search_space.items():
        if _is_continuous(spec):
            lines.append(f"    {name}: continuous [{spec[0]}, {spec[1]}]")
        else:
            lines.append(f"    {name}: categorical {list(spec)}")

    lines.append("")
    lines.append(f"  Cheap runs (phase 1): {bo.n_cheap_runs}")
    lines.append(f"  Expensive runs (phase 2): {bo.n_expensive_runs}")
    lines.append(f"  Total iterations: {bo._iteration}")
    lines.append(f"  Seed: {bo.seed}")
    lines.append("")

    if bo._history:
        best = bo.get_best()
        lines.append("  Best result:")
        lines.append(f"    Objective: {best.objective:.6f}")
        lines.append(f"    Iteration: {best.iteration}")
        lines.append(f"    Surrogate pred: {best.surrogate_pred:.6f}")
        lines.append(f"    Surrogate std: {best.surrogate_std:.6f}")
        lines.append(f"    Acquisition (EI): {best.acquisition_value:.6f}")
        lines.append(f"    Params: {json.dumps(best.params, default=str)}")
        lines.append("")

        lines.append("  Full history:")
        for r in bo.get_history():
            lines.append(f"    [{r.iteration:>3}] obj={r.objective:>12.6f}  EI={r.acquisition_value:.6f}  pred={r.surrogate_pred:.6f}")
    else:
        lines.append("  No observations recorded.")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
