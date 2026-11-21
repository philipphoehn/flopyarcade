"""Optuna-based hyperparameter search for FloPyArcade GA/DQN agents.

Roadmap A7: Replace manual hyperparameter tuning with Bayesian
optimisation via Optuna. This module defines the search space and an
objective function wrapper. The actual training loop is NOT executed
here — a human runs the search with the provided CLI.

Usage (code-only, do not run in CI)::

    # DO NOT RUN: requires extended training
    python -m flopyarcade.optuna_hp_search \
        --envtype 1s-d \
        --n-trials 50 \
        --study-name flopyarcade-1sd \
        --storage sqlite:///hp_study.db

# DO NOT RUN: requires extended training
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "NAGENTS": {"type": "int", "low": 8, "high": 64, "step": 4},
    "NGENERATIONS": {"type": "int", "low": 5, "high": 50, "step": 5},
    "NAGENTSSTEPS": {"type": "int", "low": 10, "high": 200, "step": 10},
    "NNTYPE": {"type": "categorical", "choices": [0, 1, 2, 3]},
    "CROSSVALIDATEEVERY": {"type": "int", "low": 2, "high": 10, "step": 1},
    "NGAMESAVERAGED": {"type": "int", "low": 3, "high": 10, "step": 1},
    "NGAMESCROSSVALIDATED": {"type": "int", "low": 2, "high": 8, "step": 1},
    # DQN-specific
    "EPSILONINITIAL": {"type": "float", "low": 0.5, "high": 1.0},
    "EPSILONMIN": {"type": "float", "low": 0.01, "high": 0.2},
    "EPSILONDECAY": {"type": "float", "low": 0.90, "high": 0.999},
    "REPLAYMEMORYSIZE": {"type": "int", "low": 1000, "high": 100000, "step": 1000},
    # Novelty search
    "NOVELTYSEARCH": {"type": "categorical", "choices": [False, True]},
    "NNOVELTYNEIGHBORS": {"type": "int", "low": 5, "high": 50, "step": 5},
    "NNOVELTYELITES": {"type": "int", "low": 2, "high": 16, "step": 2},
    "ADDNOVELTYEVERY": {"type": "int", "low": 5, "high": 50, "step": 5},
}


def build_search_space(exclude: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
    """Return the search space, optionally excluding keys.

    Args:
        exclude: List of parameter names to remove.

    Returns:
        Dict of parameter name -> spec.
    """
    if not exclude:
        return dict(SEARCH_SPACE)
    excluded = set(exclude)
    return {k: v for k, v in SEARCH_SPACE.items() if k not in excluded}


def suggest_params(trial: Any, space: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Suggest a parameter dict from an Optuna trial.

    Args:
        trial: An Optuna trial object (or mock with suggest_* methods).
        space: Optional custom search space (defaults to SEARCH_SPACE).

    Returns:
        Dict of suggested parameter values.
    """
    if space is None:
        space = SEARCH_SPACE

    params = {}
    for name, spec in space.items():
        ptype = spec["type"]
        if ptype == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif ptype == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown param type '{ptype}' for {name}")
    return params


def objective(trial: Any, envtype: str = "1s-d", space: Optional[dict] = None) -> float:
    """Objective function for Optuna optimisation.

    # DO NOT RUN: requires extended training

    This is a STUB that returns a placeholder score. In production, it
    would instantiate FloPyAgent, run a short training session, and
    return the best reward. The actual training loop is NOT called here.

    Args:
        trial: Optuna trial.
        envtype: Environment type string (e.g. '1s-d').
        space: Optional custom search space.

    Returns:
        Placeholder score (always 0.0 in stub mode).
    """
    params = suggest_params(trial, space)
    logger.info("Trial %d: %s", trial.number, json.dumps(params, default=str))
    # STUB: in real usage this would run training and return best_reward
    return 0.0


def run_search(
    envtype: str = "1s-d",
    n_trials: int = 50,
    study_name: str = "flopyarcade_hp",
    storage: Optional[str] = None,
    sampler: str = "TPESampler",
    seed: Optional[int] = None,
    exclude_params: Optional[list] = None,
) -> Dict[str, Any]:
    """Run the Optuna search (STUB — does not execute training).

    # DO NOT RUN: requires extended training

    Args:
        envtype: Environment type.
        n_trials: Number of trials.
        study_name: Name for the Optuna study.
        storage: Storage backend URL (e.g. 'sqlite:///study.db').
        sampler: Sampler class name.
        seed: Random seed.
        exclude_params: Parameters to exclude from search.

    Returns:
        Dict with 'n_trials', 'best_params', 'best_value'.
    """
    try:
        import optuna
    except ImportError:
        logger.error("optuna is not installed. Install with: pip install optuna")
        return {"n_trials": 0, "best_params": {}, "best_value": None, "error": "optuna not installed"}

    space = build_search_space(exclude_params)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=getattr(optuna, sampler)(seed=seed),
    )

    study.optimize(
        lambda trial: objective(trial, envtype, space),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    return {
        "n_trials": study.trials.__len__(),
        "best_params": study.best_params,
        "best_value": study.best_value,
    }


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments for the HP search CLI.

    Args:
        argv: Optional argument list (for testing).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for FloPyArcade (code-only, do not run training)"
    )
    parser.add_argument("--envtype", default="1s-d", help="Environment type (default: 1s-d)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    parser.add_argument("--study-name", default="flopyarcade_hp", help="Study name")
    parser.add_argument("--storage", default=None, help="Storage URL (e.g. sqlite:///hp.db)")
    parser.add_argument("--sampler", default="TPESampler", help="Optuna sampler (default: TPESampler)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--exclude", nargs="*", default=None, help="Parameters to exclude")
    parser.add_argument("--dry-run", action="store_true", help="Print search space and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """Entry point for CLI.

    # DO NOT RUN: requires extended training
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.dry_run:
        space = build_search_space(args.exclude)
        print(json.dumps(space, indent=2, default=str))
        return 0

    result = run_search(
        envtype=args.envtype,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        sampler=args.sampler,
        seed=args.seed,
        exclude_params=args.exclude,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
