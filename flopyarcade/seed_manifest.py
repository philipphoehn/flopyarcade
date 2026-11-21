"""Seed manifest for reproducible run metadata.

ROADMAP P3: ``{seed, git_sha, config_hash, env_version, score, ci_run_id}``
JSON per run, plus append-only JSONL log support.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "SeedManifest",
    "create_manifest",
    "save_manifest",
    "load_manifest",
    "append_to_log",
    "load_log",
]

PathLike = Union[str, Path]


@dataclass
class SeedManifest:
    """Reproducible run metadata for a single seed run."""

    seed: int
    git_sha: str
    config_hash: str
    env_version: str
    score: float
    ci_run_id: Optional[str] = None
    timestamp: str = ""  # ISO 8601 UTC


def _detect_git_sha() -> str:
    """Return the current git HEAD SHA, or ``'unknown'`` if unavailable."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        if result.returncode == 0 and sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _compute_config_hash(config: dict) -> str:
    """Compute a deterministic SHA-256 hex digest of a config dict."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_manifest(
    seed: int,
    envtype: str,
    score: float,
    git_sha: Optional[str] = None,
    ci_run_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> SeedManifest:
    """Create a :class:`SeedManifest` for a single run.

    Parameters
    ----------
    seed:
        Random seed used for the run.
    envtype:
        Environment type identifier (e.g. ``'1s-d'``).
    score:
        Final score / reward of the run.
    git_sha:
        Git commit SHA.  Auto-detected via ``git rev-parse HEAD`` if omitted.
    ci_run_id:
        CI run identifier (e.g. GitHub Actions run ID).
    config:
        Configuration dict to hash.  Defaults to ``{"envtype": envtype}``.
    """
    if git_sha is None:
        git_sha = _detect_git_sha()

    if config is None:
        config = {"envtype": envtype}

    config_hash = _compute_config_hash(config)

    try:
        from flopyarcade import __version__ as env_version
    except ImportError:
        env_version = "unknown"

    timestamp = datetime.now(timezone.utc).isoformat()

    return SeedManifest(
        seed=seed,
        git_sha=git_sha,
        config_hash=config_hash,
        env_version=env_version,
        score=score,
        ci_run_id=ci_run_id,
        timestamp=timestamp,
    )


def save_manifest(manifest: SeedManifest, path: PathLike) -> Path:
    """Write *manifest* as pretty-printed JSON to *path*.

    Parent directories are created if they do not exist.
    Returns the resolved :class:`Path`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
        f.write("\n")
    return p


def load_manifest(path: PathLike) -> SeedManifest:
    """Read a manifest JSON file back into a :class:`SeedManifest`."""
    with open(path) as f:
        data = json.load(f)
    return SeedManifest(**data)


def append_to_log(
    manifest: SeedManifest,
    log_path: PathLike = "seed_manifests.jsonl",
) -> Path:
    """Append one JSON line for *manifest* to *log_path*.

    Creates the file (and parent directories) if it does not exist.
    Returns the resolved :class:`Path`.
    """
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(asdict(manifest)) + "\n")
    return p


def load_log(log_path: PathLike = "seed_manifests.jsonl") -> list[dict]:
    """Read all JSON lines from *log_path* into a list of dicts.

    Returns an empty list if the file does not exist.
    """
    p = Path(log_path)
    if not p.exists():
        return []
    entries: list[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
