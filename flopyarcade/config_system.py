#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""FloPyArcade – Hydra-style layered configuration system.

Layered priority (lowest → highest):
  1. Built-in defaults (ConfigSchema)
  2. YAML / JSON config file
  3. CLI overrides  (key=value pairs)

Usage
-----
>>> from flopyarcade.config_system import ConfigLoader
>>> loader = ConfigLoader()
>>> cfg = loader.build_config(yaml_path='configs/training.yaml',
...                           cli_overrides=['lr=0.005', 'n_episodes=200'])
>>> print(loader.format_config_report(cfg))
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields, asdict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional YAML support – fall back to JSON when PyYAML is unavailable
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore
    _HAS_YAML = False

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class ConfigSchema:
    """Typed configuration for a FloPyArcade training run."""

    envtype: str = '1s-d'
    seed: int = 42
    n_episodes: int = 100
    agent_type: str = 'dqn'
    lr: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    reward_scale: float = 1.0
    n_particles: int = 1
    use_cost_mode: bool = False


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Load, merge, validate, and serialise FloPyArcade configs."""

    def __init__(self, base_dir: str = '.'):  # noqa: D107
        self.base_dir = base_dir

    # ---- YAML / JSON I/O ------------------------------------------------

    def load_yaml(self, path: str) -> dict:
        """Read a YAML or JSON config file and return a dict.

        Falls back to JSON parsing when PyYAML is not installed.
        """
        abs_path = os.path.join(self.base_dir, path) if not os.path.isabs(path) else path
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Config file not found: {abs_path}")

        with open(abs_path, 'r', encoding='utf-8') as fh:
            text = fh.read()

        if _HAS_YAML and (abs_path.endswith('.yaml') or abs_path.endswith('.yml')
                          or self._looks_like_yaml(text)):
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Top-level config must be a mapping, got {type(data).__name__}")
        return data

    @staticmethod
    def _looks_like_yaml(text: str) -> bool:
        """Heuristic: if the first non-empty line has no '{' it's probably YAML."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return not stripped.startswith('{') and not stripped.startswith('[')
        return False

    # ---- Deep merge -----------------------------------------------------

    def merge_configs(self, base: dict, override: dict) -> dict:
        """Recursively merge *override* into *base*; override wins."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    # ---- Apply overrides to schema --------------------------------------

    def apply_overrides(self, schema: ConfigSchema, overrides: dict) -> ConfigSchema:
        """Set fields on *schema* from *overrides*; unknown keys are ignored."""
        valid_fields = {f.name for f in fields(ConfigSchema)}
        for key, value in overrides.items():
            if key in valid_fields:
                setattr(schema, key, value)
        return schema

    # ---- CLI parsing ----------------------------------------------------

    def parse_cli_overrides(self, args: list[str]) -> dict:
        """Parse ``key=value`` pairs into a typed dict.

        Supported types inferred from the schema field type:
        int, float, bool (true/false), str.
        """
        result: dict[str, Any] = {}
        schema_defaults = asdict(ConfigSchema())

        for arg in args:
            if '=' not in arg:
                raise ValueError(f"Expected key=value, got: {arg!r}")
            key, _, raw = arg.partition('=')
            key = key.strip()
            raw = raw.strip()

            if key not in schema_defaults:
                # unknown key – store as-is; apply_overrides will ignore it
                result[key] = raw
                continue

            target_type = type(schema_defaults[key])
            if target_type is bool:
                result[key] = raw.lower() in ('true', '1', 'yes')
            elif target_type is int:
                result[key] = int(raw)
            elif target_type is float:
                result[key] = float(raw)
            else:
                result[key] = raw

        return result

    # ---- Full pipeline --------------------------------------------------

    def build_config(
        self,
        yaml_path: str | None = None,
        cli_overrides: list[str] | None = None,
    ) -> ConfigSchema:
        """Build a ConfigSchema: defaults → YAML file → CLI overrides."""
        schema = ConfigSchema()

        # Layer 1: YAML / JSON file
        if yaml_path:
            file_dict = self.load_yaml(yaml_path)
            self.apply_overrides(schema, file_dict)

        # Layer 2: CLI overrides
        if cli_overrides:
            cli_dict = self.parse_cli_overrides(cli_overrides)
            self.apply_overrides(schema, cli_dict)

        return schema

    # ---- Validation -----------------------------------------------------

    def validate(self, config: ConfigSchema) -> list[str]:
        """Return a list of warnings/errors; empty list means OK."""
        issues: list[str] = []

        if config.epsilon_start < config.epsilon_end:
            issues.append(
                f"epsilon_start ({config.epsilon_start}) < epsilon_end ({config.epsilon_end}) "
                f"– exploration will increase instead of decay."
            )
        if config.n_episodes < 1:
            issues.append(f"n_episodes must be >= 1, got {config.n_episodes}")
        if config.lr <= 0:
            issues.append(f"lr must be positive, got {config.lr}")
        if config.gamma < 0 or config.gamma >= 1:
            issues.append(f"gamma must be in [0, 1), got {config.gamma}")
        if config.n_particles < 1:
            issues.append(f"n_particles must be >= 1, got {config.n_particles}")
        if config.epsilon_decay < 1:
            issues.append(f"epsilon_decay must be >= 1, got {config.epsilon_decay}")

        return issues

    # ---- Serialisation --------------------------------------------------

    def to_dict(self, config: ConfigSchema) -> dict:
        """Convert a ConfigSchema to a plain dict."""
        return asdict(config)

    @classmethod
    def from_dict(cls, d: dict) -> ConfigSchema:
        """Build a ConfigSchema from a dict (extra keys ignored)."""
        valid = {f.name for f in fields(ConfigSchema)}
        filtered = {k: v for k, v in d.items() if k in valid}
        return ConfigSchema(**filtered)

    def save_yaml(self, config: ConfigSchema, path: str) -> None:
        """Write config to YAML (or JSON fallback)."""
        abs_path = os.path.join(self.base_dir, path) if not os.path.isabs(path) else path
        os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)

        d = self.to_dict(config)
        if _HAS_YAML:
            with open(abs_path, 'w', encoding='utf-8') as fh:
                yaml.dump(d, fh, default_flow_style=False, sort_keys=False)
        else:
            with open(abs_path, 'w', encoding='utf-8') as fh:
                json.dump(d, fh, indent=2)

    @classmethod
    def load_yaml_config(cls, path: str) -> ConfigSchema:
        """Classmethod convenience: load a YAML/JSON file into a ConfigSchema."""
        loader = cls()
        d = loader.load_yaml(path)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def generate_default_config() -> dict:
    """Return the default ConfigSchema as a dict."""
    return asdict(ConfigSchema())


def generate_env_config(envtype: str) -> dict:
    """Return tuned defaults for a specific environment type.

    Multi-well / continuous-action envs get more episodes and slower decay.
    """
    base = asdict(ConfigSchema())

    # Number of action dims as a proxy for complexity
    _COMPLEX = re.compile(r'^(3|4|5|6)[src]')
    _CONTINUOUS = envtype.endswith('-c')
    _COST = 'cost' in envtype

    if _COMPLEX.match(envtype):
        base['n_episodes'] = 500
        base['epsilon_decay'] = 50000
    if _CONTINUOUS:
        base['lr'] = 0.0005
        base['gamma'] = 0.98
    if _COST:
        base['use_cost_mode'] = True
        base['reward_scale'] = 0.5
    base['envtype'] = envtype

    return base


def format_config_report(config: ConfigSchema) -> str:
    """Return a human-readable ``key=value`` multi-line report."""
    d = asdict(config)
    return '\n'.join(f'{k}={v}' for k, v in d.items())


def config_diff(config_a: ConfigSchema, config_b: ConfigSchema) -> dict:
    """Return ``{key: {a, b, changed}}`` for every field that differs."""
    a = asdict(config_a)
    b = asdict(config_b)
    diff: dict[str, dict] = {}
    for key in a:
        if a[key] != b[key]:
            diff[key] = {'a': a[key], 'b': b[key], 'changed': True}
    return diff


__all__ = [
    'ConfigSchema',
    'ConfigLoader',
    'generate_default_config',
    'generate_env_config',
    'format_config_report',
    'config_diff',
]
