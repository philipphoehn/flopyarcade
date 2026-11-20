"""Tests for flopyarcade.seed_manifest (ROADMAP P3)."""

import json
import os
import re
import tempfile
from pathlib import Path

import pytest

from flopyarcade.seed_manifest import (
    SeedManifest,
    append_to_log,
    create_manifest,
    load_log,
    load_manifest,
    save_manifest,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)$"
)


# ---------------------------------------------------------------------------
# 1 – create_manifest with known values
# ---------------------------------------------------------------------------

def test_create_manifest_known_values():
    m = create_manifest(
        seed=42,
        envtype="1s-d",
        score=3.14,
        git_sha="a" * 40,
        ci_run_id="run-123",
        config={"envtype": "1s-d", "steps": 100},
    )
    assert m.seed == 42
    assert m.git_sha == "a" * 40
    assert m.env_version  # non-empty
    assert m.score == pytest.approx(3.14)
    assert m.ci_run_id == "run-123"
    assert len(m.config_hash) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# 2 – config_hash is deterministic
# ---------------------------------------------------------------------------

def test_config_hash_deterministic():
    cfg = {"alpha": 1, "beta": [2, 3]}
    m1 = create_manifest(1, "1s-d", 0.0, git_sha="b" * 40, config=cfg)
    m2 = create_manifest(1, "1s-d", 0.0, git_sha="b" * 40, config=cfg)
    assert m1.config_hash == m2.config_hash


# ---------------------------------------------------------------------------
# 3 – different configs → different hashes
# ---------------------------------------------------------------------------

def test_different_configs_different_hash():
    m1 = create_manifest(1, "1s-d", 0.0, git_sha="b" * 40, config={"a": 1})
    m2 = create_manifest(1, "1s-d", 0.0, git_sha="b" * 40, config={"a": 2})
    assert m1.config_hash != m2.config_hash


# ---------------------------------------------------------------------------
# 4 – save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    m = create_manifest(
        seed=7,
        envtype="2s-d",
        score=1.5,
        git_sha="c" * 40,
        ci_run_id="ci-99",
    )
    p = tmp_path / "sub" / "manifest.json"
    save_manifest(m, p)
    assert p.exists()

    loaded = load_manifest(p)
    assert loaded == m


# ---------------------------------------------------------------------------
# 5 – append_to_log creates the file
# ---------------------------------------------------------------------------

def test_append_to_log_creates_file(tmp_path):
    m = create_manifest(1, "1s-d", 0.5, git_sha="d" * 40)
    log = tmp_path / "new_log.jsonl"
    assert not log.exists()

    append_to_log(m, log)
    assert log.exists()
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["seed"] == 1


# ---------------------------------------------------------------------------
# 6 – append_to_log appends (2nd call adds a line)
# ---------------------------------------------------------------------------

def test_append_to_log_appends(tmp_path):
    log = tmp_path / "log.jsonl"
    m1 = create_manifest(1, "1s-d", 0.1, git_sha="e" * 40)
    m2 = create_manifest(2, "2s-d", 0.2, git_sha="e" * 40)

    append_to_log(m1, log)
    append_to_log(m2, log)

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["seed"] == 1
    assert json.loads(lines[1])["seed"] == 2


# ---------------------------------------------------------------------------
# 7 – load_log reads back
# ---------------------------------------------------------------------------

def test_load_log_reads_back(tmp_path):
    log = tmp_path / "log.jsonl"
    for i in range(3):
        append_to_log(
            create_manifest(i, "1s-d", float(i), git_sha="f" * 40),
            log,
        )

    entries = load_log(log)
    assert len(entries) == 3
    assert [e["seed"] for e in entries] == [0, 1, 2]
    assert entries[1]["score"] == 1.0


# ---------------------------------------------------------------------------
# 8 – git_sha auto-detection
# ---------------------------------------------------------------------------

def test_git_sha_auto_detection():
    m = create_manifest(99, "1s-d", 1.0)
    assert m.git_sha == "unknown" or bool(SHA_RE.match(m.git_sha))


# ---------------------------------------------------------------------------
# 9 – timestamp is valid ISO format
# ---------------------------------------------------------------------------

def test_timestamp_iso_format():
    m = create_manifest(1, "1s-d", 0.0, git_sha="a" * 40)
    assert m.timestamp, "timestamp must not be empty"
    assert ISO_RE.match(m.timestamp), f"not ISO-8601: {m.timestamp!r}"
    # also verify it can be parsed by the stdlib
    from datetime import datetime
    datetime.fromisoformat(m.timestamp)


# ---------------------------------------------------------------------------
# 10 – full pipeline: create → save → load → compare
# ---------------------------------------------------------------------------

def test_full_pipeline(tmp_path):
    m = create_manifest(
        seed=2024,
        envtype="3s-d",
        score=42.0,
        git_sha="0" * 40,
        ci_run_id="gh-2024",
        config={"envtype": "3s-d", "n_steps": 500},
    )
    path = tmp_path / "full" / "pipeline" / "manifest.json"
    save_manifest(m, path)
    restored = load_manifest(path)

    assert restored.seed == m.seed
    assert restored.git_sha == m.git_sha
    assert restored.config_hash == m.config_hash
    assert restored.env_version == m.env_version
    assert restored.score == m.score
    assert restored.ci_run_id == m.ci_run_id
    assert restored.timestamp == m.timestamp
    assert restored == m


# ---------------------------------------------------------------------------
# bonus – load_log on missing file returns []
# ---------------------------------------------------------------------------

def test_load_log_missing_file(tmp_path):
    assert load_log(tmp_path / "nonexistent.jsonl") == []
