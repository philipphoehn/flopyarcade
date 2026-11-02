#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Tests for B8: HDF5 novelty archive."""

import os
import tempfile

import numpy as np
import pytest

from flopyarcade.archive_hdf5 import NoveltyArchiveHDF5


class TestNoveltyArchiveHDF5:
    """Verify save/load/append semantics of the HDF5 archive."""

    @pytest.fixture
    def archive_path(self, tmp_path):
        return str(tmp_path / "test_novelty.h5")

    def test_load_nonexistent_returns_empty(self, archive_path):
        arc = NoveltyArchiveHDF5(archive_path)
        assert arc.load() == {}
        assert len(arc) == 0
        assert arc.keys() == []

    def test_save_load_roundtrip(self, archive_path):
        data = {
            "agent1": {"novelty": 0.5, "actionsUniqueID": 42, "modelFile": "m1.p"},
            "agent2": {"novelty": 1.3, "actionsUniqueID": 7, "modelFile": "m2.p"},
        }
        arc = NoveltyArchiveHDF5(archive_path)
        arc.save(data)

        loaded = arc.load()
        assert set(loaded.keys()) == {"agent1", "agent2"}
        assert loaded["agent1"]["novelty"] == 0.5
        assert loaded["agent1"]["actionsUniqueID"] == 42
        assert loaded["agent1"]["modelFile"] == "m1.p"
        assert loaded["agent2"]["novelty"] == 1.3
        assert len(arc) == 2

    def test_append_preserves_existing(self, archive_path):
        # Initial save
        arc = NoveltyArchiveHDF5(archive_path)
        arc.save({"agent1": {"novelty": 0.1, "actionsUniqueID": 1, "modelFile": "a.p"}})

        # Append new agent
        arc.save({
            "agent1": {"novelty": 0.1, "actionsUniqueID": 1, "modelFile": "a.p"},
            "agent2": {"novelty": 0.9, "actionsUniqueID": 2, "modelFile": "b.p"},
        })

        loaded = arc.load()
        assert "agent1" in loaded
        assert "agent2" in loaded
        assert loaded["agent1"]["novelty"] == 0.1
        assert loaded["agent2"]["novelty"] == 0.9
        assert len(arc) == 2

    def test_overwrite_existing_entry(self, archive_path):
        arc = NoveltyArchiveHDF5(archive_path)
        arc.save({"agent1": {"novelty": 0.1, "actionsUniqueID": 1, "modelFile": "old.p"}})
        arc.save({"agent1": {"novelty": 0.7, "actionsUniqueID": 99, "modelFile": "new.p"}})

        loaded = arc.load()
        assert loaded["agent1"]["novelty"] == 0.7
        assert loaded["agent1"]["actionsUniqueID"] == 99
        assert loaded["agent1"]["modelFile"] == "new.p"

    def test_keys_method(self, archive_path):
        arc = NoveltyArchiveHDF5(archive_path)
        arc.save({
            "agent1": {"novelty": 0.1, "actionsUniqueID": 1, "modelFile": "a.p"},
            "agent2": {"novelty": 0.2, "actionsUniqueID": 2, "modelFile": "b.p"},
        })
        assert set(arc.keys()) == {"agent1", "agent2"}

    def test_numpy_types_roundtrip(self, archive_path):
        arc = NoveltyArchiveHDF5(archive_path)
        arc.save({"agent1": {"novelty": np.float64(3.14), "actionsUniqueID": np.int64(42), "modelFile": "x.p"}})

        loaded = arc.load()
        assert loaded["agent1"]["novelty"] == pytest.approx(3.14)
        assert int(loaded["agent1"]["actionsUniqueID"]) == 42
        assert loaded["agent1"]["modelFile"] == "x.p"
