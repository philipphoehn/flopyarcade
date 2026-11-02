"""Append-only HDF5 novelty archive replacing full-rewrite pickle (B8).

Provides lazy-load and append semantics so the archive file grows
without being fully rewritten each generation.
"""

import os

import h5py
import numpy as np


class NoveltyArchiveHDF5:
    """Append-only HDF5 novelty archive.

    On-disk layout::

        /agents
            /agent1
                novelty          (float64 scalar)
                actionsUniqueID  (int64 scalar)
                modelFile        (bytes string)
            /agent2
                ...
    """

    def __init__(self, path: str):
        self.path = path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """Read the full archive. Returns ``{agent_str: {field: value}}``."""
        archive: dict = {}
        if not os.path.exists(self.path):
            return archive
        with h5py.File(self.path, "r") as f:
            if "agents" not in f:
                return archive
            for key in f["agents"].keys():
                grp = f["agents"][key]
                entry: dict = {}
                for field in ("novelty", "actionsUniqueID", "modelFile"):
                    if field in grp:
                        raw = grp[field][()]
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        entry[field] = raw
                archive[key] = entry
        return archive

    def save(self, archive: dict) -> None:
        """Write (or append) entries. Existing keys are overwritten.

        This is *append-friendly*: opening an existing file and calling
        ``save`` with a superset dict simply upserts the new keys and
        leaves the rest untouched.
        """
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        mode = "a" if os.path.exists(self.path) else "w"
        with h5py.File(self.path, mode) as f:
            if "agents" not in f:
                f.create_group("agents")
            for key, entry in archive.items():
                if key not in f["agents"]:
                    f["agents"].create_group(key)
                grp = f["agents"][key]
                for field, value in entry.items():
                    if field in grp:
                        del grp[field]
                    if isinstance(value, str):
                        grp.create_dataset(field, data=value.encode("utf-8"))
                    elif isinstance(value, (int, np.integer)):
                        grp.create_dataset(field, data=np.int64(value))
                    elif isinstance(value, (float, np.floating)):
                        grp.create_dataset(field, data=np.float64(value))
                    else:
                        grp.create_dataset(field, data=np.array(value, dtype="S"))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with h5py.File(self.path, "r") as f:
            if "agents" not in f:
                return 0
            return len(f["agents"].keys())

    def keys(self) -> list:
        if not os.path.exists(self.path):
            return []
        with h5py.File(self.path, "r") as f:
            if "agents" not in f:
                return []
            return list(f["agents"].keys())
