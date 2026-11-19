"""In-memory filesystem for MODFLOW work files.

Avoids disk I/O by storing MODFLOW input lists, binary output files,
and other work artifacts in a RAM-backed dict. Compatible with the
existing modflow_runner.py file I/O patterns.

Usage::

    from flopyarcade.ramdisk_fs import InMemoryFS

    fs = InMemoryFS()
    fs.write('mfsim.lst', 'BEGIN OPTIONS\\n...')
    data = fs.read('mfsim.lst')
    fs.write_bytes('mfsim.hds', b'\\x00\\x01...')
    exists = fs.exists('mfsim.hds')
    fs.delete('mfsim.lst')
    fs.clear()
"""

import io
import os
from typing import Dict, List, Optional


class InMemoryFile(io.BufferedIOBase):
    """File-like object backed by an in-memory buffer.

    Supports read, write, seek, tell, close, and context-manager.
    """

    def __init__(self, name: str, mode: str = 'r+'):
        super().__init__()
        self.name = name
        self.mode = mode
        self._buf = io.BytesIO()
        self._text_mode = 'b' not in mode

    def readable(self) -> bool:
        return 'r' in self.mode or '+' in self.mode

    def writable(self) -> bool:
        return 'w' in self.mode or '+' in self.mode or 'a' in self.mode

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1):
        data = self._buf.read(size)
        if self._text_mode:
            return data.decode('utf-8')
        return data

    def write(self, data) -> int:
        if self._text_mode:
            data = data.encode('utf-8')
        return self._buf.write(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._buf.seek(offset, whence)

    def tell(self) -> int:
        return self._buf.tell()

    def close(self) -> None:
        if not self.closed:
            self._buf.close()
            super().close()

    def getvalue(self) -> bytes:
        """Return the full contents as bytes."""
        pos = self._buf.tell()
        self._buf.seek(0)
        val = self._buf.read()
        self._buf.seek(pos)
        return val

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class InMemoryFS:
    """RAM-backed filesystem for MODFLOW work files.

    Provides a simple dict-based store with file-like access.
    All data lives in process memory; nothing touches disk.
    """

    def __init__(self, prefix: str = ''):
        self.prefix = prefix
        self._store: Dict[str, bytes] = {}

    def _key(self, name: str) -> str:
        return f"{self.prefix}{name}" if self.prefix else name

    @property
    def files(self) -> List[str]:
        """List of all stored file names (without prefix)."""
        p = self.prefix
        return [k[len(p):] if p and k.startswith(p) else k for k in self._store]

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        return self._key(name) in self._store

    def exists(self, name: str) -> bool:
        """Check if a file exists in the store."""
        return self._key(name) in self._store

    def write(self, name: str, content: str) -> int:
        """Write text content to a file. Returns bytes written."""
        data = content.encode('utf-8')
        self._store[self._key(name)] = data
        return len(data)

    def write_bytes(self, name: str, data: bytes) -> int:
        """Write binary content to a file. Returns bytes written."""
        self._store[self._key(name)] = data
        return len(data)

    def read(self, name: str) -> str:
        """Read text content. Raises FileNotFoundError if missing."""
        key = self._key(name)
        if key not in self._store:
            raise FileNotFoundError(f"[InMemoryFS] No such file: {name}")
        return self._store[key].decode('utf-8')

    def read_bytes(self, name: str) -> bytes:
        """Read binary content. Raises FileNotFoundError if missing."""
        key = self._key(name)
        if key not in self._store:
            raise FileNotFoundError(f"[InMemoryFS] No such file: {name}")
        return self._store[key]

    def open(self, name: str, mode: str = 'r+') -> InMemoryFile:
        """Open a file-like object. On close, buffer syncs to store."""
        f = InMemoryFile(name, mode)
        key = self._key(name)
        if key in self._store and 'w' not in mode and 'x' not in mode:
            f._buf.write(self._store[key])
            f._buf.seek(0)
        # Register write-back on close
        fs = self
        fs_key = key
        orig_close = f.close

        def _close_and_sync():
            if not f.closed:
                f._buf.seek(0)
                fs._store[fs_key] = f._buf.read()
                orig_close()

        f.close = _close_and_sync  # type: ignore
        return f

    def delete(self, name: str) -> bool:
        """Remove a file. Returns True if it existed."""
        key = self._key(name)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> int:
        """Remove all files. Returns count removed."""
        n = len(self._store)
        self._store.clear()
        return n

    def size(self, name: str) -> int:
        """Return size in bytes. Raises FileNotFoundError if missing."""
        key = self._key(name)
        if key not in self._store:
            raise FileNotFoundError(f"[InMemoryFS] No such file: {name}")
        return len(self._store[key])

    def total_size(self) -> int:
        """Total bytes across all stored files."""
        return sum(len(v) for v in self._store.values())

    def copy_from_disk(self, path: str, name: Optional[str] = None) -> str:
        """Copy a disk file into the store. Returns stored name."""
        if name is None:
            name = os.path.basename(path)
        with open(path, 'rb') as fh:
            self._store[self._key(name)] = fh.read()
        return name

    def save_to_disk(self, name: str, path: str) -> str:
        """Write a stored file to disk. Returns destination path."""
        data = self.read_bytes(name)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
        return path

    def to_dict(self) -> Dict[str, bytes]:
        """Return a shallow copy of the internal store."""
        return dict(self._store)

    @classmethod
    def from_dict(cls, data: Dict[str, bytes], prefix: str = '') -> 'InMemoryFS':
        """Create an InMemoryFS from a dict of bytes."""
        fs = cls(prefix=prefix)
        fs._store = dict(data)
        return fs
