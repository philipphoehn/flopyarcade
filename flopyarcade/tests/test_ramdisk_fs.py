"""B10: Tests for in-memory filesystem (RAM-disk) for MODFLOW work files."""

import os
import tempfile
import pytest

from flopyarcade.ramdisk_fs import InMemoryFS, InMemoryFile


class TestInMemoryFile:
    def test_read_write_text(self):
        f = InMemoryFile('test.txt', mode='r+')
        f.write('hello world')
        f.seek(0)
        data = f.read()
        assert data == 'hello world'
        f.close()

    def test_read_write_bytes(self):
        f = InMemoryFile('test.bin', mode='r+b')
        f.write(b'\x00\x01\x02')
        f.seek(0)
        data = f.read()
        assert data == b'\x00\x01\x02'
        f.close()

    def test_seek_tell(self):
        f = InMemoryFile('seek.txt', mode='r+')
        f.write('abcdef')
        f.seek(3)
        assert f.tell() == 3
        f.seek(0, 2)  # SEEK_END
        assert f.tell() == 6
        f.close()

    def test_getvalue(self):
        f = InMemoryFile('val.txt', mode='r+')
        f.write('12345')
        assert f.getvalue() == b'12345'
        f.close()

    def test_context_manager(self):
        with InMemoryFile('ctx.txt', mode='r+') as f:
            f.write('context')
        assert f.closed

    def test_readable_writable_seekable(self):
        f = InMemoryFile('flags.txt', mode='r+')
        assert f.readable()
        assert f.writable()
        assert f.seekable()
        f.close()


class TestInMemoryFSBasic:
    def test_write_read_text(self):
        fs = InMemoryFS()
        fs.write('mfsim.lst', 'BEGIN OPTIONS\nNOPRINT\nEND OPTIONS')
        content = fs.read('mfsim.lst')
        assert 'BEGIN OPTIONS' in content
        assert 'NOPRINT' in content

    def test_write_read_bytes(self):
        fs = InMemoryFS()
        data = b'\x00\x01\x02\x03\x04'
        fs.write_bytes('output.hds', data)
        assert fs.read_bytes('output.hds') == data

    def test_write_returns_byte_count(self):
        fs = InMemoryFS()
        n = fs.write('file.txt', 'hello')
        assert n == 5

    def test_write_bytes_returns_byte_count(self):
        fs = InMemoryFS()
        n = fs.write_bytes('file.bin', b'\x00' * 100)
        assert n == 100

    def test_exists(self):
        fs = InMemoryFS()
        assert not fs.exists('nope.txt')
        fs.write('nope.txt', 'x')
        assert fs.exists('nope.txt')

    def test_contains(self):
        fs = InMemoryFS()
        fs.write('a.txt', 'a')
        assert 'a.txt' in fs
        assert 'b.txt' not in fs

    def test_len(self):
        fs = InMemoryFS()
        assert len(fs) == 0
        fs.write('a', '1')
        fs.write('b', '2')
        assert len(fs) == 2

    def test_files_list(self):
        fs = InMemoryFS()
        fs.write('a.txt', 'a')
        fs.write('b.txt', 'b')
        files = fs.files
        assert 'a.txt' in files
        assert 'b.txt' in files

    def test_delete_existing(self):
        fs = InMemoryFS()
        fs.write('temp.txt', 'data')
        assert fs.delete('temp.txt') is True
        assert not fs.exists('temp.txt')

    def test_delete_nonexistent(self):
        fs = InMemoryFS()
        assert fs.delete('ghost.txt') is False

    def test_clear(self):
        fs = InMemoryFS()
        fs.write('a', '1')
        fs.write('b', '2')
        n = fs.clear()
        assert n == 2
        assert len(fs) == 0


class TestInMemoryFSErrors:
    def test_read_nonexistent_raises(self):
        fs = InMemoryFS()
        with pytest.raises(FileNotFoundError, match='No such file'):
            fs.read('missing.txt')

    def test_read_bytes_nonexistent_raises(self):
        fs = InMemoryFS()
        with pytest.raises(FileNotFoundError, match='No such file'):
            fs.read_bytes('missing.bin')

    def test_size_nonexistent_raises(self):
        fs = InMemoryFS()
        with pytest.raises(FileNotFoundError):
            fs.size('missing.txt')


class TestInMemoryFSSize:
    def test_size(self):
        fs = InMemoryFS()
        fs.write('data.txt', '1234567890')
        assert fs.size('data.txt') == 10

    def test_total_size(self):
        fs = InMemoryFS()
        fs.write('a', 'abc')
        fs.write('b', 'de')
        assert fs.total_size() == 5

    def test_total_size_empty(self):
        fs = InMemoryFS()
        assert fs.total_size() == 0


class TestInMemoryFSPrefix:
    def test_prefix_isolation(self):
        fs = InMemoryFS(prefix='session1/')
        fs.write('file.txt', 'data')
        assert fs.exists('file.txt')
        # Internal key has prefix
        assert 'session1/file.txt' in fs._store

    def test_prefix_files_list_strips_prefix(self):
        fs = InMemoryFS(prefix='run_42/')
        fs.write('mfsim.lst', 'x')
        assert 'mfsim.lst' in fs.files
        assert 'run_42/mfsim.lst' not in fs.files

    def test_two_sessions_isolated(self):
        fs1 = InMemoryFS(prefix='s1/')
        fs2 = InMemoryFS(prefix='s2/')
        fs1.write('file.txt', 'one')
        fs2.write('file.txt', 'two')
        assert fs1.read('file.txt') == 'one'
        assert fs2.read('file.txt') == 'two'


class TestInMemoryFSOpen:
    def test_open_write_close_syncs(self):
        fs = InMemoryFS()
        f = fs.open('work.txt', mode='w')
        f.write('modflow input')
        f.close()
        assert fs.read('work.txt') == 'modflow input'

    def test_open_existing_loads_content(self):
        fs = InMemoryFS()
        fs.write('existing.txt', 'original')
        f = fs.open('existing.txt', mode='r+')
        f.seek(0)
        assert f.read() == 'original'
        f.close()

    def test_open_context_manager(self):
        fs = InMemoryFS()
        with fs.open('ctx.txt', mode='w') as f:
            f.write('context manager data')
        assert fs.read('ctx.txt') == 'context manager data'


class TestInMemoryFSDiskInterop:
    def test_copy_from_disk(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
            tf.write('disk content')
            tmp_path = tf.name
        try:
            fs = InMemoryFS()
            name = fs.copy_from_disk(tmp_path)
            assert name == os.path.basename(tmp_path)
            assert fs.read(name) == 'disk content'
        finally:
            os.unlink(tmp_path)

    def test_copy_from_disk_custom_name(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lst', delete=False) as tf:
            tf.write('BEGIN OPTIONS')
            tmp_path = tf.name
        try:
            fs = InMemoryFS()
            fs.copy_from_disk(tmp_path, name='mfsim.lst')
            assert fs.exists('mfsim.lst')
            assert 'BEGIN OPTIONS' in fs.read('mfsim.lst')
        finally:
            os.unlink(tmp_path)

    def test_save_to_disk(self):
        fs = InMemoryFS()
        fs.write('output.hds', 'binary-like content')
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, 'out', 'output.hds')
            fs.save_to_disk('output.hds', dest)
            assert os.path.exists(dest)
            with open(dest, 'r') as f:
                assert f.read() == 'binary-like content'


class TestInMemoryFSSerialization:
    def test_to_dict(self):
        fs = InMemoryFS()
        fs.write('a.txt', 'hello')
        d = fs.to_dict()
        assert 'a.txt' in d
        assert d['a.txt'] == b'hello'

    def test_from_dict(self):
        fs = InMemoryFS.from_dict({'x.txt': b'world'})
        assert fs.read('x.txt') == 'world'

    def test_from_dict_with_prefix(self):
        fs = InMemoryFS.from_dict({'run1/data.bin': b'\x00\x01'}, prefix='run1/')
        assert fs.exists('data.bin')
        assert fs.read_bytes('data.bin') == b'\x00\x01'


class TestInMemoryFSModflowScenario:
    """Simulate a realistic MODFLOW work-file lifecycle."""

    def test_full_lifecycle(self):
        fs = InMemoryFS(prefix='game_001/')
        # Write MODFLOW input list
        fs.write('mfsim.lst', 'BEGIN OPTIONS\nNOPRINT\nEND OPTIONS\nBEGIN DIS\nEND')
        # Write binary output
        fs.write_bytes('mfsim.hds', b'\x00' * 512)
        # Write stdout log
        fs.write('mfsim.stdout', 'MF6 v6.4.0\nCompleted successfully.')

        # Verify all present
        assert len(fs) == 3
        assert fs.exists('mfsim.lst')
        assert fs.exists('mfsim.hds')
        assert fs.exists('mfsim.stdout')

        # Read back
        assert 'NOPRINT' in fs.read('mfsim.lst')
        assert fs.size('mfsim.hds') == 512

        # Cleanup between games
        n_cleared = fs.clear()
        assert n_cleared == 3
        assert len(fs) == 0


class TestInMemoryFSPerformance:
    def test_many_files(self):
        fs = InMemoryFS()
        for i in range(1000):
            fs.write(f'file_{i:04d}.txt', f'content {i}')
        assert len(fs) == 1000
        assert fs.total_size() > 0
        assert fs.read('file_0999.txt') == 'content 999'
