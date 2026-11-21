"""B16: Tests for progress GIF recorder."""

import os
import numpy as np
import pytest

from flopyarcade.progress_gif import ProgressGifRecorder


class TestProgressGifRecorderInit:
    def test_defaults(self):
        rec = ProgressGifRecorder()
        assert rec.output_path == 'progress.gif'
        assert rec.max_frames == 60
        assert rec.width == 800
        assert rec.height == 600
        assert rec.duration_ms == 300
        assert rec.frame_count == 0

    def test_custom_params(self):
        rec = ProgressGifRecorder(output_path='test.gif', max_frames=10,
                                  width=320, height=240, duration_ms=100)
        assert rec.output_path == 'test.gif'
        assert rec.max_frames == 10
        assert rec.width == 320
        assert rec.height == 240
        assert rec.duration_ms == 100


class TestCapture:
    def test_capture_array(self):
        rec = ProgressGifRecorder()
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rec.capture_array(arr, caption='frame 0')
        assert rec.frame_count == 1

    def test_capture_multiple(self):
        rec = ProgressGifRecorder()
        for i in range(5):
            arr = np.full((50, 50, 3), i * 50, dtype=np.uint8)
            rec.capture_array(arr, caption=f'gen {i}')
        assert rec.frame_count == 5

    def test_capture_empty_raises_on_finalize(self):
        rec = ProgressGifRecorder(output_path='/tmp/empty.gif')
        with pytest.raises(ValueError, match="No frames"):
            rec.finalize()


class TestSubsampling:
    def test_no_subsample_if_under_limit(self):
        rec = ProgressGifRecorder(max_frames=100)
        for i in range(10):
            rec.capture_array(np.zeros((10, 10, 3), dtype=np.uint8))
        frames = rec._subsample()
        assert len(frames) == 10

    def test_subsample_if_over_limit(self):
        rec = ProgressGifRecorder(max_frames=5)
        for i in range(20):
            rec.capture_array(np.full((10, 10, 3), i, dtype=np.uint8))
        frames = rec._subsample()
        assert len(frames) == 5


class TestFinalize:
    def test_creates_gif_file(self, tmp_path):
        out = str(tmp_path / 'test_progress.gif')
        rec = ProgressGifRecorder(output_path=out, width=100, height=100)
        for i in range(3):
            rec.capture_array(np.full((50, 50, 3), i * 80, dtype=np.uint8),
                              caption=f'gen {i}')
        result = rec.finalize()
        assert os.path.isfile(result)
        assert result == out

    def test_gif_is_valid(self, tmp_path):
        from PIL import Image
        out = str(tmp_path / 'valid.gif')
        rec = ProgressGifRecorder(output_path=out, width=64, height=64)
        for i in range(4):
            rec.capture_array(np.full((32, 32, 3), i * 60, dtype=np.uint8))
        rec.finalize()
        img = Image.open(out)
        assert img.format == 'GIF'

    def test_gif_multiple_frames(self, tmp_path):
        from PIL import Image
        out = str(tmp_path / 'multi.gif')
        rec = ProgressGifRecorder(output_path=out, width=64, height=64)
        for i in range(5):
            rec.capture_array(np.full((32, 32, 3), i * 50, dtype=np.uint8))
        rec.finalize()
        img = Image.open(out)
        n = getattr(img, 'n_frames', 1)
        assert n >= 4  # PIL may optimize similar frames


class TestReset:
    def test_reset_clears_frames(self):
        rec = ProgressGifRecorder()
        rec.capture_array(np.zeros((10, 10, 3), dtype=np.uint8))
        assert rec.frame_count == 1
        rec.reset()
        assert rec.frame_count == 0


def _has_matplotlib():
    try:
        import matplotlib
        return True
    except ImportError:
        return False


class TestCaptureMatplotlibFig:
    @pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not available")
    def test_capture_from_figure(self, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        out = str(tmp_path / 'fig.gif')
        rec = ProgressGifRecorder(output_path=out, width=200, height=150)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        rec.capture(fig, caption='test plot')
        plt.close(fig)
        assert rec.frame_count == 1
        rec.finalize()
        assert os.path.isfile(out)
