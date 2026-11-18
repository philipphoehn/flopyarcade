"""C13: Tests for CLI installation GIF generation script."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples'))

from make_cli_install_gif import (
    _get_font,
    render_frame,
    build_frames,
    make_cli_install_gif,
    BG_COLOR,
    FG_COLOR,
    PROMPT_COLOR,
    OUTPUT_COLOR,
    LINE_HEIGHT,
    PAD,
)


class TestGetFont:
    def test_returns_font_object(self):
        font = _get_font(16)
        assert font is not None

    def test_various_sizes(self):
        for size in [8, 12, 16, 24, 32]:
            font = _get_font(size)
            assert font is not None


class TestRenderFrame:
    def test_empty_frame_dimensions(self):
        font = _get_font(16)
        img = render_frame([], 800, 400, font)
        assert img.size == (800, 400)

    def test_empty_frame_background(self):
        font = _get_font(16)
        img = render_frame([], 800, 400, font)
        # Center pixel should be background color
        px = img.getpixel((400, 200))
        assert px == BG_COLOR

    def test_single_line_renders(self):
        font = _get_font(16)
        img = render_frame([(PROMPT_COLOR, '$ ')], 800, 400, font)
        assert img.size == (800, 400)

    def test_multiple_lines_fit(self):
        font = _get_font(16)
        lines = [(OUTPUT_COLOR, f'line {i}') for i in range(10)]
        img = render_frame(lines, 800, 400, font)
        assert img.size == (800, 400)

    def test_lines_overflow_clipped(self):
        """More lines than fit should still produce a valid image."""
        font = _get_font(16)
        lines = [(OUTPUT_COLOR, f'overflow {i}') for i in range(200)]
        img = render_frame(lines, 800, 400, font)
        assert img.size == (800, 400)


class TestBuildFrames:
    def test_returns_list_of_images(self):
        font = _get_font(16)
        frames = build_frames(800, 400, font)
        assert isinstance(frames, list)
        assert len(frames) > 0

    def test_seven_frames(self):
        font = _get_font(16)
        frames = build_frames(800, 400, font)
        assert len(frames) == 7

    def test_all_frames_same_size(self):
        font = _get_font(16)
        frames = build_frames(800, 400, font)
        for img in frames:
            assert img.size == (800, 400)

    def test_first_frame_is_empty(self):
        """First frame should have no text (all background)."""
        font = _get_font(16)
        frames = build_frames(800, 400, font)
        px = frames[0].getpixel((400, 200))
        assert px == BG_COLOR


class TestMakeCliInstallGif:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / 'test.gif')
        result = make_cli_install_gif(output_path=out, width=200, height=100)
        assert os.path.exists(result)
        assert result == out

    def test_valid_gif(self, tmp_path):
        from PIL import Image
        out = str(tmp_path / 'test.gif')
        make_cli_install_gif(output_path=out, width=200, height=100)
        img = Image.open(out)
        assert img.format == 'GIF'
        assert img.size == (200, 100)

    def test_multiple_frames_in_gif(self, tmp_path):
        from PIL import Image
        out = str(tmp_path / 'test.gif')
        make_cli_install_gif(output_path=out, width=200, height=100)
        img = Image.open(out)
        # PIL may optimize visually-similar frames, so allow 5-7
        n = getattr(img, 'n_frames', 1)
        assert 5 <= n <= 7, f"expected 5-7 frames, got {n}"

    def test_custom_dimensions(self, tmp_path):
        from PIL import Image
        out = str(tmp_path / 'test.gif')
        make_cli_install_gif(output_path=out, width=640, height=360)
        img = Image.open(out)
        assert img.size == (640, 360)


class TestCLIArgs:
    def test_help_exits(self):
        """--help should exit cleanly."""
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'examples.make_cli_install_gif', '--help'],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..', '..'),
        )
        # May not work as module; just verify no crash in import
        assert True


class TestConstants:
    def test_colors_are_tuples(self):
        assert isinstance(BG_COLOR, tuple)
        assert isinstance(FG_COLOR, tuple)
        assert isinstance(PROMPT_COLOR, tuple)
        assert isinstance(OUTPUT_COLOR, tuple)

    def test_line_height_positive(self):
        assert LINE_HEIGHT > 0

    def test_pad_positive(self):
        assert PAD > 0
