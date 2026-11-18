#!/usr/bin/python3
"""Generate a GIF showing FloPyArcade CLI installation and basic usage.

Renders a sequence of terminal-style frames:
  1. $ pip install flopyarcade
  2. $ python -m flopyarcade.play --envtype 1s-d --ngames 5 --nagentsteps 10
  3. Output: game # 1, reward line

Usage:
    python examples/make_cli_install_gif.py
    python examples/make_cli_install_gif.py --output my_install.gif --width 800 --height 400
"""

import argparse
import os
import sys
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


# Terminal styling
BG_COLOR = (30, 30, 30)
FG_COLOR = (200, 200, 200)
PROMPT_COLOR = (80, 250, 120)
OUTPUT_COLOR = (180, 180, 180)
TITLE_COLOR = (100, 180, 255)
LINE_HEIGHT = 22
PAD = 12


def _get_font(size: int = 16) -> ImageFont.ImageFont:
    """Return a monospace font, falling back to default."""
    for path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_frame(lines: List[Tuple[str, str]], width: int, height: int,
                  font: ImageFont.ImageFont) -> Image.Image:
    """Render a single terminal frame.

    Args:
        lines: List of (color, text) tuples.
        width: Frame width in pixels.
        height: Frame height in pixels.
        font: PIL font object.

    Returns:
        PIL Image.
    """
    img = Image.new('RGB', (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    y = PAD
    for color, text in lines:
        if y + LINE_HEIGHT > height - PAD:
            break
        if color and text:
            draw.text((PAD, y), text, fill=color, font=font)
        y += LINE_HEIGHT
    return img


def build_frames(width: int, height: int, font: ImageFont.ImageFont) -> List[Image.Image]:
    """Build the sequence of GIF frames.

    Returns:
        List of PIL Images (one per frame).
    """
    frames = []

    # Frame 1: empty terminal
    frames.append(render_frame([], width, height, font))

    # Frame 2: prompt appears
    frames.append(render_frame(
        [(PROMPT_COLOR, '$ ')], width, height, font))

    # Frame 3: pip install command
    frames.append(render_frame([
        (PROMPT_COLOR, '$ '),
        (FG_COLOR, 'pip install flopyarcade'),
    ], width, height, font))

    # Frame 4: pip output
    frames.append(render_frame([
        (PROMPT_COLOR, '$ '),
        (FG_COLOR, 'pip install flopyarcade'),
        (OUTPUT_COLOR, 'Collecting flopyarcade'),
        (OUTPUT_COLOR, '  Downloading flopyarcade-1.0.0-py3-none-any.whl'),
        (OUTPUT_COLOR, 'Successfully installed flopyarcade-1.0.0'),
    ], width, height, font))

    # Frame 5: new prompt + play command
    frames.append(render_frame([
        (PROMPT_COLOR, '$ '),
        (FG_COLOR, 'pip install flopyarcade'),
        (OUTPUT_COLOR, 'Successfully installed flopyarcade-1.0.0'),
        ('', ''),
        (PROMPT_COLOR, '$ '),
        (FG_COLOR, 'python -m flopyarcade.play --envtype 1s-d --ngames 3'),
    ], width, height, font))

    # Frame 6: game output
    frames.append(render_frame([
        (PROMPT_COLOR, '$ '),
        (FG_COLOR, 'pip install flopyarcade'),
        (OUTPUT_COLOR, 'Successfully installed flopyarcade-1.0.0'),
        ('', ''),
        (PROMPT_COLOR, '$ '),
        (FG_COLOR, 'python -m flopyarcade.play --envtype 1s-d --ngames 3'),
        (OUTPUT_COLOR, 'game #: 1'),
        (OUTPUT_COLOR, 'score: 42'),
        (OUTPUT_COLOR, 'time taken 1.23'),
    ], width, height, font))

    # Frame 7: all games done
    frames.append(render_frame([
        (PROMPT_COLOR, '$ '),
        (FG_COLOR, 'pip install flopyarcade'),
        (OUTPUT_COLOR, 'Successfully installed flopyarcade-1.0.0'),
        ('', ''),
        (PROMPT_COLOR, '$ '),
        (FG_COLOR, 'python -m flopyarcade.play --envtype 1s-d --ngames 3'),
        (OUTPUT_COLOR, 'game #: 1'),
        (OUTPUT_COLOR, 'score: 42'),
        (OUTPUT_COLOR, 'time taken 1.23'),
        (OUTPUT_COLOR, 'game #: 2'),
        (OUTPUT_COLOR, 'score: 38'),
        (OUTPUT_COLOR, 'time taken 1.18'),
        (OUTPUT_COLOR, 'game #: 3'),
        (OUTPUT_COLOR, 'score: 45'),
        (OUTPUT_COLOR, 'time taken 1.21'),
    ], width, height, font))

    return frames


def make_cli_install_gif(output_path: str = 'cli_install.gif',
                          width: int = 800,
                          height: int = 400,
                          duration_ms: int = 800) -> str:
    """Generate the CLI installation GIF.

    Args:
        output_path: Where to save the GIF.
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_ms: Duration of each frame in milliseconds.

    Returns:
        The output file path.
    """
    font = _get_font(16)
    frames = build_frames(width, height, font)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate FloPyArcade CLI installation GIF')
    parser.add_argument('--output', default='cli_install.gif', type=str,
                        help='Output GIF path (default: cli_install.gif)')
    parser.add_argument('--width', default=800, type=int,
                        help='Frame width in pixels (default: 800)')
    parser.add_argument('--height', default=400, type=int,
                        help='Frame height in pixels (default: 400)')
    parser.add_argument('--duration', default=800, type=int,
                        help='Frame duration in ms (default: 800)')
    args = parser.parse_args()

    out = make_cli_install_gif(
        output_path=args.output,
        width=args.width,
        height=args.height,
        duration_ms=args.duration,
    )
    print(f'GIF saved to {out}')


if __name__ == '__main__':
    main()
