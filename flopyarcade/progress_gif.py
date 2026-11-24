"""Progress GIF recorder for evolution runs.

Captures matplotlib figure frames during an evolution loop and
assembles them into an animated GIF showing progress over time.

Usage::

    from flopyarcade.progress_gif import ProgressGifRecorder

    rec = ProgressGifRecorder(output_path='evolution_progress.gif', max_frames=60)
    for gen in range(max_gens):
        # ... run one generation ...
        rec.capture(fig, caption=f'Gen {gen+1}, best={best_reward:.4f}')
    rec.finalize()
"""

import os
from typing import List, Optional

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class ProgressGifRecorder:
    """Record matplotlib figures and assemble into a progress GIF.

    Args:
        output_path: File path for the final GIF.
        max_frames: Maximum number of frames to store (subsample if exceeded).
        width: GIF frame width in pixels (default 800).
        height: GIF frame height in pixels (default 600).
        duration_ms: Frame duration in milliseconds (default 300).
    """

    def __init__(self, output_path: str = 'progress.gif',
                 max_frames: int = 60,
                 width: int = 800,
                 height: int = 600,
                 duration_ms: int = 300):
        self.output_path = output_path
        self.max_frames = max_frames
        self.width = width
        self.height = height
        self.duration_ms = duration_ms
        self._frames: List[np.ndarray] = []
        self._captions: List[str] = []

    @property
    def frame_count(self) -> int:
        """Number of frames captured so far."""
        return len(self._frames)

    def capture(self, fig, caption: str = '') -> None:
        """Capture a matplotlib figure as a frame.

        Args:
            fig: A matplotlib Figure object.
            caption: Optional text caption displayed below the plot.
        """
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Convert RGBA to RGB
        rgb = buf[:, :, :3]
        self._frames.append(rgb)
        self._captions.append(caption)

    def capture_array(self, array: np.ndarray, caption: str = '') -> None:
        """Capture a raw numpy array as a frame.

        Args:
            array: H x W x 3 RGB array.
            caption: Optional caption.
        """
        self._frames.append(array)
        self._captions.append(caption)

    def finalize(self) -> str:
        """Assemble captured frames into a GIF file.

        If more frames were captured than max_frames, uniformly subsamples.

        Returns:
            The output file path.
        """
        if not self._frames:
            raise ValueError("No frames captured; call capture() before finalize().")

        frames = self._subsample()
        captions = self._captions[:len(frames)] if len(self._captions) >= len(frames) else self._captions

        if _HAS_PIL:
            return self._write_with_pil(frames, captions)
        else:
            return self._write_fallback(frames, captions)

    def _subsample(self) -> List[np.ndarray]:
        """Uniformly subsample frames to max_frames."""
        if len(self._frames) <= self.max_frames:
            return self._frames
        indices = np.linspace(0, len(self._frames) - 1, self.max_frames).astype(int)
        return [self._frames[i] for i in indices]

    def _write_with_pil(self, frames: List[np.ndarray], captions: List[str]) -> str:
        """Write GIF using Pillow."""
        images = []
        for arr in frames:
            # Resize to target dimensions
            img = Image.fromarray(arr)
            img = img.resize((self.width, self.height), Image.LANCZOS)
            images.append(img)

        images[0].save(
            self.output_path,
            save_all=True,
            append_images=images[1:],
            duration=self.duration_ms,
            loop=0,
        )
        return self.output_path

    def _write_fallback(self, frames: List[np.ndarray], captions: List[str]) -> str:
        """Fallback: save as individual PNG frames if PIL unavailable."""
        out_dir = os.path.splitext(self.output_path)[0] + '_frames'
        os.makedirs(out_dir, exist_ok=True)
        for i, arr in enumerate(frames):
            path = os.path.join(out_dir, f'frame_{i:04d}.npy')
            np.save(path, arr)
        # Write a manifest
        manifest = os.path.join(out_dir, 'manifest.json')
        with open(manifest, 'w') as f:
            json.dump({'frames': len(frames), 'captions': captions}, f, indent=2)
        return out_dir

    def reset(self) -> None:
        """Clear all captured frames."""
        self._frames.clear()
        self._captions.clear()
