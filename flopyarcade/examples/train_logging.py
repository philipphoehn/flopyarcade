"""Central logging utility for FloPyArcade training runs.

Writes mean rewards per generation to a JSONL log file and optionally
saves the best agent episode as a GIF (B16 integration).

Default output location: ~/flopyarcade/results/
Override with FLOPYARCADE_RESULTS_DIR env var or --output-dir argument.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(os.environ.get(
    "FLOPYARCADE_RESULTS_DIR",
    str(Path.home() / "flopyarcade" / "results")
))


def get_output_dir(output_dir: Optional[str] = None) -> Path:
    """Resolve the output directory (user-definable)."""
    if output_dir:
        p = Path(output_dir).expanduser().resolve()
    else:
        p = DEFAULT_OUTPUT_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


class TrainingLogger:
    """Logs mean rewards per generation and tracks best agent."""

    def __init__(self, envtype: str = "1s-d",
                 output_dir: Optional[str] = None,
                 save_gif: bool = True,
                 gif_fps: int = 10):
        self.envtype = envtype
        self.output_dir = get_output_dir(output_dir)
        self.save_gif = save_gif
        self.gif_fps = gif_fps
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.output_dir / f"{envtype}_{self.timestamp}.jsonl"
        self.gif_path = self.output_dir / f"{envtype}_{self.timestamp}_best.gif"
        self.best_reward = float("-inf")
        self.best_generation = 0
        self.generations: List[Dict] = []
        logger.info(f"TrainingLogger: logging to {self.log_path}")

    def log_generation(self, gen: int, mean_reward: float,
                       best_reward: float, extra: Optional[Dict] = None) -> None:
        """Log a single generation's results to JSONL."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "generation": gen,
            "mean_reward": round(mean_reward, 4),
            "best_reward": round(best_reward, 4),
            "envtype": self.envtype,
        }
        if extra:
            entry.update(extra)
        self.generations.append(entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if best_reward > self.best_reward:
            self.best_reward = best_reward
            self.best_generation = gen
            logger.info(f"Gen {gen}: NEW BEST reward={best_reward:.4f}")
        logger.info(f"Gen {gen}: mean={mean_reward:.4f}, best={best_reward:.4f}")

    def save_best_gif(self, frames: List[Any]) -> Optional[Path]:
        """Save best agent episode frames as GIF (B16 integration)."""
        if not self.save_gif or not frames:
            return None
        try:
            import imageio
            import numpy as np
            rgb_frames = []
            for f in frames:
                if isinstance(f, np.ndarray):
                    rgb_frames.append(f.astype(np.uint8))
            if rgb_frames:
                imageio.mimsave(str(self.gif_path), rgb_frames, fps=self.gif_fps)
                logger.info(f"GIF saved: {self.gif_path}")
                return self.gif_path
        except Exception as e:
            logger.warning(f"Failed to save GIF: {e}")
        return None

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the training run."""
        if not self.generations:
            return {"status": "empty"}
        rewards = [g["best_reward"] for g in self.generations]
        return {
            "envtype": self.envtype,
            "n_generations": len(self.generations),
            "best_reward": max(rewards),
            "best_generation": self.best_generation + 1,
            "final_mean_reward": self.generations[-1]["mean_reward"],
            "log_path": str(self.log_path),
            "gif_path": str(self.gif_path) if self.save_gif else None,
        }


def wrap_training_loop(train_fn, envtype: str, n_generations: int,
                       output_dir: Optional[str] = None,
                       save_gif: bool = True, **kwargs) -> Dict[str, Any]:
    """Wrap a training function with central logging.

    train_fn signature: (gen: int, **kwargs) -> (mean_reward, best_reward, frames)
    """
    tl = TrainingLogger(envtype=envtype, output_dir=output_dir, save_gif=save_gif)
    all_frames = []
    for gen in range(n_generations):
        mean_r, best_r, frames = train_fn(gen, **kwargs)
        tl.log_generation(gen, mean_r, best_r)
        if frames and best_r >= tl.best_reward - 1e-6:
            all_frames = frames
    if save_gif and all_frames:
        tl.save_best_gif(all_frames)
    return tl.summary()


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="FloPyArcade training logger demo")
    parser.add_argument("--envtype", default="1s-d")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    tl = TrainingLogger(envtype=args.envtype, output_dir=args.output_dir)
    np.random.seed(42)
    for gen in range(args.generations):
        mean_r = 100 + gen * 10 + np.random.randn() * 5
        best_r = mean_r + 20
        tl.log_generation(gen, mean_r, best_r)
    print(json.dumps(tl.summary(), indent=2))
