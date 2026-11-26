"""R10: Meta-learning (MAML) on reward weights.

Outer loop adapts per env type in < 10 episodes.
Inner loop: task-specific weight adaptation via proxy loss.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import ClassVar


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

ENV_TYPES: list[str] = ["confined", "unconfined", "leaky", "fault", "river"]


@dataclass
class TaskSpec:
    """A single meta-learning task specification."""

    envtype: str
    reward_weights: list[float]
    n_episodes: int = 5
    inner_lr: float = 0.01
    outer_lr: float = 0.001


@dataclass
class MetaParams:
    """Meta-learnable parameters (base reward weights)."""

    theta: list[float]  # base reward weights


# ---------------------------------------------------------------------------
# MAML Reward Trainer
# ---------------------------------------------------------------------------

class MAMLRewardTrainer:
    """Meta-learning trainer for reward weight adaptation.

    Simulates MAML inner/outer loops using a proxy loss:
        loss = sum((w_i - target_i)^2)
    """

    def __init__(
        self,
        n_params: int = 4,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 5,
        n_tasks: int = 10,
        seed: int | None = None,
    ) -> None:
        self.n_params = n_params
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps
        self.n_tasks = n_tasks
        self.seed = seed
        self.rng = random.Random(seed)
        # Initialize theta to uniform random in [0, 1]
        self.theta = [self.rng.uniform(0.0, 1.0) for _ in range(n_params)]

    # -- task sampling -------------------------------------------------------

    def sample_task(self) -> TaskSpec:
        """Generate a random task with envtype and target reward weights."""
        envtype = self.rng.choice(ENV_TYPES)
        reward_weights = [self.rng.uniform(0.0, 1.0) for _ in range(self.n_params)]
        return TaskSpec(
            envtype=envtype,
            reward_weights=reward_weights,
            n_episodes=self.n_inner_steps,
            inner_lr=self.inner_lr,
            outer_lr=self.outer_lr,
        )

    # -- inner loop ----------------------------------------------------------

    def inner_loop(self, theta: list[float], task: TaskSpec) -> list[float]:
        """Simulate MAML inner adaptation on a single task.

        Proxy loss: L = sum((w_i - target_i)^2)
        Gradient:   dL/dw = 2*(w_i - target_i)
        Update:     theta_prime = theta - inner_lr * grad
        """
        target = task.reward_weights
        # Compute gradient of proxy loss w.r.t. theta
        grad = [2.0 * (theta[i] - target[i]) for i in range(len(theta))]
        # Apply inner-loop update
        theta_prime = [theta[i] - task.inner_lr * grad[i] for i in range(len(theta))]
        return theta_prime

    # -- outer loop ----------------------------------------------------------

    def outer_loop(self, theta: list[float], tasks: list[TaskSpec]) -> list[float]:
        """Compute meta-gradient and update base parameters.

        meta_grad = avg(theta - theta_prime) across tasks (error signal)
        theta_new = theta - outer_lr * meta_grad
        """
        n = len(theta)
        meta_grad = [0.0] * n
        for task in tasks:
            theta_prime = self.inner_loop(theta, task)
            for i in range(n):
                meta_grad[i] += (theta[i] - theta_prime[i]) / len(tasks)
        theta_new = [theta[i] - self.outer_lr * meta_grad[i] for i in range(n)]
        return theta_new

    # -- training ------------------------------------------------------------

    def train(self, n_outer_steps: int = 100) -> dict:
        """Run full MAML training loop.

        Returns dict with theta_history, loss_history, n_steps.
        """
        theta_history: list[list[float]] = [list(self.theta)]
        loss_history: list[float] = []

        for step in range(n_outer_steps):
            # Sample tasks
            tasks = [self.sample_task() for _ in range(self.n_tasks)]

            # Compute loss before update (proxy: avg distance to task targets)
            total_loss = 0.0
            for task in tasks:
                for i in range(len(self.theta)):
                    total_loss += (self.theta[i] - task.reward_weights[i]) ** 2
            loss_history.append(total_loss / len(tasks))

            # Outer loop update
            self.theta = self.outer_loop(self.theta, tasks)
            theta_history.append(list(self.theta))

        return {
            "theta_history": theta_history,
            "loss_history": loss_history,
            "n_steps": n_outer_steps,
        }

    # -- adaptation ------------------------------------------------------------

    def adapt(self, theta: list[float], task: TaskSpec) -> list[float]:
        """Single inner-loop adaptation for a new task."""
        return self.inner_loop(theta, task)

    # -- evaluation ------------------------------------------------------------

    def evaluate_adaptation(
        self, theta: list[float], n_tasks: int = 20, threshold: float = 0.1
    ) -> dict:
        """Evaluate how quickly inner adaptation converges on new tasks.

        Proxy: count inner steps until loss < threshold.
        Returns {mean_episodes_to_converge, std, best_episode, worst_episode}.
        """
        episodes: list[int] = []

        for _ in range(n_tasks):
            task = self.sample_task()
            target = task.reward_weights
            current = list(theta)
            converged_at = -1
            for step in range(1, 50):
                loss = sum((current[i] - target[i]) ** 2 for i in range(len(current)))
                if loss < threshold:
                    converged_at = step
                    break
                current = self.inner_loop(current, task)
            # Check final
            if converged_at == -1:
                loss = sum((current[i] - target[i]) ** 2 for i in range(len(current)))
                if loss < threshold:
                    converged_at = 50
            episodes.append(converged_at)

        valid = [e for e in episodes if e >= 0]
        if not valid:
            return {
                "mean_episodes_to_converge": -1.0,
                "std": 0.0,
                "best_episode": -1,
                "worst_episode": -1,
            }

        mean = sum(valid) / len(valid)
        variance = sum((e - mean) ** 2 for e in valid) / len(valid)
        std = variance ** 0.5

        return {
            "mean_episodes_to_converge": mean,
            "std": std,
            "best_episode": min(valid),
            "worst_episode": max(valid),
        }

    # -- persistence -----------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        """Save trainer state to JSON."""
        state = {
            "n_params": self.n_params,
            "inner_lr": self.inner_lr,
            "outer_lr": self.outer_lr,
            "n_inner_steps": self.n_inner_steps,
            "n_tasks": self.n_tasks,
            "seed": self.seed,
            "theta": self.theta,
        }
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load_state(cls, path: str | Path) -> MAMLRewardTrainer:
        """Load trainer state from JSON."""
        data = json.loads(Path(path).read_text())
        trainer = cls(
            n_params=data["n_params"],
            inner_lr=data["inner_lr"],
            outer_lr=data["outer_lr"],
            n_inner_steps=data["n_inner_steps"],
            n_tasks=data["n_tasks"],
            seed=data["seed"],
        )
        trainer.theta = data["theta"]
        return trainer


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def convergence_schedule(losses: list[float], threshold: float = 0.1) -> int:
    """Return first index where loss < threshold, or -1 if never reached."""
    for i, loss in enumerate(losses):
        if loss < threshold:
            return i
    return -1


def format_maml_report(trainer: MAMLRewardTrainer, train_result: dict) -> str:
    """Format a human-readable MAML training report."""
    loss_hist = train_result.get("loss_history", [])
    n_steps = train_result.get("n_steps", 0)
    initial_loss = loss_hist[0] if loss_hist else float("nan")
    final_loss = loss_hist[-1] if loss_hist else float("nan")
    improvement = "N/A" if initial_loss == 0 else f"{(1 - final_loss / initial_loss) * 100:.1f}%"

    lines = [
        "=" * 60,
        "  MAML Reward Weight Training Report",
        "=" * 60,
        f"  Parameters:      {trainer.n_params}",
        f"  Inner LR:        {trainer.inner_lr}",
        f"  Outer LR:        {trainer.outer_lr}",
        f"  Inner steps:     {trainer.n_inner_steps}",
        f"  Tasks/step:      {trainer.n_tasks}",
        f"  Outer steps:     {n_steps}",
        "-" * 60,
        f"  Initial loss:    {initial_loss:.6f}",
        f"  Final loss:      {final_loss:.6f}",
        f"  Improvement:     {improvement}",
        "-" * 60,
        f"  Final theta:     {['%.4f' % t for t in trainer.theta]}",
        "=" * 60,
    ]
    return "\n".join(lines)
