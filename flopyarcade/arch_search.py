"""C10: Network architecture search space for FloPyArcade GA.

Extends the GA hyperparameter search to include neural network
architecture parameters (layers, neurons, activations, dropout,
batch norm) so the genetic algorithm can evolve topology, not just
weights.

Usage::

    from flopyarcade.arch_search import ArchitectureSampler, ARCH_SPACE

    sampler = ArchitectureSampler(seed=42)
    arch = sampler.sample()  # dict with NHIDDENNODES, HIDDENACTIVATIONS, etc.

    # Use arch dict directly in hyParams:
    # agent.hyParams.update(arch)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

#: Activation functions available for hidden layers.
ACTIVATIONS = ('relu', 'tanh', 'sigmoid', 'elu', 'softplus', 'leaky_relu')

#: Network types the architecture sampler can produce.
NNTYPES = ('perceptron', 'convolution')

#: Architecture search space specification.
#: Each key maps to a dict describing how to sample that parameter.
ARCH_SPACE: Dict[str, Dict[str, Any]] = {
    'NNTYPE': {
        'type': 'categorical',
        'choices': NNTYPES,
        'description': 'Network type: fully-connected or convolutional',
    },
    'N_LAYERS': {
        'type': 'int',
        'low': 1,
        'high': 5,
        'step': 1,
        'description': 'Number of hidden layers',
    },
    'N_NEURONS_MIN': {
        'type': 'int',
        'low': 16,
        'high': 64,
        'step': 8,
        'description': 'Minimum neurons per hidden layer',
    },
    'N_NEURONS_MAX': {
        'type': 'int',
        'low': 128,
        'high': 512,
        'step': 64,
        'description': 'Maximum neurons per hidden layer',
    },
    'DROPOUT_RANGE': {
        'type': 'tuple',
        'low': 0.0,
        'high': 0.5,
        'description': 'Random dropout per layer drawn uniformly in [low, high]',
    },
    'BATCHNORMALIZATION': {
        'type': 'bool',
        'description': 'Whether to use batch normalization',
    },
    'ARCHITECTUREVARY': {
        'type': 'bool',
        'description': 'Whether GA is allowed to vary architecture per agent',
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureConfig:
    """A concrete sampled architecture.

    Attributes:
        nntype: 'perceptron' or 'convolution'.
        n_layers: Number of hidden layers (1-5).
        hidden_nodes: List of neuron counts per hidden layer.
        hidden_activations: List of activation names per hidden layer.
        dropouts: List of dropout rates per hidden layer.
        batch_normalization: Whether to use batch norm.
        architecture_vary: Whether GA can vary arch per agent.
    """
    nntype: str = 'perceptron'
    n_layers: int = 2
    hidden_nodes: List[int] = field(default_factory=list)
    hidden_activations: List[str] = field(default_factory=list)
    dropouts: List[float] = field(default_factory=list)
    batch_normalization: bool = False
    architecture_vary: bool = True

    def to_hyparams(self) -> Dict[str, Any]:
        """Convert to the hyParams dict format used by FloPyArcade agents.

        Returns:
            Dict with keys matching agent.py expectations:
            NNTYPE, NHIDDENNODES, HIDDENACTIVATIONS, DROPOUTS,
            BATCHNORMALIZATION, ARCHITECTUREVARY.
        """
        return {
            'NNTYPE': self.nntype,
            'NHIDDENNODES': list(self.hidden_nodes),
            'HIDDENACTIVATIONS': list(self.hidden_activations),
            'DROPOUTS': list(self.dropouts),
            'BATCHNORMALIZATION': self.batch_normalization,
            'ARCHITECTUREVARY': self.architecture_vary,
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        nodes_str = ' -> '.join(str(n) for n in self.hidden_nodes)
        return (
            f"{self.nntype} | {self.n_layers}L [{nodes_str}] "
            f"act={self.hidden_activations} "
            f"drop={self.dropouts} BN={self.batch_normalization}"
        )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class ArchitectureSampler:
    """Sample random architectures from the defined search space.

    Args:
        seed: Optional random seed for reproducibility.
        space: Custom architecture space (defaults to ARCH_SPACE).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        space: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._rng = random.Random(seed)
        self.space = space if space is not None else ARCH_SPACE

    def sample(self) -> ArchitectureConfig:
        """Sample a single random architecture.

        Returns:
            ArchitectureConfig with all parameters filled in.
        """
        # NNTYPE
        nntype = self._rng.choice(self.space['NNTYPE']['choices'])

        # N_LAYERS
        n_layers = self._rng.randint(
            self.space['N_LAYERS']['low'],
            self.space['N_LAYERS']['high'],
        )

        # Per-layer neurons: random between min and max, monotone decreasing
        n_min = self.space['N_NEURONS_MIN']['low']
        n_max = self.space['N_NEURONS_MAX']['high']
        hidden_nodes: List[int] = []
        for i in range(n_layers):
            frac = 1.0 - (i / max(n_layers - 1, 1))
            lo = int(n_min + (n_max - n_min) * 0.2 * (1 - frac))
            hi = int(n_min + (n_max - n_min) * (0.5 + 0.5 * frac))
            lo = max(lo, self.space['N_NEURONS_MIN']['low'])
            hi = min(hi, self.space['N_NEURONS_MAX']['high'])
            hi = max(hi, lo)
            hidden_nodes.append(self._rng.randint(lo, hi))

        # Activations per layer
        choices = self.space['NNTYPE']['choices']  # reuse for activation list
        act_choices = ACTIVATIONS
        hidden_activations = [self._rng.choice(act_choices) for _ in range(n_layers)]

        # Dropout per layer
        d_lo = self.space['DROPOUT_RANGE']['low']
        d_hi = self.space['DROPOUT_RANGE']['high']
        dropouts = [round(self._rng.uniform(d_lo, d_hi), 3) for _ in range(n_layers)]

        # Batch norm
        batch_norm = self._rng.random() < 0.5

        # Architecture vary
        arch_vary = self._rng.random() < 0.8

        return ArchitectureConfig(
            nntype=nntype,
            n_layers=n_layers,
            hidden_nodes=hidden_nodes,
            hidden_activations=hidden_activations,
            dropouts=dropouts,
            batch_normalization=batch_norm,
            architecture_vary=arch_vary,
        )

    def sample_population(
        self, n: int, **kwargs: Any
    ) -> List[ArchitectureConfig]:
        """Sample a population of architectures for GA initialisation.

        Args:
            n: Number of architectures to generate.
            **kwargs: Passed to individual sample calls (unused currently).

        Returns:
            List of ArchitectureConfig objects.
        """
        return [self.sample() for _ in range(n)]

    def mutate(
        self, arch: ArchitectureConfig, mutation_rate: float = 0.3
    ) -> ArchitectureConfig:
        """Apply random mutations to an existing architecture.

        Args:
            arch: The architecture to mutate.
            mutation_rate: Probability of mutating each parameter.

        Returns:
            New ArchitectureConfig (original is not modified).
        """
        import copy
        new = copy.deepcopy(arch)

        if self._rng.random() < mutation_rate:
            new.nntype = self._rng.choice(self.space['NNTYPE']['choices'])

        if self._rng.random() < mutation_rate:
            delta = self._rng.choice([-1, 1])
            new.n_layers = max(
                1,
                min(
                    self.space['N_LAYERS']['high'],
                    new.n_layers + delta,
                ),
            )
            # Adjust layer lists
            while len(new.hidden_nodes) < new.n_layers:
                new.hidden_nodes.append(self._rng.randint(16, 256))
                new.hidden_activations.append(self._rng.choice(ACTIVATIONS))
                new.dropouts.append(round(self._rng.uniform(0.0, 0.3), 3))
            while len(new.hidden_nodes) > new.n_layers:
                new.hidden_nodes.pop()
                new.hidden_activations.pop()
                new.dropouts.pop()

        if self._rng.random() < mutation_rate:
            idx = self._rng.randrange(new.n_layers)
            new.hidden_nodes[idx] = self._rng.randint(16, 512)

        if self._rng.random() < mutation_rate:
            idx = self._rng.randrange(new.n_layers)
            new.hidden_activations[idx] = self._rng.choice(ACTIVATIONS)

        if self._rng.random() < mutation_rate:
            idx = self._rng.randrange(new.n_layers)
            new.dropouts[idx] = round(self._rng.uniform(0.0, 0.5), 3)

        if self._rng.random() < mutation_rate:
            new.batch_normalization = not new.batch_normalization

        return new

    def crossover(
        self, a: ArchitectureConfig, b: ArchitectureConfig
    ) -> ArchitectureConfig:
        """Uniform crossover between two architectures.

        For each parameter, randomly pick from parent A or parent B.

        Args:
            a: First parent.
            b: Second parent.

        Returns:
            Child ArchitectureConfig.
        """
        import copy
        n_layers = self._rng.choice([a.n_layers, b.n_layers])
        child = copy.deepcopy(a)
        child.n_layers = n_layers

        # Pick neurons per layer from either parent
        child.hidden_nodes = []
        child.hidden_activations = []
        child.dropouts = []
        for i in range(n_layers):
            if self._rng.random() < 0.5:
                src = a if i < len(a.hidden_nodes) else b
            else:
                src = b if i < len(b.hidden_nodes) else a
            if i < len(src.hidden_nodes):
                child.hidden_nodes.append(src.hidden_nodes[i])
                child.hidden_activations.append(src.hidden_activations[i])
                child.dropouts.append(src.dropouts[i])
            else:
                child.hidden_nodes.append(64)
                child.hidden_activations.append('relu')
                child.dropouts.append(0.0)

        child.nntype = self._rng.choice([a.nntype, b.nntype])
        child.batch_normalization = self._rng.choice([a.batch_normalization, b.batch_normalization])
        child.architecture_vary = self._rng.choice([a.architecture_vary, b.architecture_vary])
        return child


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_architecture(arch: ArchitectureConfig) -> List[str]:
    """Validate an architecture config for internal consistency.

    Args:
        arch: Architecture to validate.

    Returns:
        List of error strings (empty if valid).
    """
    errors: List[str] = []

    if arch.n_layers < 1:
        errors.append(f"n_layers must be >= 1, got {arch.n_layers}")

    if len(arch.hidden_nodes) != arch.n_layers:
        errors.append(
            f"hidden_nodes length ({len(arch.hidden_nodes)}) "
            f"!= n_layers ({arch.n_layers})"
        )

    if len(arch.hidden_activations) != arch.n_layers:
        errors.append(
            f"hidden_activations length ({len(arch.hidden_activations)}) "
            f"!= n_layers ({arch.n_layers})"
        )

    if len(arch.dropouts) != arch.n_layers:
        errors.append(
            f"dropouts length ({len(arch.dropouts)}) != n_layers ({arch.n_layers})"
        )

    for i, n in enumerate(arch.hidden_nodes):
        if n < 1:
            errors.append(f"hidden_nodes[{i}] = {n}, must be >= 1")

    for i, act in enumerate(arch.hidden_activations):
        if act not in ACTIVATIONS:
            errors.append(f"hidden_activations[{i}] = '{act}', not in {ACTIVATIONS}")

    for i, d in enumerate(arch.dropouts):
        if not (0.0 <= d <= 1.0):
            errors.append(f"dropouts[{i}] = {d}, must be in [0, 1]")

    if arch.nntype not in NNTYPES:
        errors.append(f"nntype = '{arch.nntype}', not in {NNTYPES}")

    return errors
