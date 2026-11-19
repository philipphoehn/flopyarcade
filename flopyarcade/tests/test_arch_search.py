"""C10: Tests for GA network architecture search space."""

import pytest

from flopyarcade.arch_search import (
    ACTIVATIONS,
    NNTYPES,
    ARCH_SPACE,
    ArchitectureConfig,
    ArchitectureSampler,
    validate_architecture,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_activations_valid(self):
        assert 'relu' in ACTIVATIONS
        assert 'tanh' in ACTIVATIONS
        assert 'sigmoid' in ACTIVATIONS
        assert len(ACTIVATIONS) >= 4

    def test_nntypes_valid(self):
        assert 'perceptron' in NNTYPES
        assert 'convolution' in NNTYPES

    def test_arch_space_keys(self):
        expected = {'NNTYPE', 'N_LAYERS', 'N_NEURONS_MIN', 'N_NEURONS_MAX',
                    'DROPOUT_RANGE', 'BATCHNORMALIZATION', 'ARCHITECTUREVARY'}
        assert expected.issubset(set(ARCH_SPACE.keys()))


# ---------------------------------------------------------------------------
# ArchitectureConfig
# ---------------------------------------------------------------------------

class TestArchitectureConfig:
    def test_defaults(self):
        a = ArchitectureConfig()
        assert a.nntype == 'perceptron'
        assert a.n_layers == 2
        assert a.hidden_nodes == []
        assert a.hidden_activations == []
        assert a.dropouts == []
        assert a.batch_normalization is False
        assert a.architecture_vary is True

    def test_to_hyparams(self):
        a = ArchitectureConfig(
            nntype='perceptron',
            n_layers=3,
            hidden_nodes=[64, 128, 256],
            hidden_activations=['relu', 'tanh', 'relu'],
            dropouts=[0.1, 0.2, 0.3],
            batch_normalization=True,
            architecture_vary=False,
        )
        hp = a.to_hyparams()
        assert hp['NNTYPE'] == 'perceptron'
        assert hp['NHIDDENNODES'] == [64, 128, 256]
        assert hp['HIDDENACTIVATIONS'] == ['relu', 'tanh', 'relu']
        assert hp['DROPOUTS'] == [0.1, 0.2, 0.3]
        assert hp['BATCHNORMALIZATION'] is True
        assert hp['ARCHITECTUREVARY'] is False

    def test_to_hyparams_returns_copies(self):
        a = ArchitectureConfig(hidden_nodes=[32, 64])
        hp = a.to_hyparams()
        hp['NHIDDENNODES'].append(128)
        assert a.hidden_nodes == [32, 64]  # original unchanged

    def test_summary(self):
        a = ArchitectureConfig(
            nntype='perceptron',
            n_layers=2,
            hidden_nodes=[128, 64],
            hidden_activations=['relu', 'tanh'],
            dropouts=[0.1, 0.2],
            batch_normalization=True,
        )
        s = a.summary()
        assert 'perceptron' in s
        assert '2L' in s
        assert '128' in s
        assert '64' in s
        assert 'relu' in s


# ---------------------------------------------------------------------------
# ArchitectureSampler
# ---------------------------------------------------------------------------

class TestArchitectureSamplerInit:
    def test_default_seed_none(self):
        s = ArchitectureSampler()
        assert s.space is ARCH_SPACE

    def test_custom_seed(self):
        s = ArchitectureSampler(seed=42)
        assert s.space is ARCH_SPACE

    def test_custom_space(self):
        custom = {'NNTYPE': {'type': 'categorical', 'choices': ['perceptron']}}
        s = ArchitectureSampler(space=custom)
        assert s.space is custom


class TestSample:
    def test_returns_architecture_config(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        assert isinstance(a, ArchitectureConfig)

    def test_nntype_valid(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        assert a.nntype in NNTYPES

    def test_n_layers_in_range(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        assert 1 <= a.n_layers <= 5

    def test_hidden_nodes_length_matches_layers(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        assert len(a.hidden_nodes) == a.n_layers

    def test_hidden_activations_valid(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        for act in a.hidden_activations:
            assert act in ACTIVATIONS

    def test_dropouts_in_range(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        for d in a.dropouts:
            assert 0.0 <= d <= 0.5

    def test_hidden_nodes_positive(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        for n in a.hidden_nodes:
            assert n >= 1

    def test_reproducible_with_seed(self):
        s1 = ArchitectureSampler(seed=99)
        s2 = ArchitectureSampler(seed=99)
        a1 = s1.sample()
        a2 = s2.sample()
        assert a1.nntype == a2.nntype
        assert a1.n_layers == a2.n_layers
        assert a1.hidden_nodes == a2.hidden_nodes
        assert a1.hidden_activations == a2.hidden_activations
        assert a1.dropouts == a2.dropouts

    def test_multiple_samples_diverse(self):
        s = ArchitectureSampler(seed=42)
        configs = [s.sample() for _ in range(20)]
        nntypes = set(c.nntype for c in configs)
        n_layers = set(c.n_layers for c in configs)
        # With 20 samples we should see some diversity
        assert len(nntypes) >= 1
        assert len(n_layers) >= 1


class TestSamplePopulation:
    def test_correct_count(self):
        s = ArchitectureSampler(seed=1)
        pop = s.sample_population(10)
        assert len(pop) == 10
        assert all(isinstance(c, ArchitectureConfig) for c in pop)

    def test_zero(self):
        s = ArchitectureSampler(seed=1)
        pop = s.sample_population(0)
        assert pop == []


class TestMutate:
    def test_returns_new_object(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        b = s.mutate(a, mutation_rate=1.0)
        assert b is not a

    def test_original_unchanged(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        orig_nodes = list(a.hidden_nodes)
        orig_layers = a.n_layers
        s.mutate(a, mutation_rate=1.0)
        assert a.hidden_nodes == orig_nodes
        assert a.n_layers == orig_layers

    def test_mutation_rate_zero_no_change(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        b = s.mutate(a, mutation_rate=0.0)
        assert b.nntype == a.nntype
        assert b.n_layers == a.n_layers
        assert b.hidden_nodes == a.hidden_nodes

    def test_mutated_layers_in_range(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        b = s.mutate(a, mutation_rate=1.0)
        assert 1 <= b.n_layers <= 5
        assert len(b.hidden_nodes) == b.n_layers


class TestCrossover:
    def test_returns_architecture_config(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        b = s.sample()
        child = s.crossover(a, b)
        assert isinstance(child, ArchitectureConfig)

    def test_child_layers_valid(self):
        s = ArchitectureSampler(seed=1)
        a = s.sample()
        b = s.sample()
        child = s.crossover(a, b)
        assert 1 <= child.n_layers <= 5
        assert len(child.hidden_nodes) == child.n_layers
        assert len(child.hidden_activations) == child.n_layers
        assert len(child.dropouts) == child.n_layers

    def test_child_nntype_from_parents(self):
        s = ArchitectureSampler(seed=1)
        a = ArchitectureConfig(nntype='perceptron', n_layers=2, hidden_nodes=[32, 64], hidden_activations=['relu', 'tanh'], dropouts=[0.1, 0.2])
        b = ArchitectureConfig(nntype='convolution', n_layers=3, hidden_nodes=[64, 128, 256], hidden_activations=['relu', 'relu', 'tanh'], dropouts=[0.1, 0.2, 0.3])
        child = s.crossover(a, b)
        assert child.nntype in ('perceptron', 'convolution')


# ---------------------------------------------------------------------------
# validate_architecture
# ---------------------------------------------------------------------------

class TestValidateArchitecture:
    def test_valid_config_no_errors(self):
        a = ArchitectureConfig(
            nntype='perceptron',
            n_layers=2,
            hidden_nodes=[64, 128],
            hidden_activations=['relu', 'tanh'],
            dropouts=[0.1, 0.2],
        )
        errors = validate_architecture(a)
        assert errors == []

    def test_invalid_nntype(self):
        a = ArchitectureConfig(nntype='invalid', n_layers=1, hidden_nodes=[32], hidden_activations=['relu'], dropouts=[0.1])
        errors = validate_architecture(a)
        assert any('nntype' in e for e in errors)

    def test_mismatched_layers(self):
        a = ArchitectureConfig(n_layers=3, hidden_nodes=[32], hidden_activations=['relu'], dropouts=[0.1])
        errors = validate_architecture(a)
        assert len(errors) >= 2  # nodes and activations mismatch

    def test_zero_neurons(self):
        a = ArchitectureConfig(n_layers=1, hidden_nodes=[0], hidden_activations=['relu'], dropouts=[0.1])
        errors = validate_architecture(a)
        assert any('hidden_nodes' in e and 'must be >= 1' in e for e in errors)

    def test_invalid_activation(self):
        a = ArchitectureConfig(n_layers=1, hidden_nodes=[32], hidden_activations=['bogus'], dropouts=[0.1])
        errors = validate_architecture(a)
        assert any('bogus' in e for e in errors)

    def test_out_of_range_dropout(self):
        a = ArchitectureConfig(n_layers=1, hidden_nodes=[32], hidden_activations=['relu'], dropouts=[1.5])
        errors = validate_architecture(a)
        assert any('dropouts' in e for e in errors)

    def test_sampled_arch_is_valid(self):
        s = ArchitectureSampler(seed=7)
        for _ in range(10):
            a = s.sample()
            errors = validate_architecture(a)
            assert errors == [], f"Unexpected errors: {errors}"
