"""Coverage tests for train_rllib_* example modules."""
import sys
import types
from unittest.mock import MagicMock, patch
import pytest

# Ensure mocks are in place before importing
from unittest.mock import MagicMock as _MC

# We need to import the modules with ray/tensorflow mocked
# The conftest already handles sys.argv and basic mocks

def _import_train_module(name):
    """Import a train_rllib module with heavy deps mocked."""
    mod_path = f'flopyarcade.examples.{name}'
    if mod_path in sys.modules:
        return sys.modules[mod_path]
    
    # Ensure ray mocks
    ray_mock = MagicMock()
    ray_mock.tune = MagicMock()
    ray_mock.tune.registry = MagicMock()
    ray_mock.tune.registry.register_env = MagicMock()
    ray_mock.tune.train = MagicMock()
    ray_mock.tune.Checkpoint = MagicMock()
    ray_mock.init = MagicMock()
    ray_mock.rllib = MagicMock()
    ray_mock.rllib.algorithms = MagicMock()
    ray_mock.rllib.algorithms.dqn = MagicMock()
    ray_mock.rllib.algorithms.dqn.ApexTrainer = type("ApexTrainer", (), {})
    ray_mock.rllib.algorithms.appo = MagicMock()
    ray_mock.rllib.algorithms.appo.APPOTrain = type("APPOTrain", (), {})
    ray_mock.rllib.algorithms.impala = MagicMock()
    ray_mock.rllib.algorithms.impala.ImpalaTrainer = type("ImpalaTrainer", (), {})
    ray_mock.rllib.algorithms.ppo = MagicMock()
    ray_mock.rllib.algorithms.ppo.PPOTrainer = type("PPOTrainer", (), {})
    
    sys.modules['ray'] = ray_mock
    sys.modules['ray.tune'] = ray_mock.tune
    sys.modules['ray.tune.registry'] = ray_mock.tune.registry
    sys.modules['ray.rllib'] = ray_mock.rllib
    sys.modules['ray.rllib.algorithms'] = ray_mock.rllib.algorithms
    sys.modules['ray.rllib.algorithms.dqn'] = ray_mock.rllib.algorithms.dqn
    sys.modules['ray.rllib.algorithms.appo'] = ray_mock.rllib.algorithms.appo
    sys.modules['ray.rllib.algorithms.impala'] = ray_mock.rllib.algorithms.impala
    sys.modules['ray.rllib.algorithms.ppo'] = ray_mock.rllib.algorithms.ppo
    
    # TF mocks
    if 'tensorflow' not in sys.modules or isinstance(sys.modules.get('tensorflow'), _MC):
        tf_mock = MagicMock()
        sys.modules['tensorflow'] = tf_mock
        sys.modules['tensorflow.keras'] = tf_mock.keras
        sys.modules['tensorflow.keras.initializers'] = tf_mock.keras.initializers
        sys.modules['tensorflow.keras.layers'] = tf_mock.keras.layers
        sys.modules['tensorflow.keras.models'] = tf_mock.keras.models
        sys.modules['tensorflow.keras.optimizers'] = tf_mock.keras.optimizers
    
    # gymnasium mock
    if 'gymnasium' not in sys.modules or isinstance(sys.modules.get('gymnasium'), _MC):
        sys.modules['gymnasium'] = MagicMock()
    
    # Use importlib to import
    import importlib
    mod = importlib.import_module(mod_path)
    return mod


class TestStrToBool:
    """Test str_to_bool across all train_rllib modules."""
    
    @pytest.mark.parametrize("name", [
        "train_rllib_apexdqn", "train_rllib_appo",
        "train_rllib_impala", "train_rllib_ppo"
    ])
    def test_true_values(self, name):
        mod = _import_train_module(name)
        assert mod.str_to_bool("true") is True
        assert mod.str_to_bool("t") is True
        assert mod.str_to_bool("1") is True
        assert mod.str_to_bool("yes") is True
        assert mod.str_to_bool("y") is True
    
    @pytest.mark.parametrize("name", [
        "train_rllib_apexdqn", "train_rllib_appo",
        "train_rllib_impala", "train_rllib_ppo"
    ])
    def test_false_values(self, name):
        mod = _import_train_module(name)
        assert mod.str_to_bool("false") is False
        assert mod.str_to_bool("f") is False
        assert mod.str_to_bool("0") is False
        assert mod.str_to_bool("no") is False
        assert mod.str_to_bool("n") is False
    
    @pytest.mark.parametrize("name", [
        "train_rllib_apexdqn", "train_rllib_appo",
        "train_rllib_impala", "train_rllib_ppo"
    ])
    def test_bool_passthrough(self, name):
        mod = _import_train_module(name)
        assert mod.str_to_bool(True) is True
        assert mod.str_to_bool(False) is False
    
    @pytest.mark.parametrize("name", [
        "train_rllib_apexdqn", "train_rllib_appo",
        "train_rllib_impala", "train_rllib_ppo"
    ])
    def test_invalid_raises(self, name):
        mod = _import_train_module(name)
        with pytest.raises(ValueError):
            mod.str_to_bool("invalid")


class TestEnvCreator:
    """Test env_creator function in train_rllib modules."""
    
    @pytest.mark.parametrize("name", [
        "train_rllib_apexdqn", "train_rllib_appo",
        "train_rllib_impala", "train_rllib_ppo"
    ])
    def test_env_creator(self, name):
        mod = _import_train_module(name)
        with patch.object(mod, 'FloPyEnv', return_value=MagicMock()) as mock_env:
            env = mod.env_creator(env_config={'ENVTYPE': '1s-d'})
            assert mock_env.called


class TestTrainModuleConfigs:
    """Verify config structures exist and are valid."""
    
    @pytest.mark.parametrize("name", [
        "train_rllib_apexdqn", "train_rllib_appo",
        "train_rllib_impala", "train_rllib_ppo"
    ])
    def test_config_exists(self, name):
        mod = _import_train_module(name)
        assert hasattr(mod, 'config')
        assert isinstance(mod.config, dict)
    
    @pytest.mark.parametrize("name", [
        "train_rllib_apexdqn", "train_rllib_appo",
        "train_rllib_impala", "train_rllib_ppo"
    ])
    def test_stop_criteria(self, name):
        mod = _import_train_module(name)
        assert hasattr(mod, 'config_stopCriteria')
        assert 'training_iteration' in mod.config_stopCriteria


class TestTrainDQN:
    """Test train_dqn module."""
    
    def test_str_to_bool(self):
        from flopyarcade.examples import train_dqn
        assert train_dqn.str_to_bool("true") is True
        assert train_dqn.str_to_bool("false") is False
        assert train_dqn.str_to_bool("1") is True
        assert train_dqn.str_to_bool("0") is False
        assert train_dqn.str_to_bool(True) is True
        with pytest.raises(ValueError):
            train_dqn.str_to_bool("xyz")


class TestTrainNeuroevolution:
    """Test train_neuroevolution module."""
    
    def test_str_to_bool(self):
        from flopyarcade.examples import train_neuroevolution
        assert train_neuroevolution.str_to_bool("true") is True
        assert train_neuroevolution.str_to_bool("false") is False
        assert train_neuroevolution.str_to_bool(True) is True
