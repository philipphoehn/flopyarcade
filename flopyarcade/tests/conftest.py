"""Conftest to handle ray 2.56.1 missing ray.rllib.agents.dqn submodule."""
import sys
from unittest.mock import MagicMock

# ray >= 2.43 removed ray.rllib.agents from public API
try:
    import ray.rllib
    if not hasattr(ray.rllib, 'agents'):
        _agents_mod = MagicMock()
        _agents_mod.dqn = MagicMock()
        _agents_mod.dqn.ApexTrainer = type("ApexTrainer", (), {})
        ray.rllib.agents = _agents_mod
        sys.modules['ray.rllib.agents'] = _agents_mod
        sys.modules['ray.rllib.agents.dqn'] = _agents_mod.dqn
except ImportError:
    pass

# ray.rllib disables TF eager execution; re-enable it for Keras/TF tests
import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# test_cov_envbase.py and test_cov_render.py both replace sys.modules['flopyarcade']
# at module level during collection, corrupting imports for all other test files.
collect_ignore_glob = ['test_cov_envbase.py', 'test_cov_render.py']

# Set sys.argv before play_againstai module-level parse_args()
sys.argv = ["play_againstai.py", "--envtype", "1s-d", "--cpus", "4", "--gpus", "0",
            "--playbenchmark", "False", "--external", "True", "--suffix", "_test"]
