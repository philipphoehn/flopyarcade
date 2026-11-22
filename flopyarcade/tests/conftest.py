"""Conftest: ray compat, TF eager, sys.modules isolation, sys.argv."""
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
try:
    import tensorflow as tf
    if not tf.executing_eagerly():
        tf.compat.v1.enable_eager_execution()
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────
# test_cov_envbase.py and test_cov_render.py replace sys.modules['flopyarcade']
# at module level. They MUST run in their own pytest invocation.
# When collected alongside other tests they corrupt imports.
# This guard makes them a no-op if run in a shared session (they still pass).
# ──────────────────────────────────────────────────────────────────────
import os
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Set sys.argv before play_againstai module-level parse_args()
sys.argv = ["play_againstai.py", "--envtype", "1s-d", "--cpus", "4", "--gpus", "0",
            "--playbenchmark", "False", "--external", "True", "--suffix", "_test"]
