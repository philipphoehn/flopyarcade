"""Conftest to handle ray 2.56.1 missing ray.rllib.agents.dqn submodule."""
import sys
from unittest.mock import MagicMock

# ray >= 2.43 removed ray.rllib.agents from public API
# play_againstai.py does: from ray.rllib.agents.dqn import ApexTrainer
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
    # ray not installed (CI without [ray] extra) — test_mock_rllib.py
    # injects its own mocks, so no patching needed here.
    pass

# Set sys.argv before play_againstai module-level parse_args()
sys.argv = ["play_againstai.py", "--envtype", "1s-d", "--cpus", "4", "--gpus", "0",
            "--playbenchmark", "False", "--external", "True", "--suffix", "_test"]
