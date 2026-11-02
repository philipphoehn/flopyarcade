"""Targeted tests to push coverage above 70%."""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFloPyEnvMethods:
    @pytest.fixture
    def env(self):
        from flopyarcade.flopyarcade import FloPyEnv
        e = FloPyEnv(ENVTYPE='1s-d', flagRender=False)
        yield e
        e.close()

    def test_seed_with_value(self, env):
        env.seed(99)
        assert env._SEED == 99

    def test_seed_none(self, env):
        env.seed(None)

    def test_env_str(self, env):
        assert isinstance(str(env), str)

    def test_env_repr(self, env):
        assert isinstance(repr(env), str)

    def test_info_dict(self, env):
        assert hasattr(env, 'info')

    def test_observations_vector(self, env):
        obs = env.reset(seed=42)
        assert env.observationsVectorNormalized is not None

    def test_invalid_envtype_raises(self):
        from flopyarcade.flopyarcade import FloPyEnv
        with pytest.raises(Exception):
            e = FloPyEnv(ENVTYPE='nonexistent_xyz', flagRender=False)
            e.close()

    def test_env_type_attribute(self, env):
        assert env.ENVTYPE == '1s-d'

    def test_reset_with_seed(self, env):
        assert env.reset(seed=123) is not None

    def test_step_and_obs(self, env):
        obs = env.reset(seed=42)
        obs2, reward, done, truncated, info = env.step('up')
        assert obs2 is not None
        assert np.isscalar(reward) or hasattr(reward, 'item')


def _extract_str_to_bool():
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'play_againstai.py')
    with open(src_path) as f:
        src = f.read()
    lines = src.split('\n')
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def str_to_bool'):
            start = i
            break
    if start is None:
        raise RuntimeError('str_to_bool not found')
    end = start + 1
    while end < len(lines):
        if lines[end].strip() == '':
            if end + 1 >= len(lines) or not lines[end + 1].startswith((' ', '\t')):
                break
        elif not lines[end].startswith((' ', '\t')):
            break
        end += 1
    func_src = '\n'.join(lines[start:end])
    ns = {}
    exec(func_src, ns)
    return ns['str_to_bool']


class TestStrToBool:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.fn = _extract_str_to_bool()

    def test_valid_true(self):
        assert self.fn(True) is True
        assert self.fn('true') is True
        assert self.fn('True') is True
        assert self.fn('t') is True
        assert self.fn('1') is True
        assert self.fn('yes') is True
        assert self.fn('y') is True

    def test_valid_false(self):
        assert self.fn(False) is False
        assert self.fn('false') is False
        assert self.fn('False') is False
        assert self.fn('f') is False
        assert self.fn('0') is False
        assert self.fn('no') is False
        assert self.fn('n') is False

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            self.fn('maybe')
        with pytest.raises(ValueError):
            self.fn('2')
