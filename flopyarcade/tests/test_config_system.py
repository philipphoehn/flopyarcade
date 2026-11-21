#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Tests for flopyarcade.config_system (T2: Hydra-style config)."""

import json
import os
import tempfile

import pytest

from flopyarcade.config_system import (
    ConfigSchema,
    ConfigLoader,
    config_diff,
    format_config_report,
    generate_default_config,
    generate_env_config,
)


# ---------------------------------------------------------------------------
# 1. ConfigSchema defaults
# ---------------------------------------------------------------------------

class TestConfigSchemaDefaults:
    def test_defaults(self):
        cfg = ConfigSchema()
        assert cfg.envtype == '1s-d'
        assert cfg.seed == 42
        assert cfg.n_episodes == 100
        assert cfg.agent_type == 'dqn'
        assert cfg.lr == 0.001
        assert cfg.gamma == 0.99
        assert cfg.epsilon_start == 1.0
        assert cfg.epsilon_end == 0.01
        assert cfg.epsilon_decay == 10000
        assert cfg.reward_scale == 1.0
        assert cfg.n_particles == 1
        assert cfg.use_cost_mode is False


# ---------------------------------------------------------------------------
# 2. ConfigLoader init
# ---------------------------------------------------------------------------

class TestConfigLoaderInit:
    def test_default_base_dir(self):
        loader = ConfigLoader()
        assert loader.base_dir == '.'

    def test_custom_base_dir(self):
        loader = ConfigLoader(base_dir='/tmp')
        assert loader.base_dir == '/tmp'


# ---------------------------------------------------------------------------
# 3. merge_configs basic (override wins)
# ---------------------------------------------------------------------------

class TestMergeConfigsBasic:
    def test_override_wins(self):
        loader = ConfigLoader()
        base = {'a': 1, 'b': 2}
        override = {'b': 99, 'c': 3}
        result = loader.merge_configs(base, override)
        assert result['a'] == 1
        assert result['b'] == 99
        assert result['c'] == 3


# ---------------------------------------------------------------------------
# 4. merge_configs nested dict
# ---------------------------------------------------------------------------

class TestMergeConfigsNested:
    def test_nested_merge(self):
        loader = ConfigLoader()
        base = {'a': 1, 'nested': {'x': 10, 'y': 20}}
        override = {'nested': {'y': 99, 'z': 30}, 'b': 2}
        result = loader.merge_configs(base, override)
        assert result['a'] == 1
        assert result['nested']['x'] == 10
        assert result['nested']['y'] == 99
        assert result['nested']['z'] == 30
        assert result['b'] == 2


# ---------------------------------------------------------------------------
# 5. apply_overrides changes fields
# ---------------------------------------------------------------------------

class TestApplyOverridesChanges:
    def test_changes_fields(self):
        loader = ConfigLoader()
        cfg = ConfigSchema()
        loader.apply_overrides(cfg, {'lr': 0.01, 'n_episodes': 50})
        assert cfg.lr == 0.01
        assert cfg.n_episodes == 50


# ---------------------------------------------------------------------------
# 6. apply_overrides ignores unknown keys
# ---------------------------------------------------------------------------

class TestApplyOverridesIgnoresUnknown:
    def test_ignores_unknown(self):
        loader = ConfigLoader()
        cfg = ConfigSchema()
        loader.apply_overrides(cfg, {'lr': 0.01, 'nonexistent_field': 'hello'})
        assert cfg.lr == 0.01
        assert not hasattr(cfg, 'nonexistent_field') or getattr(cfg, 'nonexistent_field', None) is None
        # original field unchanged
        assert cfg.gamma == 0.99


# ---------------------------------------------------------------------------
# 7. parse_cli_overrides int
# ---------------------------------------------------------------------------

class TestParseCliInt:
    def test_int(self):
        loader = ConfigLoader()
        result = loader.parse_cli_overrides(['n_episodes=200', 'seed=99'])
        assert result['n_episodes'] == 200
        assert result['seed'] == 99
        assert isinstance(result['n_episodes'], int)


# ---------------------------------------------------------------------------
# 8. parse_cli_overrides float
# ---------------------------------------------------------------------------

class TestParseCliFloat:
    def test_float(self):
        loader = ConfigLoader()
        result = loader.parse_cli_overrides(['lr=0.005', 'gamma=0.95'])
        assert result['lr'] == pytest.approx(0.005)
        assert result['gamma'] == pytest.approx(0.95)
        assert isinstance(result['lr'], float)


# ---------------------------------------------------------------------------
# 9. parse_cli_overrides bool
# ---------------------------------------------------------------------------

class TestParseCliBool:
    def test_bool_true(self):
        loader = ConfigLoader()
        result = loader.parse_cli_overrides(['use_cost_mode=true'])
        assert result['use_cost_mode'] is True

    def test_bool_false(self):
        loader = ConfigLoader()
        result = loader.parse_cli_overrides(['use_cost_mode=false'])
        assert result['use_cost_mode'] is False


# ---------------------------------------------------------------------------
# 10. parse_cli_overrides str
# ---------------------------------------------------------------------------

class TestParseCliStr:
    def test_str(self):
        loader = ConfigLoader()
        result = loader.parse_cli_overrides(['agent_type=ppo', 'envtype=5s-c'])
        assert result['agent_type'] == 'ppo'
        assert result['envtype'] == '5s-c'
        assert isinstance(result['agent_type'], str)


# ---------------------------------------------------------------------------
# 11. parse_cli_overrides empty list
# ---------------------------------------------------------------------------

class TestParseCliEmpty:
    def test_empty(self):
        loader = ConfigLoader()
        result = loader.parse_cli_overrides([])
        assert result == {}


# ---------------------------------------------------------------------------
# 12. build_config with CLI only
# ---------------------------------------------------------------------------

class TestBuildConfigCliOnly:
    def test_cli_only(self):
        loader = ConfigLoader()
        cfg = loader.build_config(cli_overrides=['lr=0.01', 'n_episodes=500', 'agent_type=ppo'])
        assert cfg.lr == 0.01
        assert cfg.n_episodes == 500
        assert cfg.agent_type == 'ppo'
        # untouched defaults preserved
        assert cfg.gamma == 0.99
        assert cfg.envtype == '1s-d'


# ---------------------------------------------------------------------------
# 13. build_config with YAML file (temp file)
# ---------------------------------------------------------------------------

class TestBuildConfigYaml:
    def test_yaml_file(self, tmp_path):
        yaml_content = {
            'lr': 0.05,
            'n_episodes': 300,
            'envtype': '5s-c',
        }
        cfg_file = tmp_path / 'test_config.yaml'
        cfg_file.write_text(json.dumps(yaml_content))  # JSON is valid YAML

        loader = ConfigLoader(base_dir=str(tmp_path))
        cfg = loader.build_config(yaml_path='test_config.yaml')
        assert cfg.lr == pytest.approx(0.05)
        assert cfg.n_episodes == 300
        assert cfg.envtype == '5s-c'
        # defaults preserved
        assert cfg.seed == 42


# ---------------------------------------------------------------------------
# 14. validate catches epsilon_start < epsilon_end
# ---------------------------------------------------------------------------

class TestValidateCatchesBad:
    def test_epsilon_inverted(self):
        loader = ConfigLoader()
        cfg = ConfigSchema(epsilon_start=0.01, epsilon_end=1.0)
        issues = loader.validate(cfg)
        assert len(issues) >= 1
        assert any('epsilon_start' in msg for msg in issues)


# ---------------------------------------------------------------------------
# 15. validate passes valid config
# ---------------------------------------------------------------------------

class TestValidatePassesValid:
    def test_valid_config(self):
        loader = ConfigLoader()
        cfg = ConfigSchema()  # all defaults are valid
        issues = loader.validate(cfg)
        assert issues == []


# ---------------------------------------------------------------------------
# 16. to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------

class TestDictRoundtrip:
    def test_roundtrip(self):
        cfg = ConfigSchema(lr=0.05, n_episodes=250, agent_type='ppo')
        loader = ConfigLoader()
        d = loader.to_dict(cfg)
        assert isinstance(d, dict)
        assert d['lr'] == 0.05
        cfg2 = ConfigLoader.from_dict(d)
        assert cfg2.lr == 0.05
        assert cfg2.n_episodes == 250
        assert cfg2.agent_type == 'ppo'
        assert cfg2.envtype == cfg.envtype
        assert cfg2.seed == cfg.seed

    def test_from_dict_ignores_extra_keys(self):
        d = ConfigLoader().to_dict(ConfigSchema())
        d['extra_unknown_key'] = 'should_be_ignored'
        cfg = ConfigLoader.from_dict(d)
        assert cfg.lr == 0.001


# ---------------------------------------------------------------------------
# 17. save/load yaml roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_save_load(self, tmp_path):
        loader = ConfigLoader(base_dir=str(tmp_path))
        cfg = ConfigSchema(lr=0.02, n_episodes=42, use_cost_mode=True)
        out_path = str(tmp_path / 'saved_config.yaml')
        loader.save_yaml(cfg, out_path)
        assert os.path.isfile(out_path)

        cfg2 = ConfigLoader.load_yaml_config(out_path)
        assert cfg2.lr == pytest.approx(0.02)
        assert cfg2.n_episodes == 42
        assert cfg2.use_cost_mode is True


# ---------------------------------------------------------------------------
# 18. generate_default_config structure
# ---------------------------------------------------------------------------

class TestGenerateDefaultConfig:
    def test_structure(self):
        d = generate_default_config()
        assert isinstance(d, dict)
        expected_keys = {
            'envtype', 'seed', 'n_episodes', 'agent_type', 'lr',
            'gamma', 'epsilon_start', 'epsilon_end', 'epsilon_decay',
            'reward_scale', 'n_particles', 'use_cost_mode',
        }
        assert expected_keys.issubset(d.keys())
        assert d['envtype'] == '1s-d'
        assert d['seed'] == 42


# ---------------------------------------------------------------------------
# 19. generate_env_config different per envtype
# ---------------------------------------------------------------------------

class TestGenerateEnvConfig:
    def test_simple_vs_complex(self):
        simple = generate_env_config('1s-d')
        complex_ = generate_env_config('5s-c')
        assert simple['envtype'] == '1s-d'
        assert complex_['envtype'] == '5s-c'
        # complex env gets more episodes
        assert complex_['n_episodes'] > simple['n_episodes']

    def test_continuous_differs(self):
        discrete = generate_env_config('1s-d')
        continuous = generate_env_config('1s-c')
        assert continuous['lr'] < discrete['lr']

    def test_cost_mode(self):
        cost = generate_env_config('5s-c-cost')
        assert cost['use_cost_mode'] is True
        assert cost['reward_scale'] == 0.5


# ---------------------------------------------------------------------------
# 20. format_config_report contains key names
# ---------------------------------------------------------------------------

class TestFormatConfigReport:
    def test_contains_keys(self):
        cfg = ConfigSchema()
        report = format_config_report(cfg)
        for key in ('envtype', 'seed', 'n_episodes', 'lr', 'gamma', 'epsilon_start'):
            assert key in report

    def test_format(self):
        cfg = ConfigSchema(lr=0.99)
        report = format_config_report(cfg)
        assert 'lr=0.99' in report


# ---------------------------------------------------------------------------
# 21. config_diff detects changes
# ---------------------------------------------------------------------------

class TestConfigDiff:
    def test_detects_changes(self):
        a = ConfigSchema(lr=0.001, n_episodes=100)
        b = ConfigSchema(lr=0.01, n_episodes=200)
        diff = config_diff(a, b)
        assert 'lr' in diff
        assert 'n_episodes' in diff
        assert diff['lr']['a'] == 0.001
        assert diff['lr']['b'] == 0.01
        assert diff['lr']['changed'] is True
        # unchanged fields not in diff
        assert 'seed' not in diff

    def test_identical_configs(self):
        a = ConfigSchema()
        b = ConfigSchema()
        diff = config_diff(a, b)
        assert diff == {}
