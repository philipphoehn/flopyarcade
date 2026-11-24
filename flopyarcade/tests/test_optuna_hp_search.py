"""A7: Tests for Optuna hyperparameter search (code-only, no training)."""

import json
import pytest

from flopyarcade.optuna_hp_search import (
    SEARCH_SPACE,
    build_search_space,
    suggest_params,
    objective,
    parse_args,
    main,
)


class MockTrial:
    """Mock Optuna trial for testing suggest_params without optuna installed."""

    def __init__(self):
        self.number = 0
        self._values = {}

    def suggest_int(self, name, low, high, step=1):
        val = low + step  # deterministic: first valid value above low
        self._values[name] = val
        return val

    def suggest_float(self, name, low, high):
        val = (low + high) / 2
        self._values[name] = val
        return val

    def suggest_categorical(self, name, choices):
        val = choices[0]
        self._values[name] = val
        return val


class TestSearchSpace:
    def test_all_expected_keys(self):
        expected = {
            'NAGENTS', 'NGENERATIONS', 'NAGENTSSTEPS', 'NNTYPE',
            'CROSSVALIDATEEVERY', 'NGAMESAVERAGED', 'NGAMESCROSSVALIDATED',
            'EPSILONINITIAL', 'EPSILONMIN', 'EPSILONDECAY',
            'REPLAYMEMORYSIZE', 'NOVELTYSEARCH', 'NNOVELTYNEIGHBORS',
            'NNOVELTYELITES', 'ADDNOVELTYEVERY',
        }
        assert expected.issubset(set(SEARCH_SPACE.keys()))

    def test_int_params_have_low_high(self):
        for name, spec in SEARCH_SPACE.items():
            if spec['type'] == 'int':
                assert 'low' in spec
                assert 'high' in spec
                assert spec['low'] < spec['high']

    def test_float_params_have_low_high(self):
        for name, spec in SEARCH_SPACE.items():
            if spec['type'] == 'float':
                assert 'low' in spec
                assert 'high' in spec
                assert spec['low'] < spec['high']

    def test_categorical_params_have_choices(self):
        for name, spec in SEARCH_SPACE.items():
            if spec['type'] == 'categorical':
                assert 'choices' in spec
                assert len(spec['choices']) >= 2


class TestBuildSearchSpace:
    def test_no_exclusion_returns_all(self):
        space = build_search_space()
        assert len(space) == len(SEARCH_SPACE)

    def test_exclusion_removes_keys(self):
        space = build_search_space(exclude=['NAGENTS', 'NGENERATIONS'])
        assert 'NAGENTS' not in space
        assert 'NGENERATIONS' not in space
        assert 'NNTYPE' in space

    def test_exclude_all(self):
        space = build_search_space(exclude=list(SEARCH_SPACE.keys()))
        assert space == {}

    def test_exclude_nonexistent_key(self):
        space = build_search_space(exclude=['NONEXISTENT'])
        assert len(space) == len(SEARCH_SPACE)


class TestSuggestParams:
    def test_returns_dict_with_all_keys(self):
        trial = MockTrial()
        params = suggest_params(trial)
        assert set(params.keys()) == set(SEARCH_SPACE.keys())

    def test_int_values_in_range(self):
        trial = MockTrial()
        params = suggest_params(trial)
        for name, spec in SEARCH_SPACE.items():
            if spec['type'] == 'int':
                assert spec['low'] <= params[name] <= spec['high']

    def test_float_values_in_range(self):
        trial = MockTrial()
        params = suggest_params(trial)
        for name, spec in SEARCH_SPACE.items():
            if spec['type'] == 'float':
                assert spec['low'] <= params[name] <= spec['high']

    def test_categorical_values_in_choices(self):
        trial = MockTrial()
        params = suggest_params(trial)
        for name, spec in SEARCH_SPACE.items():
            if spec['type'] == 'categorical':
                assert params[name] in spec['choices']

    def test_custom_space(self):
        trial = MockTrial()
        custom = {'MY_PARAM': {'type': 'int', 'low': 1, 'high': 10, 'step': 1}}
        params = suggest_params(trial, space=custom)
        assert 'MY_PARAM' in params
        assert 1 <= params['MY_PARAM'] <= 10

    def test_unknown_type_raises(self):
        trial = MockTrial()
        bad = {'BAD': {'type': 'unknown', 'low': 1, 'high': 10}}
        with pytest.raises(ValueError, match="Unknown param type"):
            suggest_params(trial, space=bad)


class TestObjective:
    def test_stub_returns_zero(self):
        trial = MockTrial()
        result = objective(trial, envtype='1s-d')
        assert result == 0.0

    def test_stub_with_custom_space(self):
        trial = MockTrial()
        space = {'NAGENTS': {'type': 'int', 'low': 8, 'high': 16, 'step': 8}}
        result = objective(trial, envtype='2s-d', space=space)
        assert result == 0.0


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.envtype == '1s-d'
        assert args.n_trials == 50
        assert args.study_name == 'flopyarcade_hp'
        assert args.storage is None
        assert args.sampler == 'TPESampler'
        assert args.seed is None
        assert args.exclude is None
        assert args.dry_run is False

    def test_custom_args(self):
        args = parse_args([
            '--envtype', '3s-c',
            '--n-trials', '100',
            '--study-name', 'my_study',
            '--storage', 'sqlite:///test.db',
            '--seed', '42',
        ])
        assert args.envtype == '3s-c'
        assert args.n_trials == 100
        assert args.study_name == 'my_study'
        assert args.storage == 'sqlite:///test.db'
        assert args.seed == 42

    def test_exclude_multiple(self):
        args = parse_args(['--exclude', 'NAGENTS', 'NGENERATIONS'])
        assert args.exclude == ['NAGENTS', 'NGENERATIONS']

    def test_dry_run_flag(self):
        args = parse_args(['--dry-run'])
        assert args.dry_run is True


class TestMain:
    def test_dry_run_exits_zero(self, capsys):
        rc = main(['--dry-run'])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert 'NAGENTS' in data

    def test_dry_run_with_exclude(self, capsys):
        rc = main(['--dry-run', '--exclude', 'NAGENTS'])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert 'NAGENTS' not in data


class TestDoNotRunComments:
    def test_objective_has_dont_run(self):
        import inspect
        src = inspect.getsource(objective)
        assert 'DO NOT RUN' in src

    def test_run_search_has_dont_run(self):
        import inspect
        from flopyarcade.optuna_hp_search import run_search
        src = inspect.getsource(run_search)
        assert 'DO NOT RUN' in src
