"""B9: Verify cProfile / line_profiler integration in play.py."""

import ast
import inspect
import os
import sys

import pytest


PLAY_PY = os.path.join(os.path.dirname(__file__), '..', 'play.py')


def _read_play_py():
    with open(PLAY_PY, 'r') as f:
        return f.read()


def _parse_play_py():
    return ast.parse(_read_play_py())


def test_profile_arg_exists():
    """The --profile CLI flag is defined in play.py's ArgumentParser."""
    source = _read_play_py()
    assert '--profile' in source, "--profile flag not found in play.py"
    assert 'cProfile' in source, "cProfile not referenced in play.py"


def test_profile_flag_is_boolean():
    """The --profile argument uses str_to_bool type conversion."""
    source = _read_play_py()
    assert 'str_to_bool' in source
    # The --profile line should reference str_to_bool
    for line in source.split('\n'):
        if '--profile' in line:
            assert 'str_to_bool' in line or 'type=' in line
            break
    else:
        pytest.fail("--profile argument not found with type spec")


def test_cprofile_import_is_lazily_loaded():
    """cProfile is imported inside the if-block, not at module top level."""
    tree = _parse_play_py()
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):\
                top_level_imports.extend(a.name for a in node.names)
            else:
                top_level_imports.append(node.module or '')
    assert 'cProfile' not in top_level_imports, ("cProfile should be lazily imported, not at module top level")


def test_pstats_used_for_output():
    """pstats.Stats is used to sort and print top functions."""
    source = _read_play_py()
    assert 'pstats' in source, "pstats not referenced in play.py"
    assert 'sort_stats' in source, "sort_stats not called in play.py"
    assert 'print_stats' in source, "print_stats not called in play.py"


def test_game_loop_function_defined():
    """The game loop is wrapped in a _game_loop function for profiling."""
    source = _read_play_py()
    assert 'def _game_loop():' in source, "_game_loop function not defined"


def test_runcall_used():
    """profiler.runcall is used to execute the game loop under profiling."""
    source = _read_play_py()
    assert 'runcall' in source, "profiler.runcall not used in play.py"


def test_top20_functions():
    """The profiler prints top 20 functions by cumulative time."""
    source = _read_play_py()
    assert 'print_stats(20)' in source, "print_stats(20) not found"


def test_profile_disabled_by_default():
    """--profile defaults to False."""
    source = _read_play_py()
    for line in source.split('\n'):
        if '--profile' in line:
            assert 'default=False' in line, "--profile should default to False"
            break