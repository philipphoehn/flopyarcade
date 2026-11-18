"""C17: Verify upload_and_grade function for autograder."""

import ast
import os
import sys
import inspect

import pytest

BENCH_PY = os.path.join(os.path.dirname(__file__), '..', 'benchmark.py')


def _read_bench():
    with open(BENCH_PY, 'r') as f:
        return f.read()


def test_upload_and_grade_function_exists():
    """upload_and_grade is defined in benchmark.py."""
    source = _read_bench()
    assert 'def upload_and_grade(' in source


def test_upload_and_grade_accepts_url():
    """The function handles http/https URLs via urllib download."""
    source = _read_bench()
    assert 'http://' in source
    assert 'https://' in source
    assert 'urllib.request.urlretrieve' in source


def test_upload_and_grade_returns_dict():
    """The function returns a dict with status key."""
    source = _read_bench()
    # success path uses assignment: results['status'] = 'success'
    assert "['status'] = 'success'" in source
    # error path uses dict literal: 'status': 'error'
    assert "'status': 'error'" in source


def test_upload_and_grade_cleans_temp():
    """Downloaded temp files are cleaned up in finally block."""
    source = _read_bench()
    assert 'shutil.rmtree' in source
    assert 'ignore_errors=True' in source


def test_upload_and_grade_delegates_to_run_benchmark():
    """upload_and_grade calls run_benchmark internally."""
    source = _read_bench()
    assert 'run_benchmark(' in source
    # Ensure it's called within upload_and_grade (after its def)
    idx_func = source.index('def upload_and_grade(')
    idx_call = source.index('run_benchmark(', idx_func)
    assert idx_call > idx_func


def test_upload_and_grade_has_docstring():
    """upload_and_grade has a docstring explaining usage."""
    from flopyarcade.benchmark import upload_and_grade
    assert upload_and_grade.__doc__ is not None
    assert 'source' in upload_and_grade.__doc__


def test_upload_and_grade_error_handling():
    """Non-existent local file returns error status dict."""
    from flopyarcade.benchmark import upload_and_grade
    result = upload_and_grade('/nonexistent/model_123.keras', n_games=1)
    assert result['status'] == 'error'
    assert 'error' in result


def test_upload_and_grade_default_params():
    """upload_and_grade has sensible defaults."""
    from flopyarcade.benchmark import upload_and_grade
    sig = inspect.signature(upload_and_grade)
    params = sig.parameters
    assert params['envtype'].default == '1s-d'
    assert params['n_games'].default == 100
    assert params['seed'].default == 42
    assert params['n_steps'].default == 200
