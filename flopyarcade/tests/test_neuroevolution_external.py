"""Structural tests for A11: External neuroevolution example scripts.

These tests validate that the training scripts exist, have the expected
structure (argparse, main function, key helper functions), and that
importable sub-modules work — without executing any training loops.
"""

import ast
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

SCRIPTS = {
    "cmaes": "train_cmaes.py",
    "rllib_es": "train_rllib_es.py",
    "evotorch": "train_evotorch.py",
}


def _parse_module(path: Path) -> ast.Module:
    """Parse a Python file into an AST without importing it."""
    src = path.read_text()
    return ast.parse(src, filename=str(path))


def _top_level_functions(tree: ast.Module) -> set:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _has_argparse(tree: ast.Module) -> bool:
    src = ast.dump(tree)
    return "ArgumentParser" in src


# --- Generic structural tests for all three scripts ---

class TestCmaesStructure:
    def setup_method(self, method):
        self.path = EXAMPLES_DIR / "train_cmaes.py"
        self.tree = _parse_module(self.path)

    def test_file_exists(self):
        assert self.path.exists()

    def test_has_main(self):
        assert "main" in _top_level_functions(self.tree)

    def test_has_argparse(self):
        assert _has_argparse(self.tree)

    def test_has_parse_args(self):
        assert "parse_args" in _top_level_functions(self.tree)

    def test_has_flatten_weights(self):
        assert "flatten_weights" in _top_level_functions(self.tree)

    def test_has_set_weights_from_flat(self):
        assert "set_weights_from_flat" in _top_level_functions(self.tree)

    def test_has_evaluate_agent(self):
        assert "evaluate_agent" in _top_level_functions(self.tree)

    def test_imports_cma(self):
        src = self.path.read_text()
        assert "import cma" in src or "cma.fmin" in src

    def test_has_cmaes_objective(self):
        assert "cmaes_objective" in _top_level_functions(self.tree)


class TestRllibEsStructure:
    def setup_method(self, method):
        self.path = EXAMPLES_DIR / "train_rllib_es.py"
        self.tree = _parse_module(self.path)

    def test_file_exists(self):
        assert self.path.exists()

    def test_has_main(self):
        assert "main" in _top_level_functions(self.tree)

    def test_has_argparse(self):
        assert _has_argparse(self.tree)

    def test_has_parse_args(self):
        assert "parse_args" in _top_level_functions(self.tree)

    def test_imports_ray(self):
        src = self.path.read_text()
        assert "import ray" in src

    def test_has_get_rl_config(self):
        assert "get_rl_config" in _top_level_functions(self.tree)

    def test_has_build_env_config(self):
        assert "build_env_config" in _top_level_functions(self.tree)

    def test_has_flopyarcade_env_wrapper(self):
        src = self.path.read_text()
        assert "FloPyArcadeEnv" in src

    def test_has_env_creator_registration(self):
        src = self.path.read_text()
        assert "register_env_creator" in src


class TestEvotorchStructure:
    def setup_method(self, method):
        self.path = EXAMPLES_DIR / "train_evotorch.py"
        self.tree = _parse_module(self.path)

    def test_file_exists(self):
        assert self.path.exists()

    def test_has_main(self):
        assert "main" in _top_level_functions(self.tree)

    def test_has_argparse(self):
        assert _has_argparse(self.tree)

    def test_has_parse_args(self):
        assert "parse_args" in _top_level_functions(self.tree)

    def test_has_flatten_weights(self):
        assert "flatten_weights" in _top_level_functions(self.tree)

    def test_has_set_weights_from_flat(self):
        assert "set_weights_from_flat" in _top_level_functions(self.tree)

    def test_has_evaluate_agent(self):
        assert "evaluate_agent" in _top_level_functions(self.tree)

    def test_has_build_torch_model(self):
        assert "build_torch_model" in _top_level_functions(self.tree)

    def test_references_evotorch(self):
        src = self.path.read_text()
        assert "evotorch" in src.lower()


# --- Import tests (lightweight, no heavy deps triggered) ---

class TestCmaesImportable:
    def test_import(self):
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        try:
            from flopyarcade.examples import train_cmaes
            assert hasattr(train_cmaes, 'main')
            assert hasattr(train_cmaes, 'parse_args')
            assert hasattr(train_cmaes, 'flatten_weights')
            assert hasattr(train_cmaes, 'set_weights_from_flat')
            assert hasattr(train_cmaes, 'evaluate_agent')
            assert hasattr(train_cmaes, 'cmaes_objective')
        finally:
            sys.path.pop(0)


class TestRllibEsImportable:
    def test_import(self):
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        try:
            from flopyarcade.examples import train_rllib_es
            assert hasattr(train_rllib_es, 'main')
            assert hasattr(train_rllib_es, 'get_rl_config')
            assert hasattr(train_rllib_es, 'build_env_config')
        finally:
            sys.path.pop(0)


class TestEvotorchImportable:
    def test_import(self):
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        try:
            from flopyarcade.examples import train_evotorch
            assert hasattr(train_evotorch, 'main')
            assert hasattr(train_evotorch, 'build_torch_model')
            assert hasattr(train_evotorch, 'flatten_weights')
            assert hasattr(train_evotorch, 'set_weights_from_flat')
            assert hasattr(train_evotorch, 'evaluate_agent')
        finally:
            sys.path.pop(0)


# --- Benchmark doc test ---

class TestBenchmarkDoc:
    def test_benchmark_md_exists(self):
        bm = EXAMPLES_DIR / "BENCHMARK_COMPARISON.md"
        assert bm.exists(), "BENCHMARK_COMPARISON.md not found"

    def test_benchmark_mentions_all_methods(self):
        bm = (EXAMPLES_DIR / "BENCHMARK_COMPARISON.md").read_text()
        assert "CMA-ES" in bm
        assert "RLlib" in bm or "ES" in bm
        assert "EvoTorch" in bm or "evotorch" in bm.lower()

