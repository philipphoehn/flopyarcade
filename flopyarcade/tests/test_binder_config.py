"""Tests for binder/ configuration files (C16: BinderHub config)."""

import os
import stat
from pathlib import Path

import pytest

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

REPO_ROOT = Path(__file__).resolve().parents[2]
BINDER_DIR = REPO_ROOT / "binder"


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------

class TestBinderFilesExist:
    def test_binder_dir_exists(self):
        assert BINDER_DIR.is_dir(), f"binder/ directory not found at {BINDER_DIR}"

    def test_environment_yml_exists(self):
        assert (BINDER_DIR / "environment.yml").is_file()

    def test_postbuild_exists(self):
        assert (BINDER_DIR / "postBuild").is_file()

    def test_postbuild_is_executable(self):
        mode = os.stat(BINDER_DIR / "postBuild").st_mode
        assert mode & stat.S_IXUSR, "postBuild should be executable by owner"


# ---------------------------------------------------------------------------
# environment.yml content
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
class TestEnvironmentYml:
    @pytest.fixture
    def env(self):
        with open(BINDER_DIR / "environment.yml") as f:
            return yaml.safe_load(f)

    def test_valid_yaml(self, env):
        assert isinstance(env, dict)

    def test_has_name(self, env):
        assert "name" in env
        assert env["name"] == "flopyarcade"

    def test_has_dependencies(self, env):
        assert "dependencies" in env
        assert len(env["dependencies"]) > 0

    def test_python_version(self, env):
        deps_str = [str(d) for d in env["dependencies"]]
        assert any("python" in d for d in deps_str)

    def test_flopy_in_deps(self, env):
        deps_str = [str(d) for d in env["dependencies"]]
        assert any("flopy" in d for d in deps_str)

    def test_gymnasium_in_deps(self, env):
        deps_str = [str(d) for d in env["dependencies"]]
        assert any("gymnasium" in d for d in deps_str)

    def test_jupyterlab_in_deps(self, env):
        deps_str = [str(d) for d in env["dependencies"]]
        assert any("jupyterlab" in d for d in deps_str)


# ---------------------------------------------------------------------------
# postBuild content
# ---------------------------------------------------------------------------

class TestPostBuild:
    def test_shebang(self):
        first_line = (BINDER_DIR / "postBuild").read_text().splitlines()[0]
        assert first_line.startswith("#!"), "postBuild must have a shebang"

    def test_references_flopyarcade(self):
        content = (BINDER_DIR / "postBuild").read_text()
        assert "flopyarcade" in content
