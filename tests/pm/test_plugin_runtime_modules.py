"""Modules that plugins import from the Hermes venv must stay installed.

Plugins run inside the Hermes interpreter and may import third-party modules
that core itself no longer uses. Earlier releases carried PyYAML transitively;
when that path disappeared, plugins importing ``yaml`` failed on upgrade with
ModuleNotFoundError. Each module listed here must be declared by an extra that
``[all]`` includes, locked under ``[all]``, and importable from the built venv.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# import name -> (pyproject extra, distribution name)
PLUGIN_RUNTIME_MODULES = {
    "yaml": ("yaml", "pyyaml"),
}


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _dist(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;@]", requirement, maxsplit=1)[0].lower().replace("_", "-")


@pytest.mark.parametrize("module", sorted(PLUGIN_RUNTIME_MODULES))
def test_module_is_declared_by_an_extra_in_all(module):
    extra, dist = PLUGIN_RUNTIME_MODULES[module]
    optional = _pyproject()["project"]["optional-dependencies"]
    assert dist in {_dist(r) for r in optional[extra]}
    assert f"hermes-agent[{extra}]" in optional["all"]


@pytest.mark.parametrize("module", sorted(PLUGIN_RUNTIME_MODULES))
def test_module_is_locked_under_all(module):
    _, dist = PLUGIN_RUNTIME_MODULES[module]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    project = next(p for p in lock["package"] if p["name"] == "hermes-agent")
    locked = {d["name"] for d in project["optional-dependencies"]["all"]}
    assert dist in locked
    assert any(p["name"] == dist for p in lock["package"])


@pytest.mark.parametrize("module", sorted(PLUGIN_RUNTIME_MODULES))
def test_module_imports_from_built_environment(module):
    imported = importlib.import_module(module)
    assert Path(imported.__file__).is_file()


def test_yaml_safe_load_round_trips():
    import yaml

    assert yaml.safe_load(yaml.safe_dump({"a": [1, "b"]})) == {"a": [1, "b"]}
