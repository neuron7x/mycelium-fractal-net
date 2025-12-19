from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQ_TXT = ROOT / "requirements.txt"
REQ_LOCK = ROOT / "requirements.lock"


def _load_pyproject_requirements() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    project = data["project"]
    deps = list(project.get("dependencies", []))
    dev_deps = list(project.get("optional-dependencies", {}).get("dev", []))
    return deps + dev_deps


def _assert_pinned(req: Requirement, source: str) -> str:
    specs = list(req.specifier)
    assert (
        len(specs) == 1 and specs[0].operator == "=="
    ), f"{source} must pin {req.name} with '==': {req}"
    return specs[0].version


def _parse_lines(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        version = _assert_pinned(req, path.name)
        mapping[canonicalize_name(req.name)] = version
    return mapping


def _pyproject_version_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for dep in _load_pyproject_requirements():
        req = Requirement(dep)
        version = _assert_pinned(req, "pyproject.toml")
        mapping[canonicalize_name(req.name)] = version
    return mapping


def test_pyproject_dependencies_are_pinned() -> None:
    for dep in _load_pyproject_requirements():
        req = Requirement(dep)
        _assert_pinned(req, "pyproject.toml")


def test_requirements_matches_pyproject_versions() -> None:
    pyproject = _pyproject_version_map()
    requirements = _parse_lines(REQ_TXT)

    for name, version in pyproject.items():
        assert name in requirements, f"{name} missing from requirements.txt"
        assert (
            requirements[name] == version
        ), f"{name} version mismatch: pyproject {version} vs requirements.txt {requirements[name]}"


def test_lock_contains_core_dependencies() -> None:
    pyproject = _pyproject_version_map()
    locked = _parse_lines(REQ_LOCK)

    for name, version in pyproject.items():
        assert name in locked, f"{name} missing from requirements.lock"
        assert (
            locked[name] == version
        ), f"{name} version mismatch in requirements.lock: expected {version}, found {locked[name]}"
