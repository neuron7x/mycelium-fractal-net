"""Generate a deterministic requirements.lock based on pinned dependencies.

This script resolves the dependency tree starting from the pinned entries in
`pyproject.toml` (project + dev extras) and writes a fully pinned
`requirements.lock` file. It should be run from an environment where the
dependencies are already installed (ideally a fresh virtualenv).
"""

from __future__ import annotations

from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Iterable

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
LOCK_FILE = ROOT / "requirements.lock"
EXCLUDED = {canonicalize_name(name) for name in ("pip", "wheel", "setuptools")}


def _load_root_requirements() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    project = data["project"]
    base = project.get("dependencies", [])
    dev = project.get("optional-dependencies", {}).get("dev", [])
    return list(base) + list(dev)


def _canonical_name(req: str) -> str:
    return canonicalize_name(req)


def _resolve_closure(roots: Iterable[str]) -> dict[str, metadata.Distribution]:
    seen: dict[str, metadata.Distribution] = {}
    stack = list(roots)

    while stack:
        raw_req = stack.pop()
        parsed = Requirement(raw_req)
        if parsed.marker and not parsed.marker.evaluate({"extra": None}):
            continue

        name = _canonical_name(parsed.name)
        if name in EXCLUDED or name in seen:
            continue

        try:
            dist = metadata.distribution(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"Dependency '{raw_req}' is not installed. "
                "Install pinned requirements first, e.g. `pip install -r requirements.txt`."
            ) from exc

        seen[name] = dist
        for child in dist.requires or []:
            stack.append(child)

    return seen


def main() -> None:
    roots = _load_root_requirements()
    closure = _resolve_closure(roots)

    lines = [f"{dist.metadata['Name']}=={dist.version}" for _, dist in sorted(closure.items())]
    LOCK_FILE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} locked dependencies to {LOCK_FILE}")


if __name__ == "__main__":
    main()
