from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found]


def test_optional_dependency_extras_defined():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "http" in extras
    assert "kafka" in extras
    assert "full" in extras

    assert "aiohttp>=3.9.4" in extras["http"]
    assert "kafka-python>=2.0.0" in extras["kafka"]
    assert set(extras["full"]) == {"aiohttp>=3.9.4", "kafka-python>=2.0.0"}
