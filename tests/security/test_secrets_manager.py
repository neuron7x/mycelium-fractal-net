import os
from pathlib import Path

import pytest

from mycelium_fractal_net.security import SecretManager, SecretManagerConfig, SecretRetrievalError


def test_env_secret_with_file_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret_path = tmp_path / "api_key"
    secret_path.write_text("from-file-key")

    monkeypatch.setenv("MFN_API_KEY_FILE", str(secret_path))
    # Ensure direct env value takes precedence when set
    monkeypatch.setenv("MFN_API_KEY", "env-key")

    manager = SecretManager()
    assert manager.get_secret("MFN_API_KEY", required=True) == "env-key"

    # Remove env var to validate file-backed resolution
    monkeypatch.delenv("MFN_API_KEY", raising=False)
    assert manager.get_secret("MFN_API_KEY", required=True) == "from-file-key"


def test_list_parsing_supports_multiple_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = SecretManager()

    monkeypatch.setenv("MFN_API_KEYS", "a,b , c\n d")
    assert manager.get_list("MFN_API_KEYS") == ["a", "b", "c", "d"]

    # JSON array parsing
    monkeypatch.setenv("MFN_API_KEYS", "[\"x\", \"y\"]")
    assert manager.get_list("MFN_API_KEYS") == ["x", "y"]


def test_required_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = SecretManager()
    monkeypatch.delenv("MFN_API_KEY", raising=False)
    with pytest.raises(SecretRetrievalError):
        manager.get_secret("MFN_API_KEY", required=True)


def test_file_backend_loads_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secrets_file = tmp_path / "secrets.json"
    secrets_file.write_text("{\n  \"MFN_API_KEY\": \"file-key\",\n  \"EXTRA\": \"value\"\n}")

    monkeypatch.setenv("MFN_SECRETS_BACKEND", "file")
    monkeypatch.setenv("MFN_SECRETS_FILE", str(secrets_file))

    config = SecretManagerConfig.from_env()
    manager = SecretManager(config)

    assert manager.get_secret("MFN_API_KEY", required=True) == "file-key"
    assert manager.get_secret("EXTRA") == "value"


def test_invalid_env_file_line_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secrets_file = tmp_path / "secrets.env"
    secrets_file.write_text("INVALID_LINE")

    monkeypatch.setenv("MFN_SECRETS_BACKEND", "file")
    monkeypatch.setenv("MFN_SECRETS_FILE", str(secrets_file))

    manager = SecretManager()
    with pytest.raises(SecretRetrievalError):
        manager.get_secret("ANY_KEY")
