"""
Lightweight result persistence for API responses.

Stores selected API request/response metadata as JSONL for auditing and
post-hoc analysis. This is intentionally minimal and file-based to provide
persistence without introducing a database dependency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from .api_config import PersistenceConfig, get_api_config
from .logging_config import get_logger

logger = get_logger("persistence")


REDACT_KEYS = {
    "api_key",
    "api_keys",
    "secret",
    "token",
    "password",
    "private_key",
    "key",
    "key_id",
    "ciphertext",
    "plaintext",
    "signature",
}


@dataclass
class PersistenceEvent:
    """Serializable event envelope for persisted API results."""

    timestamp_ms: int
    endpoint: str
    method: str
    request_id: str | None
    request: Mapping[str, Any]
    response: Mapping[str, Any]


class ResultStore:
    """Append-only JSONL result store for API events."""

    def __init__(self, config: PersistenceConfig | None = None) -> None:
        self.config = config or get_api_config().persistence
        self._lock = Lock()
        self._path = Path(self.config.results_path)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def record(
        self,
        *,
        endpoint: str,
        method: str,
        request_id: str | None,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return

        event = PersistenceEvent(
            timestamp_ms=int(time.time() * 1000),
            endpoint=endpoint,
            method=method,
            request_id=request_id,
            request=_sanitize_payload(request_payload),
            response=_sanitize_payload(response_payload),
        )

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(event.__dict__, ensure_ascii=False)
            with self._lock:
                self._path.open("a", encoding="utf-8").write(payload + "\n")
        except Exception as exc:  # pragma: no cover - best-effort persistence
            logger.warning(
                "Failed to persist API result",
                extra={"endpoint": endpoint, "error": str(exc)},
            )


def _sanitize_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _truncate(_redact(payload))


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in REDACT_KEYS:
                redacted[key] = "***redacted***"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)
    return value


def _truncate(value: Any, *, max_items: int = 50) -> Any:
    if isinstance(value, Mapping):
        if len(value) <= max_items:
            return {k: _truncate(v, max_items=max_items) for k, v in value.items()}
        trimmed = dict(list(value.items())[:max_items])
        trimmed["_truncated"] = True
        trimmed["_total_keys"] = len(value)
        return {k: _truncate(v, max_items=max_items) for k, v in trimmed.items()}
    if isinstance(value, list):
        if len(value) <= max_items:
            return [_truncate(item, max_items=max_items) for item in value]
        sample = [_truncate(item, max_items=max_items) for item in value[:max_items]]
        return {
            "_truncated": True,
            "_total_items": len(value),
            "_sample": sample,
        }
    if isinstance(value, tuple):
        return tuple(_truncate(item, max_items=max_items) for item in value)
    return value


__all__ = ["ResultStore", "PersistenceEvent"]
