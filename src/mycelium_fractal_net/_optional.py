from __future__ import annotations

from importlib import import_module
from types import ModuleType


class MissingOptionalDependencyError(ImportError):
    """Raised when an optional dependency surface is accessed without its extra installed."""


_ML_INSTALL_HINT = (
    "This surface requires optional ML dependencies. Install with "
    "`pip install mycelium-fractal-net[ml]` or `uv sync --extra ml`."
)


def require_ml_dependency(module_name: str = 'torch') -> ModuleType:
    try:
        return import_module(module_name)
    except Exception as exc:  # pragma: no cover - exact ImportError text varies by platform
        raise MissingOptionalDependencyError(
            f"Missing optional ML dependency '{module_name}'. {_ML_INSTALL_HINT}"
        ) from exc


def optional_dependency_error(module_name: str = 'torch') -> MissingOptionalDependencyError:
    return MissingOptionalDependencyError(
        f"Missing optional ML dependency '{module_name}'. {_ML_INSTALL_HINT}"
    )


__all__ = ['MissingOptionalDependencyError', 'require_ml_dependency', 'optional_dependency_error']
