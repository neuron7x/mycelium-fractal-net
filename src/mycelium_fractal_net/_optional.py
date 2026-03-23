from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


class MissingOptionalDependencyError(ImportError):
    """Raised when an optional dependency surface is accessed without its extra installed."""


_ML_INSTALL_HINT = (
    "This surface requires optional ML dependencies. Install with "
    "`pip install mycelium-fractal-net[ml]` or `uv sync --extra ml`."
)


def require_ml_dependency(module_name: str = "torch") -> ModuleType:
    """Import an optional ML dependency, raising a clear error if missing.

    Args:
        module_name: The module to import (e.g. ``"torch"``).

    Returns:
        The imported module.

    Raises:
        MissingOptionalDependencyError: If the module cannot be imported.
    """
    try:
        return import_module(module_name)
    except ImportError as exc:  # pragma: no cover
        raise MissingOptionalDependencyError(
            f"Missing optional ML dependency '{module_name}'. {_ML_INSTALL_HINT}"
        ) from exc


def optional_dependency_error(
    module_name: str = "torch",
) -> MissingOptionalDependencyError:
    return MissingOptionalDependencyError(
        f"Missing optional ML dependency '{module_name}'. {_ML_INSTALL_HINT}"
    )


__all__ = [
    "MissingOptionalDependencyError",
    "optional_dependency_error",
    "require_ml_dependency",
]
