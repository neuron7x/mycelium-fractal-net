"""Deprecated root compatibility shim for the historical FastAPI application."""

from mycelium_fractal_net.api import *  # noqa: F401,F403
from mycelium_fractal_net.api import app, main
from mycelium_fractal_net.integration import schemas as _schemas  # noqa: F401

__all__ = ["app", "main"]

if __name__ == '__main__':
    main()
