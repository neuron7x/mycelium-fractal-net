"""
Compatibility shim for the MyceliumFractalNet API.

The canonical application now lives in ``mycelium_fractal_net.api``.
This module forwards to the packaged app so existing commands such as
``uvicorn api:app`` keep working when running from the repository root.
"""

from __future__ import annotations

import mycelium_fractal_net.integration  # noqa: F401
from mycelium_fractal_net import integration as _integration  # noqa: F401
from mycelium_fractal_net.api import *  # noqa: F401,F403
from mycelium_fractal_net.api import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
