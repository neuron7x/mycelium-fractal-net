#!/usr/bin/env python
"""
CLI entrypoint for MyceliumFractalNet v4.1.

Canonical CLI: ``mfn-validate`` (this file remains as a legacy shim).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Додати src/ у sys.path для локального запуску скрипта без інсталяції пакету
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mycelium_fractal_net.cli import main_validate

if __name__ == "__main__":
    main_validate()
