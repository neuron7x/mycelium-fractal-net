"""
Command-line entrypoints for MyceliumFractalNet.

Canonical console scripts:
- ``mfn-validate`` — runs the validation CLI (existing behavior)
"""

from __future__ import annotations

from mycelium_fractal_net.model import run_validation_cli


def main_validate() -> int:
    """Run the validation CLI and return an exit status."""
    run_validation_cli()
    return 0


__all__ = ["main_validate"]
