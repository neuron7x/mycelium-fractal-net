from __future__ import annotations

from mycelium_fractal_net import api
from mycelium_fractal_net.cli import main_validate


def test_api_app_importable() -> None:
    assert hasattr(api, "app")
    assert api.app.title.startswith("MyceliumFractalNet")


def test_cli_main_validate_callable() -> None:
    assert callable(main_validate)
