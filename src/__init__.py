"""MyceliumFractalNet - Adaptive fractal neural networks with bio-inspired dynamics."""
from .mfn import (
    MyceliumFractalNet,
    FractalLayer,
    NernstPotential,
    STDPModule,
    TuringGrowth,
    load_config,
)

__version__ = "4.1.0"
__all__ = [
    "MyceliumFractalNet",
    "FractalLayer",
    "NernstPotential",
    "STDPModule",
    "TuringGrowth",
    "load_config",
]
