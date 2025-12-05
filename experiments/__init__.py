"""
Experiments module for MyceliumFractalNet.

⚠️ DEPRECATION WARNING ⚠️
This root-level experiments module is DEPRECATED and will be removed in v5.0.0.

Please migrate to the canonical import:
    from mycelium_fractal_net.experiments import generate_dataset
    from mycelium_fractal_net.pipelines import run_scenario, get_preset_config

See docs/MIGRATION_GUIDE.md for detailed migration instructions.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Root-level 'experiments' module is deprecated and will be removed in v5.0.0. "
    "Please use 'from mycelium_fractal_net.experiments import ...' or "
    "'from mycelium_fractal_net.pipelines import ...' instead. "
    "See docs/MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2,
)

from .generate_dataset import SweepConfig, generate_dataset
from .inspect_features import compute_descriptive_stats, load_dataset

__all__ = [
    "generate_dataset",
    "SweepConfig",
    "load_dataset",
    "compute_descriptive_stats",
]
