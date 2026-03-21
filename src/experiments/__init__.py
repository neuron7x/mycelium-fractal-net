"""Compatibility shim for experiments; canonical code lives under mycelium_fractal_net.experiments."""

from warnings import warn

from mycelium_fractal_net.experiments.generate_dataset import (
    ConfigSampler,
    SweepConfig,
    generate_dataset,
    to_record,
)

warn(
    "Importing 'experiments' is deprecated; use 'mycelium_fractal_net.experiments' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ConfigSampler", "SweepConfig", "generate_dataset", "to_record"]
