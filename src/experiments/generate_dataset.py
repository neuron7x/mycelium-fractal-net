"""Compatibility shim for experiments.generate_dataset."""

from warnings import warn

from mycelium_fractal_net.experiments.generate_dataset import *  # noqa: F401,F403

warn(
    "Importing 'experiments.generate_dataset' is deprecated; use 'mycelium_fractal_net.experiments.generate_dataset' instead.",
    DeprecationWarning,
    stacklevel=2,
)
