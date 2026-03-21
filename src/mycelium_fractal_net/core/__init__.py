"""Core numerical and canonical operation exports."""

from __future__ import annotations

from importlib import import_module

from .engine import run_mycelium_simulation, run_mycelium_simulation_with_history
from .exceptions import NumericalInstabilityError, StabilityError, ValueOutOfRangeError
from .extract import extract
from .field import MyceliumField
from .fractal_growth_engine import FractalConfig, FractalGrowthEngine, FractalMetrics
from .membrane_engine import MembraneConfig, MembraneEngine, MembraneMetrics
from .nernst import compute_nernst_potential
from .reaction_diffusion_engine import (
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
)
from .report import report
from .stability import compute_lyapunov_exponent, compute_stability_metrics, is_stable
from .types import SimulationConfig, SimulationResult

_OPTIONAL_ATTRS = {
    'HierarchicalKrumAggregator': 'mycelium_fractal_net.core.federated',
    'aggregate_gradients_krum': 'mycelium_fractal_net.core.federated',
    'STDPPlasticity': 'mycelium_fractal_net.core.stdp',
    'estimate_fractal_dimension': 'mycelium_fractal_net.core.fractal',
    'generate_fractal_ifs': 'mycelium_fractal_net.core.fractal',
    'simulate_mycelium_field': 'mycelium_fractal_net.core.turing',
}


def __getattr__(name: str):
    if name in _OPTIONAL_ATTRS:
        module = import_module(_OPTIONAL_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    'run_mycelium_simulation', 'run_mycelium_simulation_with_history',
    'NumericalInstabilityError', 'StabilityError', 'ValueOutOfRangeError', 'extract',
    'HierarchicalKrumAggregator', 'aggregate_gradients_krum', 'MyceliumField',
    'estimate_fractal_dimension', 'generate_fractal_ifs',
    'FractalConfig', 'FractalGrowthEngine', 'FractalMetrics',
    'MembraneConfig', 'MembraneEngine', 'MembraneMetrics',
    'compute_nernst_potential', 'report',
    'ReactionDiffusionConfig', 'ReactionDiffusionEngine', 'ReactionDiffusionMetrics',
    'compute_lyapunov_exponent', 'compute_stability_metrics', 'is_stable',
    'STDPPlasticity', 'simulate_mycelium_field', 'SimulationConfig', 'SimulationResult',
]
