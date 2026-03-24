"""Bio extension for MyceliumFractalNet.

Public surface (7 symbols, stable contract):
    BioExtension, BioConfig, BioReport
    MetaOptimizer, MetaOptimizerResult
    BioMemory, HDVEncoder

Hard dependencies: numpy (always), scipy (Physarum), cmaes (optional)
Import budget: bio/ must not import from integration/, api/, cli/

Computational contracts (enforced by benchmark gates):
    PhysarumEngine.step() @ N=32: < 5ms
    BioMemory.query() @ 200 eps:  < 0.5ms
    BioExtension.step(1) @ N=16:  < 10ms
"""

from .anastomosis import AnastomosisConfig, AnastomosisEngine, AnastomosisState
from .chemotaxis import ChemotaxisConfig, ChemotaxisEngine, ChemotaxisState
from .dispersal import DispersalConfig, SporeDispersalEngine, SporeDispersalState
from .evolution import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    BioEvolutionOptimizer,
    BioEvolutionResult,
    compute_fitness,
    params_to_bio_config,
)
from .extension import BioConfig, BioExtension, BioReport
from .fhn import FHNConfig, FHNEngine, FHNState
from .memory import BioMemory, HDVEncoder, MemoryEntry
from .meta import MetaOptimizer, MetaOptimizerResult
from .physarum import PhysarumConfig, PhysarumEngine, PhysarumState

# Public contract — stable across versions
__all__ = [
    "BioConfig",
    "BioExtension",
    "BioMemory",
    "BioReport",
    "HDVEncoder",
    "MetaOptimizer",
    "MetaOptimizerResult",
]

# Internal — accessible via direct import but not part of the contract:
# AnastomosisEngine, PhysarumEngine, FHNEngine, etc.
# BioEvolutionOptimizer, PARAM_BOUNDS, PARAM_NAMES, etc.
