"""Biological fungal mechanisms for MyceliumFractalNet.

Five peer-reviewed mechanisms from mycology translated to numpy:
1. Physarum polycephalum adaptive conductivity (Tero & Nakagaki 2007, Science 2010)
2. Hyphal anastomosis + foraging front (Du et al. 2019, J. Theor. Biol.)
3. FitzHugh-Nagumo excitable signaling (Adamatzky 2023, Sci. Rep.)
4. Fat-tailed spore dispersal (Clark 1998, Kot et al. 1996)
5. Keller-Segel chemotaxis (Boswell et al. 2003, Bull. Math. Biol.)

Usage:
    import mycelium_fractal_net as mfn
    from mycelium_fractal_net.bio import BioExtension, BioConfig

    seq = mfn.simulate(mfn.SimulationSpec(grid_size=32, steps=60, seed=42))
    bio = BioExtension.from_sequence(seq)
    bio = bio.step(n=10)
    print(bio.report().summary())
"""

from .anastomosis import AnastomosisConfig, AnastomosisEngine, AnastomosisState
from .chemotaxis import ChemotaxisConfig, ChemotaxisEngine, ChemotaxisState
from .dispersal import DispersalConfig, SporeDispersalEngine, SporeDispersalState
from .extension import BioConfig, BioExtension, BioReport
from .fhn import FHNConfig, FHNEngine, FHNState
from .physarum import PhysarumConfig, PhysarumEngine, PhysarumState

# Memory-Augmented Evolution (HDV + CMA-ES)
from .evolution import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    BioEvolutionOptimizer,
    BioEvolutionResult,
    compute_fitness,
    params_to_bio_config,
)
from .memory import BioMemory, HDVEncoder, MemoryEntry
from .meta import MetaOptimizer, MetaOptimizerResult

__all__ = [
    "AnastomosisConfig",
    "AnastomosisEngine",
    "AnastomosisState",
    "BioConfig",
    "BioExtension",
    "BioReport",
    "ChemotaxisConfig",
    "ChemotaxisEngine",
    "ChemotaxisState",
    "DispersalConfig",
    "FHNConfig",
    "FHNEngine",
    "FHNState",
    "PhysarumConfig",
    "PhysarumEngine",
    "PhysarumState",
    "SporeDispersalEngine",
    "SporeDispersalState",
    # Memory-Augmented Evolution
    "BioEvolutionOptimizer",
    "BioEvolutionResult",
    "BioMemory",
    "HDVEncoder",
    "MemoryEntry",
    "MetaOptimizer",
    "MetaOptimizerResult",
    "PARAM_BOUNDS",
    "PARAM_NAMES",
    "compute_fitness",
    "params_to_bio_config",
]
