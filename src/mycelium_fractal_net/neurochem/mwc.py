from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mwc_fraction(concentration_um: float, affinity_um: float) -> float:
    if affinity_um <= 0:
        return 0.0
    conc = max(0.0, concentration_um)
    return float(conc / (conc + affinity_um))


def effective_gabaa_shunt(
    active_fraction: NDArray[np.float64], shunt_strength: float
) -> NDArray[np.float64]:
    return np.clip(active_fraction * max(0.0, shunt_strength), 0.0, 0.95)


def effective_serotonergic_gain(
    plasticity_drive: NDArray[np.float64], fluidity_coeff: float, coherence_bias: float
) -> NDArray[np.float64]:
    raw = fluidity_coeff * plasticity_drive + coherence_bias
    return np.clip(raw, -0.10, 0.25)
