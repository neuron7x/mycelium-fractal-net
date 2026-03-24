"""Causal Emergence for Turing reaction-diffusion patterns.

Ref: Hoel, Albantakis & Tononi (2013) PNAS 110:19790 DOI:10.1073/pnas.1314922110
     Hoel CE 2.0 (2025) arXiv:2503.13395

EI(T) = H(<T>) - <H(T_i)> = Determinism - Degeneracy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import entropy as scipy_entropy

__all__ = [
    "CausalEmergenceResult",
    "compute_causal_emergence",
    "discretize_turing_field",
    "effective_information",
]


@dataclass
class CausalEmergenceResult:
    """Causal emergence analysis at multiple scales."""

    EI_micro: float
    EI_macro: float
    CE_macro: float
    best_scale: str
    determinism: float
    degeneracy: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "EI_micro": round(self.EI_micro, 4),
            "EI_macro": round(self.EI_macro, 4),
            "CE_macro": round(self.CE_macro, 4),
            "best_scale": self.best_scale,
            "is_causally_emergent": self.CE_macro > 0.01,
        }


def effective_information(tpm: np.ndarray) -> float:
    """EI(T) = H(avg_output) - avg_H(rows). Hoel et al. (2013) Eq. 1."""
    tpm = np.asarray(tpm, dtype=np.float64)
    tpm = tpm / (tpm.sum(axis=1, keepdims=True) + 1e-12)
    avg_output = tpm.mean(axis=0)
    H_avg = float(scipy_entropy(avg_output + 1e-12, base=2))
    avg_H = float(np.mean([scipy_entropy(tpm[i] + 1e-12, base=2) for i in range(tpm.shape[0])]))
    return max(H_avg - avg_H, 0.0)


def compute_causal_emergence(
    tpm_micro: np.ndarray,
    tpm_macro: np.ndarray | None = None,
) -> CausalEmergenceResult:
    """Compute EI and CE at micro and macro scales."""
    EI_micro = effective_information(tpm_micro)
    EI_macro = effective_information(tpm_macro) if tpm_macro is not None else EI_micro
    CE_macro = EI_macro - EI_micro
    best = "macro" if EI_macro > EI_micro else "micro"

    avg_out = tpm_micro.mean(axis=0)
    determinism = float(scipy_entropy(avg_out + 1e-12, base=2))
    degeneracy = float(np.mean([
        scipy_entropy(tpm_micro[i] + 1e-12, base=2) for i in range(tpm_micro.shape[0])
    ]))

    return CausalEmergenceResult(
        EI_micro=EI_micro, EI_macro=EI_macro, CE_macro=CE_macro,
        best_scale=best, determinism=determinism, degeneracy=degeneracy,
    )


def discretize_turing_field(field: np.ndarray) -> int:
    """Discretize field into 4 macro states: 0=homo, 1=spots, 2=stripes, 3=chaos."""
    std = float(np.std(field))
    if std < 0.01:
        return 0
    grad_x = np.diff(field, axis=1)
    grad_y = np.diff(field, axis=0)
    gx_std = float(np.std(np.abs(grad_x)))
    gy_std = float(np.std(np.abs(grad_y)))
    anisotropy = gx_std / (gy_std + 1e-6)
    if anisotropy > 1.5 or anisotropy < 0.67:
        return 2
    if float(np.var(field)) > 0.01:
        return 3
    return 1
