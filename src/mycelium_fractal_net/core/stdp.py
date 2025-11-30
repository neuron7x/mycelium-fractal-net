"""
STDP Plasticity Module — Spike-Timing Dependent Plasticity.

This module provides the STDP (Spike-Timing Dependent Plasticity) implementation
for heterosynaptic learning based on relative spike timing between pre- and 
postsynaptic neurons.

Reference: MFN_MATH_MODEL.md Appendix C (STDP Mathematical Model)

Mathematical Model (Bi & Poo, 1998):
    Δw = A_+ exp(-Δt/τ_+)  if Δt > 0 (LTP)
    Δw = -A_- exp(Δt/τ_-)  if Δt < 0 (LTD)

Parameters (from neurophysiology):
    τ+ = τ- = 20 ms (time constant)
    A+ = 0.01 (LTP magnitude)
    A- = 0.012 (LTD magnitude, asymmetric for stability)

Biophysical Basis:
    - NMDA receptor activation requires coincident pre/post activity
    - Ca²⁺ influx magnitude determines potentiation vs depression
    - Asymmetry (A- > A+) prevents runaway excitation
"""

from __future__ import annotations

import torch
from torch import nn

# STDP parameters (from MFN_MATH_MODEL.md)
STDP_TAU_PLUS: float = 0.020  # 20 ms
STDP_TAU_MINUS: float = 0.020  # 20 ms
STDP_A_PLUS: float = 0.01
STDP_A_MINUS: float = 0.012


class STDPPlasticity(nn.Module):
    """
    Spike-Timing Dependent Plasticity (STDP) module.

    Mathematical Model (Bi & Poo, 1998):
    ------------------------------------
    The STDP learning rule implements Hebbian plasticity based on relative
    spike timing between pre- and postsynaptic neurons:

    .. math::

        \\Delta w = \\begin{cases}
            A_+ e^{-\\Delta t/\\tau_+} & \\Delta t > 0 \\text{ (LTP)} \\\\
            -A_- e^{\\Delta t/\\tau_-} & \\Delta t < 0 \\text{ (LTD)}
        \\end{cases}

    where:
        - Δt = t_post - t_pre (spike timing difference)
        - τ+ = τ- = 20 ms (time constant, from hippocampal slice recordings)
        - A+ = 0.01 (LTP magnitude, dimensionless)
        - A- = 0.012 (LTD magnitude, dimensionless, asymmetric for stability)

    Biophysical Basis:
    ------------------
    - NMDA receptor activation requires coincident pre/post activity
    - Ca²⁺ influx magnitude determines potentiation vs depression
    - Asymmetry (A- > A+) prevents runaway excitation

    Parameter Constraints:
    ----------------------
    - τ ∈ [5, 100] ms: Biological range from cortical recordings
    - A ∈ [0.001, 0.1]: Prevents weight explosion while maintaining plasticity
    - A-/A+ > 1: Ensures stable network dynamics (prevents runaway LTP)

    References:
        Bi, G. & Poo, M. (1998). Synaptic modifications in cultured
        hippocampal neurons. J. Neuroscience, 18(24), 10464-10472.

        Song, S., Miller, K.D. & Abbott, L.F. (2000). Competitive Hebbian
        learning through spike-timing-dependent synaptic plasticity.
        Nature Neuroscience, 3(9), 919-926.
    """

    # Biophysically valid parameter ranges (from empirical neurophysiology)
    TAU_MIN: float = 0.005  # 5 ms
    TAU_MAX: float = 0.100  # 100 ms
    A_MIN: float = 0.001
    A_MAX: float = 0.100

    # Numerical stability constants
    EXP_CLAMP_MAX: float = 50.0  # exp(-50) ≈ 1.9e-22, prevents underflow/overflow

    def __init__(
        self,
        tau_plus: float = STDP_TAU_PLUS,
        tau_minus: float = STDP_TAU_MINUS,
        a_plus: float = STDP_A_PLUS,
        a_minus: float = STDP_A_MINUS,
    ) -> None:
        super().__init__()

        # Validate parameters against biophysical constraints
        self._validate_time_constant(tau_plus, "tau_plus")
        self._validate_time_constant(tau_minus, "tau_minus")
        self._validate_amplitude(a_plus, "a_plus")
        self._validate_amplitude(a_minus, "a_minus")

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

    def _validate_time_constant(self, tau: float, name: str) -> None:
        """Validate time constant is within biophysical range."""
        if not (self.TAU_MIN <= tau <= self.TAU_MAX):
            tau_min_ms = self.TAU_MIN * 1000
            tau_max_ms = self.TAU_MAX * 1000
            tau_ms = tau * 1000
            raise ValueError(
                f"{name}={tau_ms:.1f}ms outside biophysical range "
                f"[{tau_min_ms:.0f}, {tau_max_ms:.0f}]ms"
            )

    def _validate_amplitude(self, a: float, name: str) -> None:
        """Validate amplitude is within stable range."""
        if not (self.A_MIN <= a <= self.A_MAX):
            raise ValueError(
                f"{name}={a} outside stable range [{self.A_MIN}, {self.A_MAX}]"
            )

    def compute_weight_update(
        self,
        pre_times: torch.Tensor,
        post_times: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute STDP weight update.

        Parameters
        ----------
        pre_times : torch.Tensor
            Presynaptic spike times of shape (batch, n_pre).
        post_times : torch.Tensor
            Postsynaptic spike times of shape (batch, n_post).
        weights : torch.Tensor
            Current weights of shape (n_pre, n_post).

        Returns
        -------
        torch.Tensor
            Weight update matrix.
        """
        # Time differences: delta_t = t_post - t_pre
        # Positive delta_t means pre before post (LTP)
        delta_t = post_times.unsqueeze(-2) - pre_times.unsqueeze(-1)

        # Clamp exponential arguments to prevent underflow/overflow
        # Uses class constant EXP_CLAMP_MAX for consistency
        clamp = self.EXP_CLAMP_MAX

        # LTP: pre before post (delta_t > 0)
        ltp_mask = delta_t > 0
        ltp_exp_arg = torch.clamp(-delta_t / self.tau_plus, min=-clamp, max=clamp)
        ltp = self.a_plus * torch.exp(ltp_exp_arg)
        ltp = ltp * ltp_mask.float()

        # LTD: post before pre (delta_t < 0)
        ltd_mask = delta_t < 0
        ltd_exp_arg = torch.clamp(delta_t / self.tau_minus, min=-clamp, max=clamp)
        ltd = -self.a_minus * torch.exp(ltd_exp_arg)
        ltd = ltd * ltd_mask.float()

        # Sum updates across batch
        delta_w = (ltp + ltd).mean(dim=0)

        return delta_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through (STDP is applied via compute_weight_update)."""
        return x


__all__ = [
    # Constants
    "STDP_TAU_PLUS",
    "STDP_TAU_MINUS",
    "STDP_A_PLUS",
    "STDP_A_MINUS",
    # Main class
    "STDPPlasticity",
]
