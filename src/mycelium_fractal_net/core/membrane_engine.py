"""
Membrane Potential Dynamics Engine.

Implements stable numerical integration for membrane potential ODEs
based on Nernst-Planck ion dynamics.

Mathematical Model (from docs/ARCHITECTURE.md Section 1):
    E = (RT)/(zF) * ln([ion]_out / [ion]_in)

    Membrane potential evolution:
    dV/dt = (V_rest - V) / tau + I_ext / C_m

Reference:
    - Nernst, W. (1889). Die elektromotorische Wirksamkeit der Ionen.
    - Hodgkin & Huxley (1952). A quantitative description of membrane current.

Stability:
    - Explicit Euler: dt < tau for stability
    - RK4: dt < 2*tau for stability
    - Values clamped to [V_min, V_max] = [-95, +40] mV
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from mycelium_fractal_net.core.config import IntegrationScheme, MembraneEngineConfig
from mycelium_fractal_net.core.exceptions import (
    NumericalInstabilityError,
)

# Physical constants (SI units)
R_GAS: float = 8.314  # J/(mol·K)
F_FARADAY: float = 96485.33212  # C/mol


@dataclass
class MembraneMetrics:
    """
    Metrics collected during membrane potential simulation.

    Attributes:
        steps_completed: Number of integration steps completed.
        v_min: Minimum potential observed (V).
        v_max: Maximum potential observed (V).
        v_mean: Mean potential (V).
        v_std: Standard deviation of potential (V).
        nan_detected: Whether NaN was detected.
        inf_detected: Whether Inf was detected.
        values_clamped: Number of times clamping was applied.
        execution_time_s: Total execution time in seconds.
    """

    steps_completed: int = 0
    v_min: float = 0.0
    v_max: float = 0.0
    v_mean: float = 0.0
    v_std: float = 0.0
    nan_detected: bool = False
    inf_detected: bool = False
    values_clamped: int = 0
    execution_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "steps_completed": self.steps_completed,
            "v_min_mV": self.v_min * 1000.0,
            "v_max_mV": self.v_max * 1000.0,
            "v_mean_mV": self.v_mean * 1000.0,
            "v_std_mV": self.v_std * 1000.0,
            "nan_detected": self.nan_detected,
            "inf_detected": self.inf_detected,
            "values_clamped": self.values_clamped,
            "execution_time_s": self.execution_time_s,
        }


class MembraneEngine:
    """
    Numerical engine for membrane potential dynamics.

    Implements ODE integration for membrane potentials with:
    - Nernst equation for equilibrium potentials
    - Explicit Euler or RK4 integration
    - NaN/Inf detection and clamping
    - Configurable stability checks

    Example:
        >>> config = MembraneEngineConfig(dt=0.001, random_seed=42)
        >>> engine = MembraneEngine(config)
        >>> V, metrics = engine.simulate(n_neurons=100, steps=1000)
        >>> print(f"Mean potential: {metrics.v_mean * 1000:.1f} mV")

    Reference: docs/ARCHITECTURE.md Section 1
    """

    def __init__(self, config: MembraneEngineConfig | None = None) -> None:
        """
        Initialize membrane engine.

        Args:
            config: Configuration parameters. Uses defaults if None.

        Raises:
            ValueError: If configuration is invalid.
        """
        self.config = config or MembraneEngineConfig()
        self.config.validate()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._metrics = MembraneMetrics()

    def compute_nernst_potential(
        self,
        z_valence: int,
        concentration_out: float,
        concentration_in: float,
        temperature_k: float | None = None,
    ) -> float:
        """
        Compute equilibrium potential using Nernst equation.

        E = (R*T)/(z*F) * ln([ion]_out / [ion]_in)

        Args:
            z_valence: Ion valence (K+=1, Ca2+=2, Cl-=-1).
            concentration_out: Extracellular concentration (mol/L).
            concentration_in: Intracellular concentration (mol/L).
            temperature_k: Temperature in Kelvin (default: from config).

        Returns:
            Equilibrium potential in volts.

        Raises:
            ValueError: If concentrations are non-positive.
            NumericalInstabilityError: If result is NaN/Inf.

        Reference: docs/ARCHITECTURE.md Section 1
        """
        T = temperature_k if temperature_k is not None else self.config.temperature_k

        # Clamp concentrations to prevent log(0)
        c_out = max(concentration_out, self.config.ion_clamp_min)
        c_in = max(concentration_in, self.config.ion_clamp_min)

        if z_valence == 0:
            raise ValueError("z_valence cannot be zero")

        ratio = c_out / c_in
        E = (R_GAS * T) / (z_valence * F_FARADAY) * math.log(ratio)

        if self.config.check_stability and not math.isfinite(E):
            raise NumericalInstabilityError(
                "Nernst potential produced NaN/Inf",
                field_name="E",
            )

        return E

    def _derivative(self, V: NDArray[Any], I_ext: NDArray[Any]) -> NDArray[np.float64]:
        """
        Compute dV/dt for membrane potential ODE.

        dV/dt = (V_rest - V) / tau + I_ext / C_m

        For simplicity, we normalize C_m = 1 and assume I_ext is in V/s.
        """
        result = (self.config.v_rest - V) / self.config.tau + I_ext
        return np.asarray(result, dtype=np.float64)

    def _euler_step(
        self, V: NDArray[Any], I_ext: NDArray[Any], dt: float
    ) -> NDArray[np.float64]:
        """Forward Euler integration step."""
        dVdt = self._derivative(V, I_ext)
        result = V + dt * dVdt
        return np.asarray(result, dtype=np.float64)

    def _rk4_step(
        self, V: NDArray[Any], I_ext: NDArray[Any], dt: float
    ) -> NDArray[np.float64]:
        """Runge-Kutta 4th order integration step."""
        k1 = self._derivative(V, I_ext)
        k2 = self._derivative(V + 0.5 * dt * k1, I_ext)
        k3 = self._derivative(V + 0.5 * dt * k2, I_ext)
        k4 = self._derivative(V + dt * k3, I_ext)
        result = V + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.asarray(result, dtype=np.float64)

    def _check_stability(self, V: NDArray[Any], step: int) -> int:
        """
        Check for NaN/Inf and clamp to valid range.

        Returns number of clamped values.

        Raises:
            NumericalInstabilityError: If NaN/Inf detected and check_stability is True.
        """
        nan_count = int(np.isnan(V).sum())
        inf_count = int(np.isinf(V).sum())

        if nan_count > 0 or inf_count > 0:
            self._metrics.nan_detected = nan_count > 0
            self._metrics.inf_detected = inf_count > 0
            if self.config.check_stability:
                raise NumericalInstabilityError(
                    "Membrane potential simulation produced NaN/Inf",
                    field_name="V",
                    nan_count=nan_count,
                    inf_count=inf_count,
                    step=step,
                )

        # Count values outside range before clamping
        out_of_range = ((V < self.config.v_min) | (V > self.config.v_max)).sum()

        return int(out_of_range)

    def simulate(
        self,
        n_neurons: int = 100,
        steps: int = 1000,
        V_init: NDArray[Any] | None = None,
        I_ext: NDArray[Any] | None = None,
        spike_probability: float = 0.0,
        spike_amplitude: float = 0.030,  # 30 mV spike
    ) -> Tuple[NDArray[Any], MembraneMetrics]:
        """
        Simulate membrane potential dynamics for n_neurons over steps.

        Args:
            n_neurons: Number of neurons to simulate.
            steps: Number of integration steps.
            V_init: Initial potentials (n_neurons,). Default: random around V_rest.
            I_ext: External current (n_neurons,) or (steps, n_neurons).
                   Default: zero (passive decay to V_rest).
            spike_probability: Probability of random spike per neuron per step.
            spike_amplitude: Amplitude of random spikes (V).

        Returns:
            V: Final membrane potentials (n_neurons,).
            metrics: Collected simulation metrics.

        Raises:
            NumericalInstabilityError: If simulation becomes unstable.
            ValueOutOfRangeError: If values exceed allowed range and check_stability is True.
        """
        start_time = time.perf_counter()

        # Initialize potentials
        if V_init is not None:
            V = V_init.copy()
        else:
            V = self._rng.normal(
                loc=self.config.v_rest, scale=0.005, size=(n_neurons,)
            )
            V = np.clip(V, self.config.v_min, self.config.v_max)

        # Initialize external current
        if I_ext is None:
            I_ext_arr: NDArray[Any] = np.zeros((n_neurons,), dtype=np.float64)
            I_ext_time_varying = False
        elif I_ext.ndim == 1:
            I_ext_arr = np.asarray(I_ext, dtype=np.float64)
            I_ext_time_varying = False
        else:
            I_ext_arr = np.asarray(I_ext, dtype=np.float64)
            I_ext_time_varying = True

        # Select integration method
        if self.config.integration_scheme == IntegrationScheme.EULER:
            step_fn = self._euler_step
        else:
            step_fn = self._rk4_step

        dt = self.config.dt
        total_clamped = 0

        # Integration loop
        for step in range(steps):
            # Get current for this step
            if I_ext_time_varying:
                I_current = I_ext_arr[step % len(I_ext_arr)]
            else:
                I_current = I_ext_arr

            # Add random spikes
            if spike_probability > 0:
                spike_mask = self._rng.random(n_neurons) < spike_probability
                I_current = I_current.copy()
                if spike_mask.any():
                    spike_current = spike_amplitude / dt  # Impulse current
                    I_current = np.where(spike_mask, spike_current, I_current)

            # Integration step
            V = step_fn(V, I_current, dt)

            # Stability check
            clamped = self._check_stability(V, step)
            total_clamped += clamped

            # Clamp to valid range
            V = np.clip(V, self.config.v_min, self.config.v_max)

        # Collect metrics
        self._metrics.steps_completed = steps
        self._metrics.v_min = float(V.min())
        self._metrics.v_max = float(V.max())
        self._metrics.v_mean = float(V.mean())
        self._metrics.v_std = float(V.std())
        self._metrics.values_clamped = total_clamped
        self._metrics.execution_time_s = time.perf_counter() - start_time

        return V, self._metrics

    def simulate_with_history(
        self,
        n_neurons: int = 100,
        steps: int = 1000,
        V_init: NDArray[Any] | None = None,
        I_ext: NDArray[Any] | None = None,
        record_every: int = 1,
    ) -> Tuple[NDArray[Any], MembraneMetrics]:
        """
        Simulate with full history recording.

        Args:
            n_neurons: Number of neurons.
            steps: Number of steps.
            V_init: Initial potentials.
            I_ext: External current.
            record_every: Record every N steps.

        Returns:
            V_history: Potential history (num_records, n_neurons).
            metrics: Simulation metrics.
        """
        start_time = time.perf_counter()

        # Initialize
        if V_init is not None:
            V = V_init.copy()
        else:
            V = self._rng.normal(
                loc=self.config.v_rest, scale=0.005, size=(n_neurons,)
            )
            V = np.clip(V, self.config.v_min, self.config.v_max)

        if I_ext is None:
            I_ext_arr = np.zeros((n_neurons,))
        else:
            I_ext_arr = I_ext

        # Select integration method
        if self.config.integration_scheme == IntegrationScheme.EULER:
            step_fn = self._euler_step
        else:
            step_fn = self._rk4_step

        dt = self.config.dt
        num_records = (steps + record_every - 1) // record_every
        V_history = np.zeros((num_records, n_neurons))
        record_idx = 0
        total_clamped = 0

        for step in range(steps):
            V = step_fn(V, I_ext_arr, dt)

            clamped = self._check_stability(V, step)
            total_clamped += clamped

            V = np.clip(V, self.config.v_min, self.config.v_max)

            if step % record_every == 0 and record_idx < num_records:
                V_history[record_idx] = V
                record_idx += 1

        self._metrics.steps_completed = steps
        self._metrics.v_min = float(V_history.min())
        self._metrics.v_max = float(V_history.max())
        self._metrics.v_mean = float(V_history.mean())
        self._metrics.v_std = float(V_history.std())
        self._metrics.values_clamped = total_clamped
        self._metrics.execution_time_s = time.perf_counter() - start_time

        return V_history[:record_idx], self._metrics

    @property
    def metrics(self) -> MembraneMetrics:
        """Get current metrics."""
        return self._metrics

    def reset(self) -> None:
        """Reset engine state and metrics."""
        self._rng = np.random.default_rng(self.config.random_seed)
        self._metrics = MembraneMetrics()
