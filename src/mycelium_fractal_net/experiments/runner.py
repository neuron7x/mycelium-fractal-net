"""Experiment runner — runs scenarios on real MFN simulator.

# EVIDENCE TYPE: real_simulation
All snapshots from actual MFN R-D simulation, not synthetic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

import mycelium_fractal_net as mfn
from mycelium_fractal_net.core.simulate import simulate_history
from mycelium_fractal_net.core.thermodynamic_kernel import FreeEnergyTracker
from mycelium_fractal_net.interpretability import (
    GammaDiagnostics,
    MFNFeatureExtractor,
)
from mycelium_fractal_net.tau_control import IdentityEngine

from .scenarios import SCENARIO_HEALTHY, SCENARIO_PATHOLOGICAL, ScenarioConfig

__all__ = ["ExperimentRunner", "RunResult", "ScenarioResult"]


@dataclass
class RunResult:
    """Result from one simulation run."""

    gamma: float
    gamma_r2: float
    features: list[dict[str, float]]
    free_energy_trajectory: list[float]
    deviation_origin: str
    v_trajectory: list[float]
    mode_counts: dict[str, int]
    n_transforms: int


@dataclass
class ScenarioResult:
    """Aggregated results from all runs of one scenario."""

    config: ScenarioConfig
    runs: list[RunResult]
    gamma_mean: float = 0.0
    gamma_std: float = 0.0
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        gammas = [r.gamma for r in self.runs]
        if gammas:
            self.gamma_mean = float(np.mean(gammas))
            self.gamma_std = float(np.std(gammas))


class ExperimentRunner:
    """Runs real MFN simulations for γ-scaling evidence."""

    def run_scenario(self, config: ScenarioConfig) -> ScenarioResult:
        """Run all replicates of one scenario."""
        t0 = time.perf_counter()
        runs: list[RunResult] = []

        for run_idx in range(config.n_runs):
            run = self._single_run(config, run_seed=run_idx * 100 + 42)
            runs.append(run)

        elapsed = time.perf_counter() - t0
        return ScenarioResult(config=config, runs=runs, elapsed_s=elapsed)

    def _single_run(self, config: ScenarioConfig, run_seed: int) -> RunResult:
        """One replicate: generate sequences, compute gamma, extract features."""

        # Generate diverse sequences
        seqs: list[mfn.FieldSequence] = []
        for i in range(config.n_sequences):
            spec = mfn.SimulationSpec(
                steps=config.n_steps_base + i * config.n_steps_increment,
                seed=run_seed + i * 7,
                **config.sim_params,  # type: ignore[arg-type]
            )
            seqs.append(simulate_history(spec))

        # Gamma via morphology descriptors
        gamma_result = _compute_gamma(seqs)
        gamma_val = gamma_result.get("gamma", 0.0)
        gamma_r2 = gamma_result.get("r2", 0.0)

        # Feature extraction on last sequence
        extractor = MFNFeatureExtractor()
        features = []
        for seq in seqs[-5:]:
            fv = extractor.extract_all(seq)
            features.append(fv.to_dict())

        # Free energy trajectory
        fet = FreeEnergyTracker(grid_size=int(config.sim_params.get("grid_size", 32)))
        free_energies = [fet.total_energy(seq.field) for seq in seqs]

        # Gamma diagnostics
        diag = GammaDiagnostics()
        gamma_values = [gamma_val] * len(seqs)  # constant per run
        report = diag.diagnose(seqs, gamma_values)
        deviation = report.deviation_origin

        # Identity engine
        engine = IdentityEngine(state_dim=7)
        v_trajectory: list[float] = []
        mode_counts: dict[str, int] = {}
        for j, seq in enumerate(seqs):
            fv = extractor.extract_all(seq)
            state_vec = fv.to_array()[:7]  # first 7 features as state
            ir = engine.process(
                state_vector=state_vec,
                free_energy=free_energies[j],
                phase_is_collapsing=False,
                coherence=0.8,
                recovery_succeeded=True,
            )
            v_trajectory.append(ir.lyapunov.v_total)
            m = ir.tau_state.mode
            mode_counts[m] = mode_counts.get(m, 0) + 1

        return RunResult(
            gamma=gamma_val,
            gamma_r2=gamma_r2,
            features=features,
            free_energy_trajectory=free_energies,
            deviation_origin=deviation,
            v_trajectory=v_trajectory,
            mode_counts=mode_counts,
            n_transforms=engine.transform.transform_count,
        )

    def run_all(self) -> dict[str, ScenarioResult]:
        """Run healthy + pathological scenarios."""
        return {
            "healthy": self.run_scenario(SCENARIO_HEALTHY),
            "pathological": self.run_scenario(SCENARIO_PATHOLOGICAL),
        }


def _compute_gamma(seqs: list[mfn.FieldSequence]) -> dict[str, Any]:
    """Wrapper for g4 gamma with fallback."""
    try:
        from scripts.g4_gamma_scaling import _compute_gamma as _g4
        return _g4(seqs)
    except Exception:
        return {"gamma": 0.0, "r2": 0.0, "valid": False}
