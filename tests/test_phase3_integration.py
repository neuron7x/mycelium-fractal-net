"""Phase 3 integration tests — real MFN simulation evidence for PRR.

# EVIDENCE TYPE: real_simulation
All tests run actual MFN R-D simulations.
"""

from __future__ import annotations

import numpy as np

from mycelium_fractal_net.experiments.prr_export import PRRExporter
from mycelium_fractal_net.experiments.runner import ExperimentRunner
from mycelium_fractal_net.experiments.scenarios import (
    SCENARIO_HEALTHY,
    SCENARIO_PATHOLOGICAL,
    ScenarioConfig,
)


def _quick_scenario(base: ScenarioConfig, n_runs: int = 3, n_seq: int = 8) -> ScenarioConfig:
    """Reduce run count for CI speed."""
    return ScenarioConfig(
        name=base.name,
        sim_params=base.sim_params,
        n_steps_base=base.n_steps_base,
        n_steps_increment=base.n_steps_increment,
        n_sequences=n_seq,
        n_runs=n_runs,
        expected_gamma_range=base.expected_gamma_range,
        description=base.description,
    )


class TestPhase3:
    def test_healthy_gamma_in_range(self) -> None:
        """Healthy scenario produces gamma in expected range."""
        runner = ExperimentRunner()
        result = runner.run_scenario(_quick_scenario(SCENARIO_HEALTHY, n_runs=3))
        # Healthy MFN: gamma should be negative (cost-efficient scaling)
        assert result.gamma_mean < 0, f"Healthy gamma should be < 0, got {result.gamma_mean}"
        assert len(result.runs) == 3

    def test_pathological_gamma_differs(self) -> None:
        """Pathological scenario produces different gamma."""
        runner = ExperimentRunner()
        h = runner.run_scenario(_quick_scenario(SCENARIO_HEALTHY, n_runs=3))
        p = runner.run_scenario(_quick_scenario(SCENARIO_PATHOLOGICAL, n_runs=3))
        # Gamma values should differ between conditions
        assert abs(h.gamma_mean - p.gamma_mean) > 0.5, (
            f"Insufficient separation: healthy={h.gamma_mean:.3f} patho={p.gamma_mean:.3f}"
        )

    def test_deviation_origin_populated(self) -> None:
        """Deviation origin is not empty for all runs."""
        runner = ExperimentRunner()
        result = runner.run_scenario(_quick_scenario(SCENARIO_HEALTHY, n_runs=2))
        for run in result.runs:
            assert run.deviation_origin != ""

    def test_free_energy_available(self) -> None:
        """Real F from FreeEnergyTracker is available.
        # FRISTON STATUS: PARTIAL — F available but not full Friston proof.
        """
        runner = ExperimentRunner()
        result = runner.run_scenario(_quick_scenario(SCENARIO_HEALTHY, n_runs=1))
        fe = result.runs[0].free_energy_trajectory
        assert len(fe) > 0
        assert all(np.isfinite(f) for f in fe)

    def test_lyapunov_on_real_data(self) -> None:
        """V trajectory from real simulation."""
        runner = ExperimentRunner()
        result = runner.run_scenario(_quick_scenario(SCENARIO_HEALTHY, n_runs=1))
        v = result.runs[0].v_trajectory
        assert len(v) > 0
        assert all(np.isfinite(vi) for vi in v)

    def test_identity_engine_no_transforms_healthy(self) -> None:
        """Healthy scenario: identity engine stays in IDLE/RECOVERY, no transforms."""
        runner = ExperimentRunner()
        result = runner.run_scenario(_quick_scenario(SCENARIO_HEALTHY, n_runs=2))
        total_transforms = sum(r.n_transforms for r in result.runs)
        assert total_transforms == 0, f"Unexpected {total_transforms} transforms in healthy"

    def test_prr_tables_generated(self) -> None:
        """All 5 PRR tables are non-empty."""
        runner = ExperimentRunner()
        h = runner.run_scenario(_quick_scenario(SCENARIO_HEALTHY, n_runs=2))
        p = runner.run_scenario(_quick_scenario(SCENARIO_PATHOLOGICAL, n_runs=2))

        exporter = PRRExporter()
        report = exporter.export(h, p)

        assert len(report.table_1) > 50
        assert len(report.table_2) > 50
        assert len(report.table_3) > 50
        assert len(report.table_4) > 20
        assert len(report.table_5) > 50
        assert report.evidence_type == "real_simulation"
