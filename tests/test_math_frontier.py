"""Tests for Mathematical Frontier: TDA + Wasserstein + FIM + CE + RMT."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

import mycelium_fractal_net as mfn
from mycelium_fractal_net.analytics.causal_emergence import (
    compute_causal_emergence,
    discretize_turing_field,
    effective_information,
)
from mycelium_fractal_net.analytics.math_frontier import run_math_frontier
from mycelium_fractal_net.analytics.rmt_spectral import rmt_diagnostics
from mycelium_fractal_net.analytics.tda_ews import compute_tda, tda_ews_trajectory
from mycelium_fractal_net.analytics.wasserstein_geometry import (
    ot_basin_stability,
    wasserstein_distance,
    wasserstein_trajectory_speed,
)


@pytest.fixture(scope="module")
def seq() -> mfn.FieldSequence:
    return mfn.simulate(mfn.SimulationSpec(grid_size=32, steps=60, seed=42))


# ── TDA ──────────────────────────────────────────────────────────────────────


def test_tda_signature(seq: mfn.FieldSequence) -> None:
    sig = compute_tda(seq.field)
    assert hasattr(sig, "beta_0")
    assert sig.pattern_type in ("spots", "stripes", "labyrinth", "mixed", "indeterminate")


def test_tda_uniform() -> None:
    sig = compute_tda(np.ones((32, 32)))
    assert sig.beta_0 == 0
    assert sig.beta_1 == 0


def test_tda_trajectory(seq: mfn.FieldSequence) -> None:
    m = tda_ews_trajectory(seq.history, stride=10)
    assert "beta_0" in m
    assert "total_pers_0" in m
    assert len(m["beta_0"]) == len(m["timesteps"])


def test_tda_json(seq: mfn.FieldSequence) -> None:
    json.dumps(compute_tda(seq.field).to_dict())


# ── WASSERSTEIN ──────────────────────────────────────────────────────────────


def test_w2_self_zero(seq: mfn.FieldSequence) -> None:
    w = wasserstein_distance(seq.field, seq.field)
    assert w < 1e-6


def test_w2_different_positive(seq: mfn.FieldSequence) -> None:
    w = wasserstein_distance(seq.history[0], seq.history[-1])
    assert w > 0.0


def test_w2_trajectory(seq: mfn.FieldSequence) -> None:
    speeds = wasserstein_trajectory_speed(seq.history, stride=10)
    assert len(speeds) > 0
    assert np.all(np.isfinite(speeds))


def test_ot_basin_stability() -> None:
    s1 = mfn.simulate(mfn.SimulationSpec(grid_size=16, steps=20, seed=1))
    s2 = mfn.simulate(mfn.SimulationSpec(grid_size=16, steps=20, seed=2))
    membership = ot_basin_stability([s1.field, s2.field], [s1.field, s2.field])
    assert membership.shape == (2, 2)
    np.testing.assert_allclose(membership.sum(axis=1), 1.0, atol=1e-5)
    assert membership[0, 0] > 0.5


# ── CAUSAL EMERGENCE ─────────────────────────────────────────────────────────


def test_ei_structured_gt_random() -> None:
    tpm_s = np.eye(4) * 0.8 + np.ones((4, 4)) * 0.05
    tpm_s /= tpm_s.sum(axis=1, keepdims=True)
    tpm_r = np.random.default_rng(0).dirichlet(np.ones(4), 4)
    assert effective_information(tpm_s) > effective_information(tpm_r)


def test_ei_non_negative() -> None:
    tpm = np.random.default_rng(0).dirichlet(np.ones(5), 5)
    assert effective_information(tpm) >= 0.0


def test_ce_json() -> None:
    tpm = np.eye(4) * 0.7 + 0.075
    tpm /= tpm.sum(axis=1, keepdims=True)
    r = compute_causal_emergence(tpm)
    json.dumps(r.to_dict())


def test_discretize(seq: mfn.FieldSequence) -> None:
    s = discretize_turing_field(seq.field)
    assert 0 <= s <= 3


# ── RMT ──────────────────────────────────────────────────────────────────────


def test_rmt_basic() -> None:
    from mycelium_fractal_net.bio.memory_anonymization import GapJunctionDiffuser
    from mycelium_fractal_net.bio.physarum import PhysarumEngine

    s = mfn.simulate(mfn.SimulationSpec(grid_size=16, steps=20, seed=42))
    eng = PhysarumEngine(16)
    src = s.field > 0
    snk = s.field < -0.05
    phys = eng.initialize(src, snk)
    for _ in range(3):
        phys = eng.step(phys, src, snk)
    diff = GapJunctionDiffuser()
    L = diff.build_laplacian(phys.D_h, phys.D_v).toarray()
    diag = rmt_diagnostics(L)
    assert 0 <= diag.r_ratio <= 1
    assert diag.fiedler_value >= 0
    json.dumps(diag.to_dict())


def test_rmt_structured() -> None:
    """Physarum after adaptation should be structured (r < 0.45)."""
    from mycelium_fractal_net.bio.memory_anonymization import GapJunctionDiffuser
    from mycelium_fractal_net.bio.physarum import PhysarumEngine

    s = mfn.simulate(mfn.SimulationSpec(grid_size=16, steps=20, seed=42))
    eng = PhysarumEngine(16)
    src = s.field > 0
    snk = s.field < -0.05
    phys = eng.initialize(src, snk)
    for _ in range(10):
        phys = eng.step(phys, src, snk)
    L = GapJunctionDiffuser().build_laplacian(phys.D_h, phys.D_v).toarray()
    assert rmt_diagnostics(L).r_ratio < 0.45


# ── UNIFIED FRONTIER ─────────────────────────────────────────────────────────


def test_frontier_runs(seq: mfn.FieldSequence) -> None:
    report = run_math_frontier(seq, run_rmt=True)
    assert report.compute_time_ms > 0
    assert "[MATH]" in report.summary()
    json.dumps(report.to_dict())


def test_frontier_performance(seq: mfn.FieldSequence) -> None:
    t0 = time.perf_counter()
    run_math_frontier(seq, run_rmt=True)
    ms = (time.perf_counter() - t0) * 1000
    assert ms < 800, f"Too slow: {ms:.0f}ms"
