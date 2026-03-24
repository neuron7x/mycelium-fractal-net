"""Calibrated performance regression gates for bio/ hot paths.

Gates are RELATIVE to benchmarks/bio_baseline.json (3x multiplier).
Prevents false failures across different runner speeds.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

import mycelium_fractal_net as mfn
from mycelium_fractal_net.bio import BioExtension
from mycelium_fractal_net.bio.memory import BioMemory, HDVEncoder
from mycelium_fractal_net.bio.physarum import PhysarumEngine

REGRESSION_MULTIPLIER = 3.0


def _measure_ms(fn: object, rounds: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()  # type: ignore[operator]
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()  # type: ignore[operator]
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def test_gate_physarum(bio_baseline: dict) -> None:  # type: ignore[type-arg]
    N = 32
    eng = PhysarumEngine(N)
    f = np.random.default_rng(0).standard_normal((N, N))
    src, snk = f > 0, f < -0.05
    state = eng.initialize(src, snk)
    for _ in range(3):
        state = eng.step(state, src, snk)
    measured = _measure_ms(lambda: eng.step(state, src, snk))
    gate = bio_baseline["physarum_step_32"]["median_ms"] * REGRESSION_MULTIPLIER
    assert measured <= gate, f"physarum REGRESSION: {measured:.1f}ms > {gate:.1f}ms"


def test_gate_memory_query(bio_baseline: dict) -> None:  # type: ignore[type-arg]
    enc = HDVEncoder(n_features=8, D=10000, seed=0)
    mem = BioMemory(enc, capacity=500)
    rng = np.random.default_rng(0)
    for _ in range(200):
        mem.store(enc.encode(rng.standard_normal(8)), fitness=rng.random(), params={})
    q = enc.encode(rng.standard_normal(8))
    # Warmup: trigger matrix rebuild before measurement
    mem.query(q, 5)
    measured = _measure_ms(lambda: mem.query(q, 5), rounds=50, warmup=5)
    gate = bio_baseline["memory_query_200"]["median_ms"] * REGRESSION_MULTIPLIER
    assert measured <= gate, f"memory REGRESSION: {measured:.2f}ms > {gate:.2f}ms"


def test_gate_hdv_encode(bio_baseline: dict) -> None:  # type: ignore[type-arg]
    enc = HDVEncoder(n_features=8, D=10000, seed=0)
    f = np.random.default_rng(0).standard_normal(8)
    measured = _measure_ms(lambda: enc.encode(f), rounds=100)
    gate = bio_baseline["hdv_encode"]["median_ms"] * REGRESSION_MULTIPLIER
    assert measured <= gate, f"encode REGRESSION: {measured:.2f}ms > {gate:.2f}ms"


def test_gate_bio_step(bio_baseline: dict) -> None:  # type: ignore[type-arg]
    seq = mfn.simulate(mfn.SimulationSpec(grid_size=16, steps=20, seed=42))
    bio = BioExtension.from_sequence(seq).step(n=1)
    measured = _measure_ms(lambda: bio.step(n=1), rounds=10, warmup=1)
    gate = bio_baseline["bio_step_16"]["median_ms"] * REGRESSION_MULTIPLIER
    assert measured <= gate, f"bio_step REGRESSION: {measured:.1f}ms > {gate:.1f}ms"
