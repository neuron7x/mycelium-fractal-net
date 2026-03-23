"""Benchmark gates for bio/ hot paths. Fail on performance regression."""

from __future__ import annotations

import numpy as np
import pytest

import mycelium_fractal_net as mfn
from mycelium_fractal_net.bio import BioExtension
from mycelium_fractal_net.bio.memory import BioMemory, HDVEncoder
from mycelium_fractal_net.bio.physarum import PhysarumEngine


@pytest.fixture(scope="module")
def physarum_32():
    N = 32
    eng = PhysarumEngine(N)
    f = np.random.default_rng(0).standard_normal((N, N))
    src, snk = f > 0, f < -0.05
    state = eng.initialize(src, snk)
    for _ in range(3):
        state = eng.step(state, src, snk)
    return eng, state, src, snk


@pytest.fixture(scope="module")
def memory_200():
    enc = HDVEncoder(n_features=8, D=10000, seed=0)
    mem = BioMemory(enc, capacity=500)
    rng = np.random.default_rng(0)
    for _ in range(200):
        mem.store(enc.encode(rng.standard_normal(8)), fitness=rng.random(), params={})
    return mem, enc, enc.encode(rng.standard_normal(8))


def test_bench_physarum_step(benchmark, physarum_32) -> None:  # type: ignore[no-untyped-def]
    """GATE: Physarum step @ 32x32 < 5ms."""
    eng, state, src, snk = physarum_32
    benchmark(eng.step, state, src, snk)
    assert benchmark.stats["mean"] < 0.005


def test_bench_memory_query(benchmark, memory_200) -> None:  # type: ignore[no-untyped-def]
    """GATE: Memory query 200 episodes < 0.5ms."""
    mem, _, query = memory_200
    benchmark(mem.query, query, 5)
    assert benchmark.stats["mean"] < 0.0005


def test_bench_hdv_encode(benchmark) -> None:  # type: ignore[no-untyped-def]
    """GATE: HDV encode < 1ms."""
    enc = HDVEncoder(n_features=8, D=10000, seed=0)
    f = np.random.default_rng(0).standard_normal(8)
    benchmark(enc.encode, f)
    assert benchmark.stats["mean"] < 0.001


def test_bench_bio_step_16(benchmark) -> None:  # type: ignore[no-untyped-def]
    """GATE: BioExtension.step(1) @ 16x16 < 10ms."""
    seq = mfn.simulate(mfn.SimulationSpec(grid_size=16, steps=20, seed=42))
    bio = BioExtension.from_sequence(seq).step(n=1)
    benchmark(bio.step, 1)
    assert benchmark.stats["mean"] < 0.010
