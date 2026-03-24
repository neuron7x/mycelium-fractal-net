"""Regression tests for bugs found during bio/ code audit."""

from __future__ import annotations

import time

import numpy as np

from mycelium_fractal_net.bio.evolution import (
    DEFAULT_PARAMS,
    PARAM_BOUNDS,
    BioEvolutionOptimizer,
    params_to_bio_config,
)
from mycelium_fractal_net.bio.memory import BioMemory, HDVEncoder
from mycelium_fractal_net.bio.physarum import PhysarumEngine


def test_nan_params_safe() -> None:
    p = DEFAULT_PARAMS.copy()
    p[0] = float("nan")
    config = params_to_bio_config(p)
    assert np.isfinite(config.physarum.gamma)


def test_inf_params_safe() -> None:
    p = DEFAULT_PARAMS.copy()
    p[0] = float("inf")
    config = params_to_bio_config(p)
    assert config.physarum.gamma <= PARAM_BOUNDS[0, 1]


def test_physarum_step_performance() -> None:
    N = 32
    eng = PhysarumEngine(N)
    f = np.random.default_rng(0).standard_normal((N, N))
    src = f > 0
    snk = f < -0.1
    state = eng.initialize(src, snk)
    state = eng.step(state, src, snk)
    t0 = time.perf_counter()
    for _ in range(10):
        state = eng.step(state, src, snk)
    ms_per_step = (time.perf_counter() - t0) / 10 * 1000
    assert ms_per_step < 10.0, f"Physarum step too slow: {ms_per_step:.1f}ms"


def test_memory_query_vectorized() -> None:
    enc = HDVEncoder(n_features=8, D=10000, seed=0)
    mem = BioMemory(enc, capacity=500)
    rng = np.random.default_rng(0)
    for _ in range(200):
        mem.store(enc.encode(rng.standard_normal(8)), fitness=rng.random(), params={})
    query = enc.encode(rng.standard_normal(8))
    t0 = time.perf_counter()
    for _ in range(500):
        mem.query(query, k=5)
    ms_per = (time.perf_counter() - t0) / 500 * 1000
    assert ms_per < 1.0, f"query() too slow: {ms_per:.3f}ms"


def test_memory_query_correctness() -> None:
    enc = HDVEncoder(n_features=8, D=1000, seed=42)
    mem = BioMemory(enc, capacity=50)
    rng = np.random.default_rng(1)
    feats = [rng.standard_normal(8) for _ in range(20)]
    for i, feat in enumerate(feats):
        mem.store(enc.encode(feat), fitness=float(i) / 20, params={"i": float(i)})
    results = mem.query(enc.encode(feats[0]), k=3)
    assert results[0][0] > 0.9, f"Top sim={results[0][0]:.3f}"


def test_familiarity_range() -> None:
    enc = HDVEncoder(n_features=8, D=10000, seed=7)
    mem = BioMemory(enc, capacity=100)
    rng = np.random.default_rng(2)
    for _ in range(50):
        mem.store(enc.encode(rng.standard_normal(8)), fitness=rng.random(), params={})
    for _ in range(20):
        f = mem.superposition_familiarity(enc.encode(rng.standard_normal(8)))
        assert 0.0 <= f <= 1.0


def test_evolution_deterministic() -> None:
    opt = BioEvolutionOptimizer(grid_size=8, steps=8, bio_steps=2, seed=0)
    f1 = opt.evaluate(DEFAULT_PARAMS)
    f2 = opt.evaluate(DEFAULT_PARAMS)
    assert abs(f1 - f2) < 1e-10


def test_all_nan_params() -> None:
    p = np.full_like(DEFAULT_PARAMS, float("nan"))
    config = params_to_bio_config(p)
    for name in ["physarum", "anastomosis", "fhn", "chemotaxis", "dispersal"]:
        obj = getattr(config, name)
        for field_name in obj.__dataclass_fields__:
            val = getattr(obj, field_name)
            if isinstance(val, float):
                assert np.isfinite(val), f"NaN in {name}.{field_name}"
