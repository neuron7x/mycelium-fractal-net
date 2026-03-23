"""Hyperdimensional Vector Memory — MAP model + Random Fourier Features.

Ref: Kanerva (2009) Cognitive Computation 1:139-159
     Rahimi & Recht (2007) NIPS — Random Fourier Features

D=10000: P(|sim(random,random)| > 0.1) ≈ 10^-50
Capacity: ~0.2*D = 2000 reliable memories in superposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["BioMemory", "HDVEncoder", "MemoryEntry"]

DEFAULT_D = 10_000


@dataclass
class MemoryEntry:
    hdv: np.ndarray
    fitness: float
    params: dict[str, float]
    metadata: dict[str, Any]
    step: int = 0


class HDVEncoder:
    """Float vector → ±1 hypervector via Random Fourier Features."""

    def __init__(
        self, n_features: int, D: int = DEFAULT_D, sigma: float = 1.0, seed: int = 42
    ) -> None:
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = 1.0
        self.n_features = n_features
        self.D = D
        rng = np.random.default_rng(seed)
        self._omega = rng.standard_normal((D, n_features)) / sigma
        self._b = rng.uniform(0, 2 * np.pi, D)

    def encode(self, features: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(
            np.asarray(features, dtype=np.float64).ravel(),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        padded = np.zeros(self.n_features)
        n = min(len(x), self.n_features)
        padded[:n] = x[:n]
        padded = np.clip(padded, -1e6, 1e6)
        projection = self._omega @ padded + self._b
        projection = np.nan_to_num(projection, nan=0.0, posinf=np.pi, neginf=-np.pi)
        raw = np.sign(np.cos(projection))
        # Guarantee ±1 output (sign(0)=0 → map to +1)
        return np.where(raw == 0.0, 1.0, raw).astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b)) / self.D


class BioMemory:
    """Episodic memory with O(D) superposition familiarity check."""

    def __init__(self, encoder: HDVEncoder, capacity: int = 1000) -> None:
        self.encoder = encoder
        self.capacity = capacity
        self._episodes: list[MemoryEntry] = []
        self._superposition = np.zeros(encoder.D, dtype=np.float64)
        self._total_stored = 0
        self._hdv_matrix: np.ndarray | None = None
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._episodes)

    @property
    def is_empty(self) -> bool:
        return len(self._episodes) == 0

    def store(
        self,
        hdv: np.ndarray,
        fitness: float,
        params: dict[str, float],
        metadata: dict[str, Any] | None = None,
        step: int = 0,
    ) -> None:
        entry = MemoryEntry(
            hdv=hdv.copy(),
            fitness=float(fitness),
            params=dict(params),
            metadata=metadata or {},
            step=step,
        )
        if len(self._episodes) >= self.capacity:
            old = self._episodes.pop(0)
            self._superposition -= old.hdv.astype(np.float64)
            self._dirty = True  # eviction: full rebuild needed
        elif self._hdv_matrix is not None and not self._dirty:
            # Append-only fast path: extend matrix without full rebuild
            new_row = hdv.astype(np.float32).reshape(1, -1)
            self._hdv_matrix = np.vstack([self._hdv_matrix, new_row])
        else:
            self._dirty = True
        self._episodes.append(entry)
        self._superposition += hdv.astype(np.float64)
        self._total_stored += 1
        self._dirty = True

    def _rebuild_matrix(self) -> None:
        if not self._episodes:
            self._hdv_matrix = None
            return
        self._hdv_matrix = np.stack([ep.hdv for ep in self._episodes], axis=0).astype(np.float64)

    def query(
        self, query_hdv: np.ndarray, k: int = 5
    ) -> list[tuple[float, float, dict[str, float], dict[str, Any]]]:
        if self.is_empty:
            return []
        if self._dirty:
            self._rebuild_matrix()
            self._dirty = False
        if self._hdv_matrix is None:
            return []
        sims = (self._hdv_matrix @ query_hdv.astype(np.float64)) / self.encoder.D
        n = min(k, len(sims))
        if n >= len(sims):
            top_idx = np.argsort(sims)[::-1][:n]
        else:
            top_idx = np.argpartition(sims, -n)[-n:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [
            (
                float(sims[i]),
                self._episodes[i].fitness,
                self._episodes[i].params,
                self._episodes[i].metadata,
            )
            for i in top_idx
        ]

    def superposition_familiarity(self, query_hdv: np.ndarray) -> float:
        if self.is_empty:
            return 0.0
        sp_norm = self._superposition / (np.linalg.norm(self._superposition) + 1e-12)
        raw = float(np.dot(query_hdv, sp_norm)) / self.encoder.D
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    def predict_fitness(self, query_hdv: np.ndarray, k: int = 5) -> float:
        if self.is_empty:
            return 0.0
        results = self.query(query_hdv, k=min(k, self.size))
        sims = np.array([r[0] for r in results])
        fits = np.array([r[1] for r in results])
        weights = np.exp(sims - sims.max())
        weights /= weights.sum() + 1e-12
        return float(np.dot(weights, fits))

    def best_known_fitness(self) -> float:
        return max((ep.fitness for ep in self._episodes), default=0.0)

    def best_known_params(self) -> dict[str, float]:
        if self.is_empty:
            return {}
        return dict(max(self._episodes, key=lambda e: e.fitness).params)

    def fitness_landscape(self) -> dict[str, float]:
        if self.is_empty:
            return {}
        f = [ep.fitness for ep in self._episodes]
        return {
            "mean": float(np.mean(f)),
            "std": float(np.std(f)),
            "min": float(np.min(f)),
            "max": float(np.max(f)),
            "count": len(f),
            "total_stored": self._total_stored,
        }
