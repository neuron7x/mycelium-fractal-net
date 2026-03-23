"""Physarum polycephalum adaptive conductivity model.

Ref: Tero et al. (2007) J. Theor. Biol. 244:553-564
     Tero et al. (2010) Science 327:439-442

Core equations:
    Q_ij = D_ij * (p_i - p_j)                   [flux]
    sum_j D_ij * (p_i - p_j) = b_i              [Kirchhoff]
    dD_ij/dt = |Q_ij|^gamma - alpha * D_ij      [adaptation]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

__all__ = ["PhysarumConfig", "PhysarumEngine", "PhysarumState"]

DEFAULT_GAMMA = 1.0
DEFAULT_ALPHA = 0.01
DEFAULT_MU = 1.8
DEFAULT_I0 = 1.0


@dataclass(frozen=True)
class PhysarumConfig:
    gamma: float = DEFAULT_GAMMA
    alpha: float = DEFAULT_ALPHA
    mu: float = DEFAULT_MU
    use_sigmoid: bool = False
    dt: float = 0.01


@dataclass
class PhysarumState:
    D_h: np.ndarray
    D_v: np.ndarray
    p: np.ndarray
    Q_h: np.ndarray
    Q_v: np.ndarray
    u_h: np.ndarray
    u_v: np.ndarray
    step_count: int = 0

    def conductivity_map(self) -> np.ndarray:
        N = self.D_h.shape[0]
        c = np.zeros((N, N))
        c[:, :-1] += self.D_h
        c[:, 1:] += self.D_h
        c[:-1, :] += self.D_v
        c[1:, :] += self.D_v
        return c / 4.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conductivity_mean": float(np.mean(self.D_h) + np.mean(self.D_v)) / 2,
            "conductivity_max": float(max(np.max(self.D_h), np.max(self.D_v))),
            "pressure_range": float(np.max(self.p) - np.min(self.p)),
            "flux_max": float(max(np.max(np.abs(self.Q_h)), np.max(np.abs(self.Q_v)))),
            "step_count": self.step_count,
        }


class PhysarumEngine:
    def __init__(self, N: int, config: PhysarumConfig | None = None) -> None:
        self.N = N
        self.config = config or PhysarumConfig()

    def initialize(
        self,
        source_mask: np.ndarray,
        sink_mask: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> PhysarumState:
        N = self.N
        D_h = np.ones((N, N - 1), dtype=np.float64)
        D_v = np.ones((N - 1, N), dtype=np.float64)
        p = np.zeros((N, N), dtype=np.float64)
        Q_h = np.zeros((N, N - 1), dtype=np.float64)
        Q_v = np.zeros((N - 1, N), dtype=np.float64)
        state = PhysarumState(
            D_h=D_h,
            D_v=D_v,
            p=p,
            Q_h=Q_h,
            Q_v=Q_v,
            u_h=Q_h.copy(),
            u_v=Q_v.copy(),
        )
        b = self._build_source_vector(source_mask, sink_mask)
        state = self._solve_pressure(state, b)
        state = self._compute_flux(state)
        return state

    def step(
        self,
        state: PhysarumState,
        source_mask: np.ndarray,
        sink_mask: np.ndarray,
    ) -> PhysarumState:
        b = self._build_source_vector(source_mask, sink_mask)
        cfg = self.config
        if cfg.use_sigmoid:
            f_h = np.abs(state.Q_h) ** cfg.mu / (1.0 + np.abs(state.Q_h) ** cfg.mu)
            f_v = np.abs(state.Q_v) ** cfg.mu / (1.0 + np.abs(state.Q_v) ** cfg.mu)
        else:
            f_h = np.abs(state.Q_h) ** cfg.gamma
            f_v = np.abs(state.Q_v) ** cfg.gamma
        D_h_new = np.clip(state.D_h + cfg.dt * (f_h - cfg.alpha * state.D_h), 1e-8, None)
        D_v_new = np.clip(state.D_v + cfg.dt * (f_v - cfg.alpha * state.D_v), 1e-8, None)
        new_state = PhysarumState(
            D_h=D_h_new,
            D_v=D_v_new,
            p=state.p.copy(),
            Q_h=state.Q_h.copy(),
            Q_v=state.Q_v.copy(),
            u_h=state.u_h.copy(),
            u_v=state.u_v.copy(),
            step_count=state.step_count + 1,
        )
        new_state = self._solve_pressure(new_state, b)
        new_state = self._compute_flux(new_state)
        return new_state

    def _build_source_vector(
        self,
        source_mask: np.ndarray,
        sink_mask: np.ndarray,
    ) -> np.ndarray:
        N = self.N
        b = np.zeros(N * N, dtype=np.float64)
        src_idx = np.where(source_mask.ravel())[0]
        snk_idx = np.where(sink_mask.ravel())[0]
        if len(src_idx) > 0:
            b[src_idx] = DEFAULT_I0 / max(len(src_idx), 1)
        if len(snk_idx) > 0:
            b[snk_idx] = -DEFAULT_I0 / max(len(snk_idx), 1)
        b -= b.mean()
        return b

    def _solve_pressure(self, state: PhysarumState, b: np.ndarray) -> PhysarumState:
        N = self.N
        n_nodes = N * N
        L = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                node = i * N + j
                if j < N - 1:
                    d = state.D_h[i, j]
                    nb = i * N + j + 1
                    L[node, node] += d
                    L[node, nb] -= d
                    L[nb, nb] += d
                    L[nb, node] -= d
                if i < N - 1:
                    d = state.D_v[i, j]
                    nb = (i + 1) * N + j
                    L[node, node] += d
                    L[node, nb] -= d
                    L[nb, nb] += d
                    L[nb, node] -= d

        L_csr = L.tocsr()
        L_csr[0, :] = 0
        L_csr[0, 0] = 1.0
        b_fixed = b.copy()
        b_fixed[0] = 0.0
        p_flat, _ = cg(L_csr, b_fixed, atol=1e-8, maxiter=1000)
        state.p = p_flat.reshape(N, N)
        return state

    def _compute_flux(self, state: PhysarumState) -> PhysarumState:
        p = state.p
        state.Q_h = state.D_h * (p[:, :-1] - p[:, 1:])
        state.Q_v = state.D_v * (p[:-1, :] - p[1:, :])
        state.u_h = state.Q_h / np.clip(state.D_h, 1e-12, None)
        state.u_v = state.Q_v / np.clip(state.D_v, 1e-12, None)
        return state
