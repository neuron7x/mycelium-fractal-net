from __future__ import annotations

from typing import Iterable

import numpy as np


def build_embedding(parts: Iterable[dict[str, float]]) -> tuple[float, ...]:
    values: list[float] = []
    for part in parts:
        for key in sorted(part):
            values.append(float(part[key]))
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.maximum(1.0, np.abs(arr))
    normalized = arr / scale
    return tuple(float(v) for v in normalized)
