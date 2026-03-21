from __future__ import annotations

import numpy as np

from mycelium_fractal_net.types.field import FieldSequence


def compute_connectivity_features(sequence: FieldSequence) -> dict[str, float]:
    field = sequence.field.astype(np.float64)
    threshold = float(np.mean(field) + 0.5 * np.std(field))
    active = field > threshold
    active_ratio = float(np.mean(active))

    north = active[:-1, :] & active[1:, :]
    east = active[:, :-1] & active[:, 1:]
    degree_sum = 2.0 * (float(np.sum(north)) + float(np.sum(east)))
    node_count = max(1.0, float(np.sum(active)))
    gbc_like = degree_sum / node_count

    row_strength = np.mean(active, axis=1)
    col_strength = np.mean(active, axis=0)
    hierarchy_flattening = float(1.0 - (np.std(row_strength) + np.std(col_strength)) / 2.0)

    modularity_proxy = float(np.mean(np.abs(np.diff(row_strength))) + np.mean(np.abs(np.diff(col_strength))))

    if sequence.history is not None and sequence.history.shape[0] >= 2:
        frames = sequence.history.astype(np.float64)
        time_active = frames > (np.mean(frames, axis=(1, 2), keepdims=True) + 0.5 * np.std(frames, axis=(1, 2), keepdims=True))
        coherence = np.mean(time_active, axis=(1, 2))
        global_coherence_shift = float(np.max(coherence) - np.min(coherence))
        connectivity_divergence = float(np.mean(np.abs(np.diff(coherence))))
    else:
        global_coherence_shift = 0.0
        connectivity_divergence = 0.0

    return {
        'gbc_like_summary': gbc_like,
        'modularity_proxy': modularity_proxy,
        'hierarchy_flattening': hierarchy_flattening,
        'global_coherence_shift': global_coherence_shift,
        'connectivity_divergence': connectivity_divergence,
        'active_ratio': active_ratio,
    }
