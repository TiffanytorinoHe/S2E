"""Shared spatial-profile processing for phi-a figures and quantification.

The main-text spatial profile is intentionally shape based:

1. collapse each segment's time trace to a positive temporal peak;
2. average segment peaks within fixed distance bins;
3. normalize the distance-bin vector so the bins sum to one.

This separates spatial redistribution from whole-cell amplitude gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


N_DISTANCE_BINS = 8
EPS = 1e-12


@dataclass(frozen=True)
class SpatialProfile:
    bin_centers: np.ndarray
    q_mean: np.ndarray
    q_sem: np.ndarray
    Q_mean: np.ndarray
    Q_sem: np.ndarray
    q_by_seed: np.ndarray
    Q_by_seed: np.ndarray
    bin_edges: np.ndarray


def positive_temporal_peak(trace: np.ndarray, axis: int) -> np.ndarray:
    """Collapse a time trace to a non-negative peak along ``axis``."""
    peak = np.max(np.asarray(trace, dtype=np.float64), axis=axis)
    return np.maximum(peak, 0.0)


def distance_bin_edges(distance: np.ndarray, n_bins: int = N_DISTANCE_BINS) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    if distance.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    dmin = float(np.nanmin(distance))
    dmax = float(np.nanmax(distance))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        return np.linspace(0.0, float(max(distance.size, 1)), n_bins + 1)
    return np.linspace(dmin, dmax, n_bins + 1)


def bin_mean_profile(
    segment_values: np.ndarray,
    distance: np.ndarray,
    n_bins: int = N_DISTANCE_BINS,
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return distance-bin mean ``Q`` and normalized spatial profile ``q``."""
    values = np.asarray(segment_values, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    nseg = int(min(values.size, distance.size))
    values = values[:nseg]
    distance = distance[:nseg]
    edges = distance_bin_edges(distance, n_bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_id = np.digitize(distance, edges, right=False) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)

    Q = np.zeros(n_bins, dtype=np.float64)
    for idx in range(n_bins):
        mask = bin_id == idx
        if np.any(mask):
            Q[idx] = float(np.nanmean(values[mask]))
    q = Q / (float(np.sum(Q)) + eps)
    return Q, q, centers, edges


def mean_spatial_profile(
    values_by_seed: Iterable[np.ndarray],
    distance: np.ndarray,
    n_bins: int = N_DISTANCE_BINS,
    eps: float = EPS,
) -> SpatialProfile:
    """Compute seed-level Q/q profiles and return their seed mean."""
    Q_rows = []
    q_rows = []
    centers = None
    edges = None
    for values in values_by_seed:
        Q, q, centers, edges = bin_mean_profile(values, distance, n_bins=n_bins, eps=eps)
        if np.sum(Q) > 0:
            Q_rows.append(Q)
            q_rows.append(q)

    if centers is None or edges is None:
        edges = distance_bin_edges(np.asarray(distance), n_bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

    if not Q_rows:
        empty = np.zeros((0, n_bins), dtype=np.float64)
        zeros = np.zeros(n_bins, dtype=np.float64)
        return SpatialProfile(centers, zeros, zeros, zeros, zeros, empty, empty, edges)

    Q_by_seed = np.asarray(Q_rows, dtype=np.float64)
    q_by_seed = np.asarray(q_rows, dtype=np.float64)
    q_sem = np.std(q_by_seed, axis=0, ddof=1) / np.sqrt(q_by_seed.shape[0]) if q_by_seed.shape[0] > 1 else np.zeros(n_bins)
    Q_sem = np.std(Q_by_seed, axis=0, ddof=1) / np.sqrt(Q_by_seed.shape[0]) if Q_by_seed.shape[0] > 1 else np.zeros(n_bins)
    return SpatialProfile(
        bin_centers=np.asarray(centers, dtype=np.float64),
        q_mean=np.mean(q_by_seed, axis=0),
        q_sem=q_sem,
        Q_mean=np.mean(Q_by_seed, axis=0),
        Q_sem=Q_sem,
        q_by_seed=q_by_seed,
        Q_by_seed=Q_by_seed,
        bin_edges=np.asarray(edges, dtype=np.float64),
    )
