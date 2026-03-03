"""Topological summaries from persistence diagrams."""

import numpy as np


def _finite_bars(dgm: np.ndarray) -> np.ndarray:
    """Filter out infinite bars (death == inf). Exclude from summary statistics.

    Only rows with finite death values are kept. Empty diagrams return unchanged.
    Expected behavior for empty or all-infinite diagrams:
    - max_persistence -> 0.0
    - persistence_count -> 0
    - persistence_entropy -> 0.0
    """
    if len(dgm) == 0:
        return dgm
    finite = np.isfinite(dgm[:, 1])
    return dgm[finite]


def max_persistence(dgm: np.ndarray) -> float:
    """Maximum persistence (death - birth) in a diagram. Infinite bars excluded.

    Returns 0.0 for empty diagrams or when all bars are infinite."""
    dgm = _finite_bars(dgm)
    if len(dgm) == 0:
        return 0.0
    persistences = dgm[:, 1] - dgm[:, 0]
    return float(np.max(persistences))


def persistence_count(dgm: np.ndarray, threshold: float = 0.1) -> int:
    """Number of persistence pairs above a given threshold. Infinite bars excluded.

    Returns 0 for empty diagrams or when all bars are infinite."""
    dgm = _finite_bars(dgm)
    if len(dgm) == 0:
        return 0
    persistences = dgm[:, 1] - dgm[:, 0]
    return int(np.sum(persistences > threshold))


def _to_scalar(value) -> float:
    """Coerce persim result to Python float (handles 0-d array, 1-d array)."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.ravel()
    return float(flat[0])


def persistence_entropy(dgm: np.ndarray) -> float:
    """Persistence entropy from persim. Infinite bars excluded.

    Returns 0.0 for empty diagrams or when all bars are infinite."""
    try:
        from persim.persistent_entropy import persistent_entropy
        dgm_finite = _finite_bars(dgm)
        if len(dgm_finite) == 0:
            return 0.0
        pe = persistent_entropy([dgm_finite])
        pe_arr = np.asarray(pe).ravel()
        return float(pe_arr[0]) if pe_arr.size else 0.0
    except ImportError:
        return float("nan")


def summarize_by_dimension(dgms: list, threshold: float = 0.1) -> dict:
    """Return dict: dim -> {max_persistence, count, entropy}."""
    out = {}
    for dim, dgm in enumerate(dgms):
        out[dim] = {
            "max_persistence": max_persistence(dgm),
            "count": persistence_count(dgm, threshold),
            "entropy": persistence_entropy(dgm),
        }
    return out
