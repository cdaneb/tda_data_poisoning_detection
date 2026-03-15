"""Topological summaries from persistence diagrams."""

import numpy as np

# Grid resolution for landscape and Betti curve discretization.
_LANDSCAPE_BETTI_GRID = 200


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


def wasserstein_amplitude(dgm: np.ndarray) -> float:
    """L1 Wasserstein distance from the empty diagram (sum of persistences).

    Equivalent to sum of (death - birth) over finite bars. Returns 0.0 for empty
    or all-infinite diagrams. No extra libraries; computed from raw diagram."""
    dgm = _finite_bars(dgm)
    if len(dgm) == 0:
        return 0.0
    persistences = dgm[:, 1] - dgm[:, 0]
    return float(np.sum(persistences))


def _landscape_l1_at_t(t: float, birth: float, death: float) -> float:
    """Height of the first landscape (tent) for one bar at filtration value t."""
    if t <= birth or t >= death:
        return 0.0
    return float(min(t - birth, death - t))


def landscape_amplitude(dgm: np.ndarray, n_grid: int = _LANDSCAPE_BETTI_GRID) -> float:
    """L2 norm of the first persistence landscape.

    Discretizes the filtration axis and computes lambda_1(t) = max over bars
    of tent height at t, then returns sqrt(integral lambda_1(t)^2 dt).
    Returns 0.0 for empty or all-infinite diagrams. No giotto-tda."""
    dgm = _finite_bars(dgm)
    if len(dgm) == 0:
        return 0.0
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    t_min = float(np.min(births))
    t_max = float(np.max(deaths))
    if t_max <= t_min:
        return 0.0
    t_grid = np.linspace(t_min, t_max, num=n_grid, endpoint=True)
    dt = (t_max - t_min) / max(1, n_grid - 1)
    lambda_1 = np.zeros(n_grid)
    for i, t in enumerate(t_grid):
        for b, d in zip(births, deaths):
            if b <= t < d:
                h = _landscape_l1_at_t(float(t), float(b), float(d))
                if h > lambda_1[i]:
                    lambda_1[i] = h
    return float(np.sqrt(np.sum(lambda_1 ** 2) * dt))


def betti_curve_mean(dgm: np.ndarray, n_grid: int = _LANDSCAPE_BETTI_GRID) -> float:
    """Mean of the Betti curve across filtration values.

    Betti curve beta(t) = number of bars with birth <= t < death. Returns the
    mean of beta(t) over a uniform grid. Returns 0.0 for empty diagrams."""
    dgm = _finite_bars(dgm)
    if len(dgm) == 0:
        return 0.0
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    t_min = float(np.min(births))
    t_max = float(np.max(deaths))
    if t_max <= t_min:
        return 0.0
    t_grid = np.linspace(t_min, t_max, num=n_grid, endpoint=True)
    betti = np.zeros(n_grid)
    for i, t in enumerate(t_grid):
        betti[i] = np.sum((births <= t) & (t < deaths))
    return float(np.mean(betti))


def summarize_by_dimension(dgms: list, threshold: float = 0.1) -> dict:
    """Return dict: dim -> {max_persistence, count, entropy, wasserstein_amplitude,
    landscape_amplitude, betti_curve_mean}."""
    out = {}
    for dim, dgm in enumerate(dgms):
        out[dim] = {
            "max_persistence": max_persistence(dgm),
            "count": persistence_count(dgm, threshold),
            "entropy": persistence_entropy(dgm),
            "wasserstein_amplitude": wasserstein_amplitude(dgm),
            "landscape_amplitude": landscape_amplitude(dgm),
            "betti_curve_mean": betti_curve_mean(dgm),
        }
    return out
