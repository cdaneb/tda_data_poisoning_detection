"""Persistent homology computation via ripser."""

from typing import Any

import numpy as np
import ripser


def compute_persistence(
    X: np.ndarray,
    maxdim: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute persistent homology for a point cloud.

    Args:
        X: Point cloud of shape (n_samples, n_features)
        maxdim: Maximum homology dimension (0=components, 1=loops, 2=voids)
        **kwargs: Additional arguments passed to ripser.ripser

    Returns:
        Full ripser result dict with keys 'dgms', 'cocycles', etc.
        dgms[0] = H0, dgms[1] = H1, ...
    """
    result = ripser.ripser(X, maxdim=maxdim, **kwargs)
    return result
