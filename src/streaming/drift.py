"""Natural concept drift transforms for streaming features."""

from __future__ import annotations

from typing import List

import numpy as np


def apply_natural_drift(
    x: np.ndarray,
    t: int,
    *,
    drift_start: int,
    drift_end: int,
    dims: List[int],
    magnitude: float,
) -> np.ndarray:
    """Apply a simple linear ramp drift to selected dimensions.

    Outside the ``[drift_start, drift_end]`` window the input is returned
    unchanged (same values). Inside the window we apply a linear ramp:

    ``alpha = (t - drift_start) / max(1, drift_end - drift_start)``

    and for each selected dimension ``d`` we add ``alpha * magnitude`` to
    ``x[d]`` when ``0 <= d < len(x)``.
    """
    x_arr = np.asarray(x, dtype=float).ravel()

    if t < drift_start or t > drift_end:
        # Outside the drift window: no change in values.
        return x_arr

    denom = max(1, int(drift_end) - int(drift_start))
    alpha = (int(t) - int(drift_start)) / float(denom)

    x_mod = x_arr.copy()
    n = x_mod.shape[0]
    for d in dims:
        if isinstance(d, int) and 0 <= d < n:
            x_mod[d] = x_mod[d] + alpha * float(magnitude)

    return x_mod

