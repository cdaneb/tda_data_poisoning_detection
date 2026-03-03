"""Baseline calibration utilities for TDA-based anomaly scores.

This module provides a small, streaming-friendly calibration layer that can be
fit on an initial clean baseline slice and then reused for subsequent windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


def empirical_quantile_threshold(scores: Sequence[float], quantile: float) -> float:
    """Return the empirical quantile of a 1D collection of scores.

    This is a thin wrapper around ``np.quantile`` to make the behavior explicit
    and easily testable.
    """
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute empirical quantile on an empty score sequence.")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be in (0, 1).")
    return float(np.quantile(arr, float(quantile)))


@dataclass
class BaselineCalibrator:
    """Feature-wise baseline calibration using robust or classic z-scores.

    The calibrator is fit on a collection of baseline feature vectors
    (e.g. warmup windows) and then used to transform subsequent feature
    vectors into normalized coordinates.
    """

    feature_names: List[str]
    mode: str = "robust_z"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.mode = str(self.mode)
        if self.mode not in ("robust_z", "classic_z"):
            raise ValueError(f"Unsupported baseline mode: {self.mode!r}")
        self._centers: np.ndarray | None = None
        self._scales: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._centers is not None and self._scales is not None

    def fit(self, baseline_rows: Iterable[Dict[str, float]]) -> None:
        """Fit baseline statistics from an iterable of feature dicts."""
        rows_list = list(baseline_rows)
        if not rows_list:
            raise ValueError("BaselineCalibrator.fit requires at least one baseline row.")

        X = np.asarray(
            [[float(row.get(name, 0.0)) for name in self.feature_names] for row in rows_list],
            dtype=float,
        )

        if self.mode == "robust_z":
            centers = np.median(X, axis=0)
            mad = np.median(np.abs(X - centers[None, :]), axis=0)
            scales = 1.4826 * mad
        else:
            centers = X.mean(axis=0)
            scales = X.std(axis=0, ddof=0)

        # Guard against degenerate scales which would produce NaNs/Infs.
        scales = np.where(scales < self.eps, self.eps, scales)

        self._centers = centers.astype(float)
        self._scales = scales.astype(float)

    def transform_dict(self, row: Dict[str, float]) -> np.ndarray:
        """Transform a single feature dict into normalized coordinates."""
        if not self.is_fitted:
            raise RuntimeError("BaselineCalibrator must be fit before calling transform_dict.")
        assert self._centers is not None
        assert self._scales is not None

        x = np.asarray(
            [float(row.get(name, 0.0)) for name in self.feature_names],
            dtype=float,
        )
        z = (x - self._centers) / self._scales
        return z

    def to_params(self) -> Dict[str, object]:
        """Return a JSON-serializable snapshot of the calibration parameters."""
        return {
            "feature_names": list(self.feature_names),
            "mode": self.mode,
            "eps": float(self.eps),
            "centers": None if self._centers is None else self._centers.tolist(),
            "scales": None if self._scales is None else self._scales.tolist(),
        }

