"""TDA-based monitoring over sliding windows of streaming data."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..homology import compute_persistence
from ..summaries import summarize_by_dimension
from .window_buffer import WindowBuffer


class TDAMonitor:
    """Compute and track TDA features over a sliding window buffer."""

    def __init__(
        self,
        window_buffer: WindowBuffer,
        *,
        threshold: float = 3.0,
        warmup_windows: int = 5,
        k_consecutive: int = 2,
        point_cloud_mode: str = "features",
        tda_threshold: float = 0.1,
        maxdim: int = 1,
    ) -> None:
        self.window_buffer = window_buffer
        self.threshold = float(threshold)
        self.warmup_windows = int(warmup_windows)
        self.k_consecutive = int(k_consecutive)
        self.point_cloud_mode = str(point_cloud_mode)
        self.tda_threshold = float(tda_threshold)
        self.maxdim = int(maxdim)

        self.rows: List[Dict[str, Any]] = []
        self._streak: int = 0

    def compute_tda_features(self, point_cloud: np.ndarray) -> Dict[str, float]:
        """Compute flattened TDA summary features for a point cloud."""
        result = compute_persistence(point_cloud, maxdim=self.maxdim)
        dgms = result.get("dgms", [])
        summaries = summarize_by_dimension(dgms, threshold=self.tda_threshold) if dgms else {}

        def _get(dim: int, key: str) -> float:
            s = summaries.get(dim, {})
            val = s.get(key, 0.0)
            return float(val)

        features = {
            "h0_max_persistence": _get(0, "max_persistence"),
            "h0_count": _get(0, "count"),
            "h0_entropy": _get(0, "entropy"),
            "h1_max_persistence": _get(1, "max_persistence"),
            "h1_count": _get(1, "count"),
            "h1_entropy": _get(1, "entropy"),
        }
        return features

    def _compute_score(self, current: Dict[str, float]) -> float:
        """Compute z-score based detection score for the current row.

        Uses the first ``warmup_windows`` rows as a fixed baseline so that
        later windows are compared against an early, typically clean, regime.
        """
        n_rows = len(self.rows)
        if n_rows <= self.warmup_windows:
            # Explicitly indicate that no detection is available yet.
            return float("nan")

        baseline = self.rows[: self.warmup_windows]

        # Choose feature subset: default H1, fallback to H0 if H1 is degenerate now.
        if current["h1_max_persistence"] == 0.0 and current["h1_count"] == 0.0:
            feature_keys = ["h0_max_persistence"]
        else:
            feature_keys = ["h1_max_persistence", "h1_count"]

        zs = []
        eps = 1e-8
        for key in feature_keys:
            vals = np.array([float(r[key]) for r in baseline], dtype=float)
            mean = float(vals.mean()) if vals.size else 0.0
            std = float(vals.std(ddof=0)) if vals.size else 0.0
            if std < eps:
                std = eps
            z = abs((current[key] - mean) / std)
            zs.append(z)

        if not zs:
            return 0.0
        return float(max(zs))

    def update(self, t: int) -> Dict[str, Any]:
        """Fetch point cloud from the buffer, compute TDA features and detection."""
        point_cloud = self.window_buffer.get_point_cloud(mode=self.point_cloud_mode)
        features = self.compute_tda_features(point_cloud)

        row: Dict[str, Any] = {"t": int(t)}
        row.update(features)
        self.rows.append(row)

        score = self._compute_score(features)
        row["score"] = score

        if np.isnan(score):
            flag = False
            self._streak = 0
        else:
            if score > self.threshold:
                self._streak += 1
            else:
                self._streak = 0
            flag = self._streak >= self.k_consecutive

        row["flag"] = flag
        # Also update the stored row with the detection outputs.
        self.rows[-1] = row
        return row


