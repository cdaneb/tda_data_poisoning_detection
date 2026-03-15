"""TDA-based monitoring over sliding windows of streaming data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..homology import compute_persistence
from ..summaries import summarize_by_dimension
from .baseline import BaselineCalibrator, empirical_quantile_threshold
from .window_buffer import WindowBuffer


class TDAMonitor:
    """Compute and track TDA features over a sliding window buffer.

    This class encapsulates persistent-homology feature extraction together
    with a streaming-safe baseline calibration and thresholding scheme.
    """

    def __init__(
        self,
        window_buffer: WindowBuffer,
        *,
        threshold: float = 3.0,
        warmup_windows: int = 5,
        calibration_windows: int = 50,
        k_consecutive: int = 2,
        point_cloud_mode: str = "features",
        tda_threshold: float = 0.1,
        maxdim: int = 1,
        # Dimensionality reduction.
        dr_method: str = "pca",
        pca_n_components: Optional[int] = 15,
        pca_variance: Optional[float] = None,
        pca_max_components: int = 20,
        pca_max_points_per_window: int = 50,
        pca_strict: bool = False,
        # Baseline and thresholding.
        baseline_mode: str = "robust_z",
        threshold_mode: str = "empirical_quantile",
        threshold_quantile: float = 0.995,
        score_mode: str = "l2",
        score_from: str = "h1_then_h0",
        random_state: Optional[int] = None,
    ) -> None:
        self.window_buffer = window_buffer
        self.threshold = float(threshold)
        self.warmup_windows = int(warmup_windows)
        self.calibration_windows = int(calibration_windows)
        self.k_consecutive = int(k_consecutive)
        self.point_cloud_mode = str(point_cloud_mode)
        self.tda_threshold = float(tda_threshold)
        self.maxdim = int(maxdim)

        # Dimensionality reduction configuration.
        self.dr_method = str(dr_method)
        self.pca_n_components = pca_n_components
        self.pca_variance = pca_variance
        self.pca_max_components = int(pca_max_components)
        self.pca_max_points_per_window = int(pca_max_points_per_window)
        self.pca_strict = bool(pca_strict)
        # Track requested vs effective PCA dimensionality for later reporting.
        self.pca_n_components_requested: Optional[int] = (
            int(pca_n_components) if pca_n_components is not None else None
        )

        # Baseline / threshold configuration.
        self.baseline_mode = str(baseline_mode)
        self.threshold_mode = str(threshold_mode)
        self.threshold_quantile = float(threshold_quantile)
        self.score_mode = str(score_mode)
        self.score_from = str(score_from)

        self.rows: List[Dict[str, Any]] = []
        self._consecutive_flags: int = 0

        # RNG for any stochastic components (e.g. PCA subsampling).
        self._rng = np.random.default_rng(random_state)

        # Calibration state (12 features: 6 per dimension).
        self._feature_names = [
            "h0_max_persistence",
            "h0_count",
            "h0_entropy",
            "h0_wasserstein_amplitude",
            "h0_landscape_amplitude",
            "h0_betti_curve_mean",
            "h1_max_persistence",
            "h1_count",
            "h1_entropy",
            "h1_wasserstein_amplitude",
            "h1_landscape_amplitude",
            "h1_betti_curve_mean",
        ]
        self._baseline_rows: List[Dict[str, float]] = []
        self._calibration_scores: List[float] = []
        self._calibrator: Optional[BaselineCalibrator] = None
        self._threshold_from_quantile: Optional[float] = None

        # PCA state.
        self._pca = None  # type: ignore[assignment]
        self._pca_samples: List[np.ndarray] = []
        self._pca_disabled: bool = False
        self.pca_clamped: bool = False
        self.pca_n_components_eff: Optional[int] = None

        # H1 empty statistics.
        self._h1_empty_count: int = 0

    @property
    def h1_nonempty_frequency(self) -> float:
        """Return the fraction of windows with non-empty H1 (by count)."""
        total = len(self.rows)
        if total == 0:
            return 0.0
        return float((total - self._h1_empty_count) / float(total))

    def _maybe_fit_pca(self) -> None:
        if self.dr_method != "pca" or self._pca is not None or self._pca_disabled:
            return
        if not self._pca_samples:
            return
        from sklearn.decomposition import PCA

        X = np.vstack(self._pca_samples)

        n_samples, n_features = X.shape
        max_allowed = min(
            int(n_samples),
            int(n_features),
            max(1, int(self.window_buffer.window_size) - 1),
        )

        # If the data is too small to safely reduce dimension, disable PCA for
        # this monitor instance and fall back to identity mapping.
        if max_allowed < 2:
            self._pca_disabled = True
            self._pca = None
            self.pca_n_components_eff = None
            return

        # Determine requested dimensionality.
        if self.pca_n_components is not None:
            requested = int(self.pca_n_components)
            if requested > max_allowed:
                if self.pca_strict:
                    raise ValueError(
                        "Requested pca_n_components exceeds the maximum allowed for the "
                        f"available data and window_size (requested={requested}, "
                        f"max_allowed={max_allowed})."
                    )
                requested = max_allowed
                self.pca_clamped = True
            n_components_arg: Any = requested
        else:
            # Variance-based or full-dim PCA; delegate dimensionality choice to
            # sklearn and clip after fitting.
            n_components_arg = (
                float(self.pca_variance) if self.pca_variance is not None else None
            )

        pca = PCA(n_components=n_components_arg, random_state=None)
        pca.fit(X)

        # Effective dimensionality as determined by sklearn.
        actual_n = int(getattr(pca, "n_components_", X.shape[1]))

        # Apply configured caps: first by user-provided maximum components,
        # then by max_allowed from data + window_size.
        if actual_n > self.pca_max_components:
            actual_n = self.pca_max_components
        if actual_n > max_allowed:
            if self.pca_strict:
                raise ValueError(
                    "Fitted PCA dimensionality exceeds the maximum allowed for the "
                    f"available data and window_size (n_components={actual_n}, "
                    f"max_allowed={max_allowed})."
                )
            actual_n = max_allowed

        # Truncate PCA parameters if clipping was required.
        if hasattr(pca, "components_") and pca.components_.shape[0] != actual_n:
            pca.components_ = pca.components_[:actual_n]
        if hasattr(pca, "explained_variance_") and pca.explained_variance_.shape[0] != actual_n:
            pca.explained_variance_ = pca.explained_variance_[:actual_n]
        if hasattr(pca, "explained_variance_ratio_") and pca.explained_variance_ratio_.shape[0] != actual_n:
            pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:actual_n]

        self._pca = pca
        self.pca_n_components_eff = int(actual_n)

    def _maybe_accumulate_pca_samples(self, point_cloud: np.ndarray) -> None:
        if self.dr_method != "pca" or self._pca_disabled:
            return
        if point_cloud.size == 0:
            return
        n_points = point_cloud.shape[0]
        k = min(self.pca_max_points_per_window, n_points)
        if k <= 0:
            return
        idx = self._rng.choice(n_points, size=k, replace=False)
        self._pca_samples.append(point_cloud[idx])

    def _apply_dr(self, point_cloud: np.ndarray) -> np.ndarray:
        if self.dr_method != "pca" or self._pca_disabled:
            return point_cloud
        if self._pca is None:
            return point_cloud
        return self._pca.transform(point_cloud)

    def compute_tda_features(self, point_cloud: np.ndarray) -> Dict[str, float]:
        """Compute flattened TDA summary features for a point cloud."""
        point_cloud = self._apply_dr(point_cloud)
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
            "h0_wasserstein_amplitude": _get(0, "wasserstein_amplitude"),
            "h0_landscape_amplitude": _get(0, "landscape_amplitude"),
            "h0_betti_curve_mean": _get(0, "betti_curve_mean"),
            "h1_max_persistence": _get(1, "max_persistence"),
            "h1_count": _get(1, "count"),
            "h1_entropy": _get(1, "entropy"),
            "h1_wasserstein_amplitude": _get(1, "wasserstein_amplitude"),
            "h1_landscape_amplitude": _get(1, "landscape_amplitude"),
            "h1_betti_curve_mean": _get(1, "betti_curve_mean"),
        }

        if features["h1_count"] == 0.0:
            self._h1_empty_count += 1
        return features

    def _ensure_calibrator(self) -> None:
        if self._calibrator is not None:
            return
        calib = BaselineCalibrator(feature_names=list(self._feature_names), mode=self.baseline_mode)
        calib.fit(self._baseline_rows)
        self._calibrator = calib

    def _compute_normalized_vector(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Return normalized feature vector z or None if calibration not ready."""
        window_idx = len(self.rows)
        if window_idx < self.warmup_windows:
            self._baseline_rows.append({k: float(features[k]) for k in self._feature_names})
            return None

        self._ensure_calibrator()
        assert self._calibrator is not None
        return self._calibrator.transform_dict(features)

    def _select_score_coordinates(self, z: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """Select which coordinates to use for scoring based on score_from.

        With 12 features: H0 block = 0..5, H1 block = 6..11.
        h1_then_h0: original 3 per dimension [0,1,2] and [6,7,8].
        h1_extended: 4 features (max_persistence, count, entropy, wasserstein_amplitude):
            H1 = [6,7,8,9], H0 fallback = [0,1,2,3] when H1 is empty."""
        h0_idx = [0, 1, 2]   # max_persistence, count, entropy
        h1_idx = [6, 7, 8]   # same for H1
        h0_extended = [0, 1, 2, 3]   # + wasserstein_amplitude
        h1_extended = [6, 7, 8, 9]   # + wasserstein_amplitude

        mode = self.score_from
        if mode == "all":
            return z
        if mode == "h0_only":
            return z[h0_idx]
        if mode == "h1_only":
            return z[h1_idx]
        if mode == "h1_then_h0":
            if features.get("h1_count", 0.0) and features.get("h1_max_persistence", 0.0):
                return z[h1_idx]
            return z[h0_idx]
        if mode == "h1_extended":
            if features.get("h1_count", 0.0) and features.get("h1_max_persistence", 0.0):
                return z[h1_extended]
            return z[h0_extended]
        # Default: same as h1_then_h0 for backwards compatibility.
        return z

    def _aggregate_score(self, z_sel: np.ndarray) -> float:
        if self.score_mode == "weighted_sum":
            if z_sel.size == 0:
                return 0.0
            w = np.ones_like(z_sel) / float(z_sel.size)
            return float(np.sum(np.abs(z_sel) * w))
        # Default: L2 norm of selected coordinates.
        return float(np.linalg.norm(z_sel, ord=2))

    def _update_threshold_and_flags(self, score: float) -> Dict[str, Any]:
        """Update empirical threshold state and return detection metadata."""
        window_idx = len(self.rows)

        if self.threshold_mode == "empirical_quantile":
            if window_idx < self.warmup_windows:
                return {"threshold": float("nan"), "flagged_window": False}

            cal_end = self.warmup_windows + self.calibration_windows

            if window_idx < cal_end:
                self._calibration_scores.append(float(score))
                return {"threshold": float("nan"), "flagged_window": False}

            if self._threshold_from_quantile is None:
                self._threshold_from_quantile = empirical_quantile_threshold(
                    self._calibration_scores, self.threshold_quantile
                )

            thresh = float(self._threshold_from_quantile)
        else:
            # Fixed-threshold (legacy) mode uses the configured threshold after warmup.
            if window_idx < self.warmup_windows:
                return {"threshold": float("nan"), "flagged_window": False}
            thresh = self.threshold

        if np.isnan(score):
            self._consecutive_flags = 0
            return {"threshold": thresh, "flagged_window": False}

        if score > thresh:
            self._consecutive_flags += 1
        else:
            self._consecutive_flags = 0

        flagged_window = self._consecutive_flags >= self.k_consecutive
        return {
            "threshold": thresh,
            "flagged_window": bool(flagged_window),
        }

    def update(self, t: int) -> Dict[str, Any]:
        """Fetch point cloud from the buffer, compute TDA features and detection."""
        point_cloud = self.window_buffer.get_point_cloud(mode=self.point_cloud_mode)

        # Accumulate PCA samples and (if enough information is available) fit PCA
        # before computing persistence so that subsequent windows are evaluated
        # in a reduced space.
        self._maybe_accumulate_pca_samples(point_cloud)
        self._maybe_fit_pca()

        features = self.compute_tda_features(point_cloud)

        row: Dict[str, Any] = {"t": int(t)}
        row.update(features)
        self.rows.append(row)

        z = self._compute_normalized_vector(features)
        if z is None:
            score = float("nan")
        else:
            z_sel = self._select_score_coordinates(z, features)
            score = self._aggregate_score(z_sel)

        row["score"] = score

        det_meta = self._update_threshold_and_flags(score)
        row["threshold"] = det_meta["threshold"]
        row["consecutive_flags"] = int(self._consecutive_flags)
        row["flagged_window"] = bool(det_meta["flagged_window"])
        # Backwards-compatible name expected by existing analysis code/tests.
        row["flag"] = bool(det_meta["flagged_window"])

        # Also update the stored row with the detection outputs.
        self.rows[-1] = row
        return row

