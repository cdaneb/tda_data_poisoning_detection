"""Clean baseline experiment with TDA monitoring and publication-quality artifacts.

This entrypoint runs a *clean* streaming experiment (no poisoning) with an
online learner, TDA-based monitor, robust baseline calibration, and empirical
thresholding. It writes per-window metrics and calibration parameters for
reproducible, publication-quality baselines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from ..streaming.cicids2017 import make_cicids_stream
from ..streaming.online_model import OnlineLearner
from ..streaming.stream import DataStream, make_classification_stream
from ..streaming.tda_monitor import TDAMonitor
from ..streaming.window_buffer import WindowBuffer


@dataclass
class CleanBaselineConfig:
    """Configuration for clean baseline streaming + TDA experiment."""

    # Dataset configuration (synthetic vs CICIDS2017).
    use_cicids: bool = True
    n_steps: int = 2000
    n_features: int = 5
    test_size: float = 0.2
    class_sep: float = 1.0
    flip_y: float = 0.0

    cicids_root_dir: Optional[str] = None
    cicids_day: Optional[str] = None  # maps to cicids_file_glob
    cicids_max_files: Optional[int] = None
    label_col: str = "Label"
    benign_label: str | int = "BENIGN"
    time_col: str = "Timestamp"
    max_rows: Optional[int] = None

    # Sliding-window and TDA configuration.
    window_size: int = 50
    stride: int = 10
    point_cloud_mode: str = "residuals"
    warmup_windows: int = 50
    calibration_windows: int = 50
    k_consecutive: int = 2
    tda_threshold: float = 0.1
    maxdim: int = 1

    # Dimensionality reduction (PCA) and baseline calibration.
    dr_method: str = "pca"
    pca_n_components: Optional[int] = 15
    pca_variance: Optional[float] = None
    pca_max_components: int = 20
    pca_max_points_per_window: int = 50
    pca_strict: bool = False

    baseline_mode: str = "robust_z"
    threshold_mode: str = "empirical_quantile"
    threshold_quantile: float = 0.995
    score_mode: str = "l2"
    score_from: str = "h1_then_h0"

    # Seed and output.
    random_state: int = 0
    seed: int = 0
    output_dir: str = "outputs"


def _build_stream_and_test(config: CleanBaselineConfig) -> Tuple[DataStream, Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
    """Construct training DataStream and held-out test set."""
    dataset_meta: Dict[str, Any]

    if config.use_cicids and config.cicids_root_dir:
        stream, (X_test, y_test), dataset_meta = make_cicids_stream(
            root_dir=config.cicids_root_dir,
            file_glob=config.cicids_day,
            label_col=config.label_col,
            time_col=config.time_col,
            benign_label=str(config.benign_label),
            test_size=config.test_size,
            random_state=config.random_state,
            max_rows=config.max_rows,
            max_files=config.cicids_max_files,
        )
        config.n_features = int(stream.X.shape[1])
    else:
        stream, (X_test, y_test) = make_classification_stream(
            n_steps=config.n_steps,
            n_features=config.n_features,
            random_state=config.random_state,
            class_sep=config.class_sep,
            flip_y=config.flip_y,
            test_size=config.test_size,
        )
        dataset_meta = {
            "feature_names": [f"x{i}" for i in range(config.n_features)],
            "files_used": [],
            "n_rows": int(len(stream) + X_test.shape[0]),
            "has_timestamps": False,
            "root_dir": None,
        }

    return stream, (X_test, y_test), dataset_meta


def _compute_window_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute per-window classification metrics for the attack class (label 1)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    n = int(y_true.shape[0])
    if n == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision_1": float("nan"),
            "recall_1": float("nan"),
            "f1_1": float("nan"),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "attack_prevalence": float("nan"),
        }

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / n

    # Sensitivity (recall for class 1) and specificity for class 0.
    rec_p = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    rec_n = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    if np.isfinite(rec_p) and np.isfinite(rec_n):
        bal_acc = (rec_p + rec_n) / 2.0
    else:
        try:
            bal_acc = balanced_accuracy_score(y_true, y_pred)
        except Exception:
            bal_acc = float("nan")

    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall_1 = rec_p
    if np.isfinite(precision_1) and np.isfinite(recall_1) and (precision_1 + recall_1) > 0:
        f1_1 = 2.0 * precision_1 * recall_1 / (precision_1 + recall_1)
    else:
        f1_1 = float("nan")

    attack_prevalence = (tp + fn) / n

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bal_acc),
        "precision_1": float(precision_1),
        "recall_1": float(recall_1),
        "f1_1": float(f1_1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "attack_prevalence": float(attack_prevalence),
    }


def run_clean_baseline(config: CleanBaselineConfig, *, use_cicids: Optional[bool] = None) -> Dict[str, Any]:
    """Run the clean baseline experiment and return artifact paths."""
    if use_cicids is not None:
        config.use_cicids = bool(use_cicids)

    # Resolve CICIDS root if not provided explicitly.
    if config.cicids_root_dir is None:
        config.cicids_root_dir = str(Path("data") / "cicids2017")

    # Make random_state mirror seed if not overridden.
    if config.random_state is None:
        config.random_state = config.seed

    # Output layout: outputs/clean/seed_<seed>/
    base_output = Path(config.output_dir)
    run_output = base_output / "clean" / f"seed_{config.seed}"
    run_output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build data stream and held-out test set.
    # ------------------------------------------------------------------
    stream, (X_test, y_test), dataset_meta = _build_stream_and_test(config)

    # ------------------------------------------------------------------
    # Scaling: fit StandardScaler on an initial prefix of the training stream.
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    prefix_n = min(1000, 5 * config.window_size, len(stream))
    if prefix_n <= 0:
        raise ValueError("Training stream must contain at least one sample.")
    X_prefix = stream.X[:prefix_n]
    scaler.fit(X_prefix)
    stream.X = scaler.transform(stream.X)
    X_test_scaled = scaler.transform(X_test)

    learner = OnlineLearner(
        n_features=config.n_features,
        random_state=config.random_state,
    )

    buffer = WindowBuffer(
        window_size=config.window_size,
        n_features=config.n_features,
    )

    monitor = TDAMonitor(
        window_buffer=buffer,
        threshold=3.0,
        warmup_windows=config.warmup_windows,
        calibration_windows=config.calibration_windows,
        k_consecutive=config.k_consecutive,
        point_cloud_mode=config.point_cloud_mode,
        tda_threshold=config.tda_threshold,
        maxdim=config.maxdim,
        dr_method=config.dr_method,
        pca_n_components=config.pca_n_components,
        pca_variance=config.pca_variance,
        pca_max_components=config.pca_max_components,
        pca_max_points_per_window=config.pca_max_points_per_window,
        pca_strict=config.pca_strict,
        baseline_mode=config.baseline_mode,
        threshold_mode=config.threshold_mode,
        threshold_quantile=config.threshold_quantile,
        score_mode=config.score_mode,
        score_from=config.score_from,
        random_state=config.random_state,
    )

    window_rows: List[Dict[str, Any]] = []
    window_id = 0

    # ------------------------------------------------------------------
    # Main streaming loop.
    # ------------------------------------------------------------------
    for t_stream, x, y in stream:
        x_scaled = np.asarray(x, dtype=float).ravel()
        x_used = x_scaled
        y_used = int(y)

        # Online update of the learner.
        learner.update(x_used, int(y_used), t=int(t_stream))
        y_prob = float(learner.predict_proba(x_used))
        y_hat = 1 if y_prob >= 0.5 else 0
        err = float(abs(int(y_used) - y_hat))

        buffer.add(
            int(t_stream),
            x_used,
            int(y_used),
            y_prob=y_prob,
            err=err,
            poisoned=False,
        )

        # TDA monitoring every 'stride' steps once the buffer is full enough.
        if (t_stream % config.stride == 0) and len(buffer) >= config.window_size:
            row_tda = monitor.update(int(t_stream))

            # Per-window classification metrics from the current buffer contents.
            data = buffer.data
            y_true = np.array([int(d["y"]) for d in data], dtype=int)
            y_prob_win = np.array(
                [float(0.0 if d.get("y_prob") is None else d["y_prob"]) for d in data],
                dtype=float,
            )
            y_pred_win = (y_prob_win >= 0.5).astype(int)

            cls_metrics = _compute_window_classification_metrics(y_true, y_pred_win)

            row: Dict[str, Any] = {
                "t": int(t_stream),
                "window_id": int(window_id),
            }
            row.update(cls_metrics)

            # TDA features and detection outputs.
            for key in [
                "h0_max_persistence",
                "h0_count",
                "h0_entropy",
                "h1_max_persistence",
                "h1_count",
                "h1_entropy",
                "score",
                "threshold",
                "flagged_window",
                "consecutive_flags",
            ]:
                if key in row_tda:
                    if key == "score":
                        row["anomaly_score"] = float(row_tda[key])
                    elif key == "threshold":
                        row["threshold_used"] = float(row_tda[key])
                    else:
                        row[key] = row_tda[key]

            row["h1_nonempty"] = bool(row_tda.get("h1_count", 0.0) > 0.0)

            # PCA metadata snapshot (same across windows but convenient here).
            row["pca_requested"] = monitor.pca_n_components_requested
            row["pca_effective"] = monitor.pca_n_components_eff
            row["pca_clamped"] = monitor.pca_clamped
            row["pca_disabled"] = getattr(monitor, "_pca_disabled", False)
            row["dr_method"] = monitor.dr_method

            window_rows.append(row)
            window_id += 1

    # ------------------------------------------------------------------
    # Convert to DataFrame and write window_metrics.csv.
    # ------------------------------------------------------------------
    window_df = pd.DataFrame(window_rows)
    window_metrics_path = run_output / "window_metrics.csv"
    window_df.to_csv(window_metrics_path, index=False)

    # ------------------------------------------------------------------
    # Baseline parameter snapshot (PCA, calibrator, threshold, summary).
    # ------------------------------------------------------------------
    # Ensure calibrator is available if any baseline rows were collected.
    try:
        monitor._ensure_calibrator()  # type: ignore[attr-defined]
    except Exception:
        pass

    calibrator_params: Dict[str, Any]
    if getattr(monitor, "_calibrator", None) is not None:
        calibrator_params = monitor._calibrator.to_params()  # type: ignore[assignment]
    else:
        calibrator_params = {}

    if getattr(monitor, "_pca", None) is not None:
        pca_obj = monitor._pca  # type: ignore[assignment]
        evr = getattr(pca_obj, "explained_variance_ratio_", None)
        evr_list = evr.tolist() if evr is not None else None
    else:
        evr_list = None

    # Summary statistics from window metrics.
    if not window_df.empty and "anomaly_score" in window_df.columns:
        scores = window_df["anomaly_score"].to_numpy(dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size > 0:
            p95 = float(np.percentile(scores, 95))
            p99 = float(np.percentile(scores, 99))
            p995 = float(np.percentile(scores, 99.5))
        else:
            p95 = p99 = p995 = float("nan")
    else:
        p95 = p99 = p995 = float("nan")

    windows_evaluated = int(len(window_df))
    flagged_windows = int(window_df["flagged_window"].sum()) if "flagged_window" in window_df.columns else 0
    flag_rate = float(flagged_windows / windows_evaluated) if windows_evaluated > 0 else 0.0
    h1_nonempty_freq = (
        float(window_df["h1_nonempty"].mean()) if "h1_nonempty" in window_df.columns and not window_df.empty else 0.0
    )

    baseline_params = {
        "config_snapshot": {
            "clean_baseline_config": asdict(config),
            "dataset_meta": dataset_meta,
        },
        "pca": {
            "method": monitor.dr_method,
            "requested_components": monitor.pca_n_components_requested,
            "effective_components": monitor.pca_n_components_eff,
            "clamped": monitor.pca_clamped,
            "disabled": getattr(monitor, "_pca_disabled", False),
            "explained_variance_ratio": evr_list,
        },
        "calibrator": calibrator_params,
        "threshold": {
            "mode": monitor.threshold_mode,
            "quantile": float(config.threshold_quantile),
            "value": float(monitor._threshold_from_quantile)  # type: ignore[attr-defined]
            if getattr(monitor, "_threshold_from_quantile", None) is not None
            else float("nan"),
        },
        "summary": {
            "warmup_windows": int(config.warmup_windows),
            "calibration_windows": int(config.calibration_windows),
            "windows_evaluated": windows_evaluated,
            "flagged_windows": flagged_windows,
            "flag_rate": float(flag_rate),
            "h1_nonempty_freq": float(h1_nonempty_freq),
            "score_percentiles": {
                "p95": p95,
                "p99": p99,
                "p995": p995,
            },
        },
    }

    baseline_params_path = run_output / "baseline_params.json"
    with baseline_params_path.open("w", encoding="utf-8") as f:
        json.dump(baseline_params, f, indent=2)

    # ------------------------------------------------------------------
    # Printed summary (stdout).
    # ------------------------------------------------------------------
    def _agg_stats(series: pd.Series) -> Tuple[float, float, float]:
        if series.empty:
            return float("nan"), float("nan"), float("nan")
        vals = series.astype(float)
        return float(vals.mean()), float(vals.min()), float(vals.max())

    if not window_df.empty:
        ba_mean, ba_min, ba_max = _agg_stats(window_df["balanced_accuracy"])
        r_mean, r_min, r_max = _agg_stats(window_df["recall_1"])
        ap_mean, ap_min, ap_max = _agg_stats(window_df["attack_prevalence"])
    else:
        ba_mean = ba_min = ba_max = float("nan")
        r_mean = r_min = r_max = float("nan")
        ap_mean = ap_min = ap_max = float("nan")

    print(f"Clean baseline output directory: {run_output}")
    print(f"Balanced accuracy: mean={ba_mean:.4f}, min={ba_min:.4f}, max={ba_max:.4f}")
    print(f"Recall (class 1): mean={r_mean:.4f}, min={r_min:.4f}, max={r_max:.4f}")
    print(f"Attack prevalence: mean={ap_mean:.4f}, min={ap_min:.4f}, max={ap_max:.4f}")
    print(f"Windows evaluated: {windows_evaluated}")
    print(f"Flagged windows: {flagged_windows}")
    print(f"Flag rate: {flag_rate:.4f}")
    print(f"Score percentiles: p95={p95:.4f}, p99={p99:.4f}, p99.5={p995:.4f}")
    print(f"H1 non-empty frequency: {h1_nonempty_freq:.4f}")

    return {
        "output_dir": str(run_output),
        "window_metrics_path": str(window_metrics_path),
        "baseline_params_path": str(baseline_params_path),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run clean baseline (no poisoning) with TDA monitoring and calibration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the run (also used as random_state if not overridden).",
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="CICIDS2017 day filter (e.g. 'Monday'); maps to cicids_file_glob.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to load from CICIDS2017 CSV files.",
    )
    parser.add_argument(
        "--cicids-root",
        type=str,
        default=str(Path("data") / "cicids2017"),
        help="Root directory for CICIDS2017 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base directory to write clean baseline outputs.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic make_classification stream instead of CICIDS2017.",
    )

    args = parser.parse_args()

    cfg = CleanBaselineConfig()
    cfg.seed = int(args.seed)
    cfg.random_state = int(args.seed)
    cfg.output_dir = str(args.output_dir)
    cfg.cicids_root_dir = str(args.cicids_root)
    cfg.cicids_day = args.day
    cfg.max_rows = args.max_rows
    cfg.use_cicids = not bool(args.synthetic)

    run_clean_baseline(cfg, use_cicids=cfg.use_cicids)


if __name__ == "__main__":
    main()

