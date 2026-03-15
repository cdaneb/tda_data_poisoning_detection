"""End-to-end streaming experiment with online learning and TDA monitoring.

This module is wired up in multiple incremental steps. Initially it only
defines the public configuration dataclass and a placeholder run_experiment
function so that imports are stable for later implementation and tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..streaming.online_model import OnlineLearner
from ..streaming.poison import PoisoningAttack
from ..streaming.stream import DataStream, make_classification_stream
from ..streaming.tda_monitor import TDAMonitor
from ..streaming.window_buffer import WindowBuffer
from ..streaming.cicids2017 import make_cicids_stream
from ..streaming.drift import apply_natural_drift


@dataclass
class ExperimentConfig:
    """Configuration for a streaming TDA + poisoning experiment."""

    n_steps: int = 2000
    n_features: int = 5
    test_size: float = 0.2
    class_sep: float = 1.0
    flip_y: float = 0.0
    window_size: int = 50
    stride: int = 10
    point_cloud_mode: str = "residuals"
    warmup_windows: int = 50
    calibration_windows: int = 50
    threshold_quantile: float = 0.995
    k_consecutive: int = 2
    tda_threshold: float = 0.1
    maxdim: int = 1
    eval_every: int = 50
    # Threshold: unused when threshold_mode is empirical_quantile (must be derived from calibration).
    threshold: float = 0.0
    threshold_mode: str = "empirical_quantile"
    dr_method: str = "pca"
    pca_n_components: int | None = 15
    pca_variance: float | None = None
    pca_max_components: int = 20
    pca_max_points_per_window: int = 50
    pca_strict: bool = False
    baseline_mode: str = "robust_z"
    score_mode: str = "l2"
    score_from: str = "h1_then_h0"

    poison_mode: str = "label_flip"
    poison_start_t: int | None = 4000
    poison_end_t: int | None = None  # None = one-way, stays poisoned to end
    poison_rate: float = 0.3
    poison_target_class: int | None = 0  # flip benign (0) -> attack (1)
    trigger_value: float = 0.5
    trigger_dims: List[int] = field(default_factory=lambda: [0])
    poison_target_label: int = 1
    random_state: int = 0
    output_dir: str = "outputs"

    # CICIDS2017 integration.
    cicids_root_dir: str | None = None
    cicids_file_glob: str | None = None
    cicids_max_files: int | None = None
    label_col: str = "Label"
    benign_label: str | int = "BENIGN"
    time_col: str = "Timestamp"
    max_rows: int | None = 20000

    # Drift configuration.
    drift_start_t: int | None = None
    drift_end_t: int | None = None
    drift_dims: List[int] = field(default_factory=lambda: [0])
    drift_magnitude: float = 0.0

    # High-level experimental condition and seed.
    # clean, label_flip, trigger, drift, drift_poison
    condition: str = "clean"
    seed: int = 0


def _compute_recent_poison_rate(buffer: WindowBuffer, window: int = 100) -> float:
    data = buffer.data
    if not data:
        return 0.0
    recent = data[-min(len(data), window) :]
    poisoned_flags = [
        bool(item.get("poisoned")) for item in recent if item.get("poisoned") is not None
    ]
    if not poisoned_flags:
        return 0.0
    return float(np.mean(poisoned_flags))


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the streaming poisoning experiment and return basic metadata."""
    # Output directory: use config.output_dir as run root when custom (e.g. outputs/label_flip/rate_0.05/seed_0).
    base_output = Path(config.output_dir)
    if config.output_dir.strip() != "outputs":
        run_output = base_output
    else:
        run_output = base_output / str(config.condition) / f"seed_{config.seed}"
    run_output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build data stream and held-out test set.
    # ------------------------------------------------------------------
    dataset_meta: Dict[str, Any]

    if config.cicids_root_dir:
        stream, (X_test, y_test), dataset_meta = make_cicids_stream(
            root_dir=config.cicids_root_dir,
            file_glob=config.cicids_file_glob,
            label_col=config.label_col,
            time_col=config.time_col,
            benign_label=str(config.benign_label),
            test_size=config.test_size,
            random_state=config.random_state,
            max_rows=config.max_rows,
            max_files=config.cicids_max_files,
        )
        # Ensure n_features matches the loaded dataset.
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
        # Synthetic dataset metadata.
        dataset_meta = {
            "feature_names": [f"x{i}" for i in range(config.n_features)],
            "files_used": [],
            "n_rows": int(len(stream) + X_test.shape[0]),
            "has_timestamps": False,
            "root_dir": None,
        }

    # ------------------------------------------------------------------
    # Scaling: fit StandardScaler on an initial prefix of the training stream.
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    prefix_n = min(1000, 5 * config.window_size, len(stream))
    if prefix_n <= 0:
        raise ValueError("Training stream must contain at least one sample.")
    X_prefix = stream.X[:prefix_n]
    scaler.fit(X_prefix)
    # Transform the full training stream and held-out test set.
    stream.X = scaler.transform(stream.X)
    X_test_scaled = scaler.transform(X_test)

    learner = OnlineLearner(
        n_features=config.n_features,
        random_state=config.random_state,
    )

    # Determine poisoning mode based on high-level condition.
    if config.condition == "label_flip":
        poisoning_mode = "label_flip"
    elif config.condition == "trigger":
        poisoning_mode = "trigger"
    else:
        poisoning_mode = config.poison_mode

    attack = PoisoningAttack(
        mode=poisoning_mode,
        start_t=config.poison_start_t,
        end_t=config.poison_end_t,
        poison_rate=config.poison_rate,
        target_class=config.poison_target_class,
        trigger_value=config.trigger_value,
        trigger_dims=config.trigger_dims,
        target_label=config.poison_target_label,
        random_state=config.random_state,
    )

    buffer = WindowBuffer(
        window_size=config.window_size,
        n_features=config.n_features,
    )

    monitor = TDAMonitor(
        window_buffer=buffer,
        threshold=config.threshold,
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

    metrics_rows: List[Dict[str, Any]] = []
    flip_count = 0  # number of training samples whose label was actually flipped by the attack

    # Main streaming loop.
    for t_stream, x, y in stream:
        # x is already scaled via the StandardScaler.
        x_scaled = np.asarray(x, dtype=float).ravel()

        # Optional natural drift.
        drift_enabled = (
            config.condition in {"drift", "drift_poison"}
            and config.drift_start_t is not None
            and config.drift_end_t is not None
            and config.drift_magnitude != 0.0
        )
        if drift_enabled:
            x_for_attack = apply_natural_drift(
                x_scaled,
                int(t_stream),
                drift_start=int(config.drift_start_t),
                drift_end=int(config.drift_end_t),
                dims=list(config.drift_dims),
                magnitude=float(config.drift_magnitude),
            )
        else:
            x_for_attack = x_scaled

        # Optional poisoning (one-way when poison_end_t is None: active for t >= poison_start_t).
        poisoning_enabled = (
            config.condition in {"label_flip", "trigger", "drift_poison"}
            and config.poison_start_t is not None
            and config.poison_rate > 0.0
        )
        if poisoning_enabled:
            x_used, y_used, poisoned = attack.apply(x_for_attack, int(y), t_stream)
            if poisoned:
                flip_count += 1
        else:
            x_used, y_used, poisoned = x_for_attack, int(y), False

        # Online update of the learner.
        learner.update(x_used, int(y_used), t=t_stream)
        y_prob = float(learner.predict_proba(x_used))
        y_hat = 1 if y_prob >= 0.5 else 0
        err = float(abs(int(y_used) - y_hat))

        buffer.add(
            t_stream,
            x_used,
            int(y_used),
            y_prob=y_prob,
            err=err,
            poisoned=poisoned,
        )

        # TDA monitoring every 'stride' steps once the buffer is full enough.
        if (t_stream % config.stride == 0) and len(buffer) >= config.window_size:
            monitor.update(t_stream)

        # Periodic evaluation on held-out test set.
        if (t_stream % config.eval_every) == 0:
            y_pred_test = np.array([learner.predict(xi) for xi in X_test_scaled], dtype=int)
            test_acc = float(np.mean(y_pred_test == y_test))
            poisoned_rate_recent = _compute_recent_poison_rate(buffer)
            metrics_rows.append(
                {
                    "t": int(t_stream),
                    "test_accuracy": test_acc,
                    "poisoned_rate_recent": poisoned_rate_recent,
                }
            )

    # Convert to DataFrames and save.
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = run_output / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    tda_df = pd.DataFrame(monitor.rows)
    tda_path = run_output / "tda_features.csv"
    tda_df.to_csv(tda_path, index=False)

    # Per-window metrics with ground truth for poisoning evaluation (window_id, t, in_poison_region, anomaly_score, flagged).
    poison_start = config.poison_start_t
    window_metrics_rows = []
    for i, row in enumerate(monitor.rows):
        t = int(row["t"])
        in_poison_region = (poison_start is not None and t >= poison_start)
        r = {
            "window_id": i,
            "t": t,
            "in_poison_region": bool(in_poison_region),
            "anomaly_score": float(row.get("score", float("nan"))),
            "flagged": bool(row.get("flagged_window", row.get("flag", False))),
        }
        for k in [
            "h0_max_persistence", "h0_count", "h0_entropy",
            "h0_wasserstein_amplitude", "h0_landscape_amplitude", "h0_betti_curve_mean",
            "h1_max_persistence", "h1_count", "h1_entropy",
            "h1_wasserstein_amplitude", "h1_landscape_amplitude", "h1_betti_curve_mean",
            "threshold",
        ]:
            if k in row:
                r[k] = row[k]
        window_metrics_rows.append(r)
    window_metrics_df = pd.DataFrame(window_metrics_rows)
    window_metrics_path = run_output / "window_metrics.csv"
    window_metrics_df.to_csv(window_metrics_path, index=False)

    detections_df = tda_df[tda_df.get("flag", False).astype(bool)] if not tda_df.empty else tda_df
    detections_path = run_output / "detections.csv"
    detections_df.to_csv(detections_path, index=False)

    # Plot helpers: vertical line at poison onset; no shaded region for one-way (poison_end_t is None).
    def _poison_onset_line(ax):
        if poison_start is not None and config.poison_rate > 0.0:
            ax.axvline(poison_start, color="red", linestyle="--", label="poison onset")

    thresh_val = getattr(monitor, "_threshold_from_quantile", None)
    thresh_plot = float(thresh_val) if thresh_val is not None and np.isfinite(thresh_val) else None

    # 1) Test accuracy vs t
    if not metrics_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(metrics_df["t"], metrics_df["test_accuracy"], marker="o", label="test accuracy")
        _poison_onset_line(ax)
        ax.set_xlabel("timestep")
        ax.set_ylabel("test accuracy")
        ax.set_title("Test accuracy over time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_output / "accuracy_over_time.png")
        plt.close(fig)

    # 2) Detection score vs t (threshold from empirical quantile)
    if not tda_df.empty and "score" in tda_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tda_df["t"], tda_df["score"], label="detection score")
        if thresh_plot is not None:
            ax.axhline(thresh_plot, color="orange", linestyle="--", label="threshold")
        _poison_onset_line(ax)
        ax.set_xlabel("timestep")
        ax.set_ylabel("score (z)")
        ax.set_title("Detection score over time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_output / "detection_score_over_time.png")
        plt.close(fig)

    # 3) TDA features: two subplots (H0 and H1) with independent y-axes
    if not tda_df.empty:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        if "h0_max_persistence" in tda_df.columns:
            ax0.plot(tda_df["t"], tda_df["h0_max_persistence"], label="h0_max_persistence")
        if "h0_count" in tda_df.columns:
            ax0.plot(tda_df["t"], tda_df["h0_count"], label="h0_count")
        if "h0_entropy" in tda_df.columns:
            ax0.plot(tda_df["t"], tda_df["h0_entropy"], label="h0_entropy")
        ax0.set_ylabel("H0")
        ax0.set_title("TDA features over time (H0)")
        ax0.legend()
        _poison_onset_line(ax0)
        if "h1_max_persistence" in tda_df.columns:
            ax1.plot(tda_df["t"], tda_df["h1_max_persistence"], label="h1_max_persistence")
        if "h1_count" in tda_df.columns:
            ax1.plot(tda_df["t"], tda_df["h1_count"], label="h1_count")
        if "h1_entropy" in tda_df.columns:
            ax1.plot(tda_df["t"], tda_df["h1_entropy"], label="h1_entropy")
        ax1.set_ylabel("H1")
        ax1.set_xlabel("timestep")
        ax1.set_title("H1")
        ax1.legend()
        _poison_onset_line(ax1)
        fig.tight_layout()
        fig.savefig(run_output / "tda_features_over_time.png")
        plt.close(fig)

    # Save config (including dataset metadata and run summary) for reproducibility.
    config_dict = asdict(config)
    config_dict["dataset_meta"] = dataset_meta
    config_dict["run_summary"] = {"flip_count": int(flip_count)}
    config_path = run_output / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    return {
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "tda_features_path": str(tda_path),
        "window_metrics_path": str(window_metrics_path),
        "detections_path": str(detections_path),
        "n_metrics": int(len(metrics_df)),
        "n_tda_rows": int(len(tda_df)),
        "n_detections": int(len(detections_df)),
    }


def run_suite(
    base_config: ExperimentConfig,
    conditions: List[str],
    seeds: List[int],
) -> pd.DataFrame:
    """Run a grid of conditions x seeds and summarize results."""
    rows: List[Dict[str, Any]] = []

    for condition in conditions:
        for seed in seeds:
            cfg = replace(base_config)
            cfg.condition = str(condition)
            cfg.seed = int(seed)
            cfg.random_state = int(seed)

            result = run_experiment(cfg)

            metrics_df = pd.read_csv(result["metrics_path"])
            tda_df = pd.read_csv(result["tda_features_path"])

            detections_count = int(result.get("n_detections", 0))
            mean_test_acc = float(metrics_df["test_accuracy"].mean()) if not metrics_df.empty else float("nan")
            min_test_acc = float(metrics_df["test_accuracy"].min()) if not metrics_df.empty else float("nan")

            poison_start = cfg.poison_start_t
            poison_end = cfg.poison_end_t
            mean_score_pre = float("nan")
            mean_score_during = float("nan")
            if (
                not tda_df.empty
                and "score" in tda_df.columns
                and poison_start is not None
                and poison_end is not None
            ):
                pre = tda_df[tda_df["t"] < poison_start]
                during = tda_df[(tda_df["t"] >= poison_start) & (tda_df["t"] <= poison_end)]
                if not pre.empty:
                    mean_score_pre = float(pre["score"].mean())
                if not during.empty:
                    mean_score_during = float(during["score"].mean())

            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "detections_count": detections_count,
                    "mean_test_acc": mean_test_acc,
                    "min_test_acc": min_test_acc,
                    "mean_score_pre_poison": mean_score_pre,
                    "mean_score_during_poison": mean_score_during,
                    "config_path": result["config_path"],
                    "metrics_path": result["metrics_path"],
                    "tda_features_path": result["tda_features_path"],
                    "detections_path": result["detections_path"],
                }
            )

    summary_df = pd.DataFrame(rows)
    base_output = Path(base_config.output_dir)
    summary_path = base_output / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df

def _build_config_from_args(args: Any) -> ExperimentConfig:
    cfg = ExperimentConfig()
    # Only override a subset of fields from CLI.
    if args.n_steps is not None:
        cfg.n_steps = args.n_steps
    if args.poison_rate is not None:
        cfg.poison_rate = args.poison_rate
    if args.poison_mode is not None:
        cfg.poison_mode = args.poison_mode
    if getattr(args, "poison_start", None) is not None:
        cfg.poison_start_t = int(args.poison_start)
    if getattr(args, "poison_end", None) is not None:
        cfg.poison_end_t = int(args.poison_end)
    if getattr(args, "trigger_value", None) is not None:
        cfg.trigger_value = float(args.trigger_value)
    if getattr(args, "trigger_dims", None) is not None:
        dims_str = args.trigger_dims.strip()
        cfg.trigger_dims = [int(d.strip()) for d in dims_str.split(",") if d.strip()]
    if args.point_cloud_mode is not None:
        cfg.point_cloud_mode = args.point_cloud_mode
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.random_state is not None:
        cfg.random_state = args.random_state
    if getattr(args, "cicids_root_dir", None) is not None:
        cfg.cicids_root_dir = args.cicids_root_dir
    if getattr(args, "cicids_file_glob", None) is not None:
        cfg.cicids_file_glob = args.cicids_file_glob
    if getattr(args, "day", None) is not None:
        cfg.cicids_file_glob = args.day
        if cfg.cicids_root_dir is None:
            cfg.cicids_root_dir = str(Path("data") / "cicids2017")
    if getattr(args, "cicids_max_files", None) is not None:
        cfg.cicids_max_files = args.cicids_max_files
    if getattr(args, "label_col", None) is not None:
        cfg.label_col = args.label_col
    if getattr(args, "benign_label", None) is not None:
        cfg.benign_label = args.benign_label
    if getattr(args, "time_col", None) is not None:
        cfg.time_col = args.time_col
    if getattr(args, "max_rows", None) is not None:
        cfg.max_rows = args.max_rows
    if getattr(args, "condition", None) is not None:
        cfg.condition = args.condition
    if getattr(args, "score_from", None) is not None:
        cfg.score_from = args.score_from
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
        # If random_state was not explicitly provided, mirror from seed.
        if args.random_state is None:
            cfg.random_state = cfg.seed
    # Use CICIDS when --day is provided; ensure root is set.
    if getattr(args, "day", None) is not None and cfg.cicids_root_dir is None:
        cfg.cicids_root_dir = str(Path("data") / "cicids2017")
    # Default condition to label_flip when running poisoning with poison-start.
    if (
        getattr(args, "poison_start", None) is not None
        and args.poison_rate is not None
        and args.poison_rate > 0
        and getattr(args, "condition", None) is None
    ):
        cfg.condition = "label_flip"
    return cfg


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run streaming online-learning + TDA poisoning experiment."
    )
    parser.add_argument("--n_steps", type=int, default=None, help="Total number of timesteps.")
    parser.add_argument(
        "--poison_rate",
        "--poison-rate",
        type=float,
        default=None,
        dest="poison_rate",
        help="Poisoning rate inside the active window.",
    )
    parser.add_argument(
        "--poison_mode",
        type=str,
        default=None,
        choices=["label_flip", "trigger"],
        help="Poisoning mode.",
    )
    parser.add_argument(
        "--point_cloud_mode",
        type=str,
        default=None,
        choices=["features", "residuals"],
        help="Point-cloud construction mode for TDA.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Directory to write metrics, features, and plots.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Random seed for data and model.",
    )
    parser.add_argument(
        "--cicids_root_dir",
        type=str,
        default=str(Path("data") / "cicids2017"),
        help="Root directory for CICIDS2017 CSV files.",
    )
    parser.add_argument(
        "--cicids_file_glob",
        type=str,
        default=None,
        help="Optional substring filter for CICIDS2017 filenames (e.g., 'Monday').",
    )
    parser.add_argument(
        "--cicids_max_files",
        type=int,
        default=None,
        help="Optional limit on number of CICIDS2017 CSV files to load.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Experimental condition: clean, label_flip, trigger, drift, drift_poison.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Per-run seed (also used as random_state if not provided).",
    )
    parser.add_argument(
        "--score-from",
        type=str,
        default=None,
        dest="score_from",
        help="Score from mode: h1_then_h0, h1_extended, all, etc.",
    )
    parser.add_argument(
        "--run_suite",
        action="store_true",
        help="Run a suite across conditions and seeds instead of a single run.",
    )
    parser.add_argument(
        "--max_rows",
        "--max-rows",
        type=int,
        default=None,
        dest="max_rows",
        help="Maximum number of rows to load from the dataset.",
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="CICIDS2017 day filter (e.g. 'Monday'); sets cicids_file_glob.",
    )
    parser.add_argument(
        "--poison-start",
        type=int,
        default=None,
        dest="poison_start",
        help="Timestep at which poisoning begins (one-way if --poison-end not set).",
    )
    parser.add_argument(
        "--poison-end",
        type=int,
        default=None,
        dest="poison_end",
        help="Optional timestep at which poisoning ends; omit for one-way poison.",
    )
    parser.add_argument(
        "--trigger-value",
        type=float,
        default=None,
        dest="trigger_value",
        help="Scalar offset added to trigger_dims (in scaled units).",
    )
    parser.add_argument(
        "--trigger-dims",
        type=str,
        default=None,
        dest="trigger_dims",
        help="Comma-separated feature indices to perturb, e.g. '0' or '0,1,2'.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds for --run_suite, e.g. '0,1,2,3'.",
    )

    args = parser.parse_args()
    if args.run_suite:
        base_config = _build_config_from_args(args)
        if args.condition is not None:
            conditions = [c.strip() for c in args.condition.split(",") if c.strip()]
        else:
            conditions = ["clean", "label_flip", "trigger", "drift", "drift_poison"]
        if args.seeds is not None:
            seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        else:
            seeds = [0]
        run_suite(base_config, conditions, seeds)
    else:
        config = _build_config_from_args(args)
        run_experiment(config)


if __name__ == "__main__":
    main()


