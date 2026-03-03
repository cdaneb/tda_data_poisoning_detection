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
    warmup_windows: int = 5
    threshold: float = 3.0
    k_consecutive: int = 2
    tda_threshold: float = 0.1
    maxdim: int = 1
    eval_every: int = 50
    poison_mode: str = "label_flip"
    poison_start_t: int | None = 800
    poison_end_t: int | None = 1200
    poison_rate: float = 0.3
    poison_target_class: int | None = None
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
    max_rows: int | None = None

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
    # Output directory layout:
    # base output_dir / <condition> / seed_<seed>/
    base_output = Path(config.output_dir)
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
        k_consecutive=config.k_consecutive,
        point_cloud_mode=config.point_cloud_mode,
        tda_threshold=config.tda_threshold,
        maxdim=config.maxdim,
    )

    metrics_rows: List[Dict[str, Any]] = []

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

        # Optional poisoning.
        poisoning_enabled = (
            config.condition in {"label_flip", "trigger", "drift_poison"}
            and config.poison_start_t is not None
            and config.poison_end_t is not None
            and config.poison_rate > 0.0
        )
        if poisoning_enabled:
            x_used, y_used, poisoned = attack.apply(x_for_attack, int(y), t_stream)
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

    detections_df = tda_df[tda_df.get("flag", False).astype(bool)] if not tda_df.empty else tda_df
    detections_path = run_output / "detections.csv"
    detections_df.to_csv(detections_path, index=False)

    # Plots
    poison_start = config.poison_start_t
    poison_end = config.poison_end_t

    def _shade_poison(ax):
        if poison_start is not None and poison_end is not None:
            ax.axvspan(poison_start, poison_end, color="red", alpha=0.15, label="poison window")

    # 1) Test accuracy vs t
    if not metrics_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(metrics_df["t"], metrics_df["test_accuracy"], marker="o", label="test accuracy")
        _shade_poison(ax)
        ax.set_xlabel("timestep")
        ax.set_ylabel("test accuracy")
        ax.set_title("Test accuracy over time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_output / "accuracy_over_time.png")
        plt.close(fig)

    # 2) Detection score vs t
    if not tda_df.empty and "score" in tda_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tda_df["t"], tda_df["score"], label="detection score")
        ax.axhline(config.threshold, color="orange", linestyle="--", label="threshold")
        _shade_poison(ax)
        ax.set_xlabel("timestep")
        ax.set_ylabel("score (z)")
        ax.set_title("Detection score over time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_output / "detection_score_over_time.png")
        plt.close(fig)

    # 3) h1_max_persistence vs t (fall back to h0 if needed)
    if not tda_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        if "h1_max_persistence" in tda_df.columns:
            ax.plot(tda_df["t"], tda_df["h1_max_persistence"], label="h1_max_persistence")
        if "h0_max_persistence" in tda_df.columns:
            ax.plot(tda_df["t"], tda_df["h0_max_persistence"], label="h0_max_persistence")
        _shade_poison(ax)
        ax.set_xlabel("timestep")
        ax.set_ylabel("max persistence")
        ax.set_title("TDA features over time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_output / "tda_features_over_time.png")
        plt.close(fig)

    # Save config (including dataset metadata) for reproducibility.
    config_dict = asdict(config)
    config_dict["dataset_meta"] = dataset_meta
    config_path = run_output / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    return {
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "tda_features_path": str(tda_path),
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
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
        # If random_state was not explicitly provided, mirror from seed.
        if args.random_state is None:
            cfg.random_state = cfg.seed
    return cfg


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run streaming online-learning + TDA poisoning experiment."
    )
    parser.add_argument("--n_steps", type=int, default=None, help="Total number of timesteps.")
    parser.add_argument(
        "--poison_rate",
        type=float,
        default=None,
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
        type=str,
        default=None,
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
        "--run_suite",
        action="store_true",
        help="Run a suite across conditions and seeds instead of a single run.",
    )
    parser.add_argument(
        "--max_rows", 
        type=int, 
        default=None, 
        help="Maximum number of rows to load from the dataset.",
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


