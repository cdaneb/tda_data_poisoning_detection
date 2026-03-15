"""Diagnostic plots for strongest trigger runs: val_5 dims_5 and val_3 dims_5.
Produces side-by-side figures (one per run) and saves to outputs/trigger/diagnostics/."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POISON_START_T = 4000
BASE = Path(__file__).resolve().parent.parent / "outputs" / "trigger"
OUT_DIR = BASE / "diagnostics"

RUNS = [
    ("val_5", "dims_5", "trigger_value=5.0, dims=[0,1,2,3,4]"),
    ("val_3", "dims_5", "trigger_value=3.0, dims=[0,1,2,3,4]"),
]
H1_FEATURES = ["h1_wasserstein_amplitude", "h1_landscape_amplitude", "h1_betti_curve_mean"]


def get_onset_window_id(df: pd.DataFrame) -> int | None:
    onset = df[df["t"] >= POISON_START_T]
    if onset.empty:
        return None
    return int(onset.iloc[0]["window_id"])


def get_threshold(df: pd.DataFrame) -> float | None:
    th = df["threshold"].dropna()
    if th.empty:
        return None
    return float(th.iloc[-1])


def plot_one_run(ax_anomaly, ax_h1_list, df: pd.DataFrame, title: str) -> None:
    df = df.copy()
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")
    window_id = df["window_id"].to_numpy()
    onset_wi = get_onset_window_id(df)
    threshold = get_threshold(df)

    # Post-onset mask (windows with t >= 4000)
    post_onset = df["in_poison_region"].to_numpy()
    if onset_wi is not None:
        post_onset = df["window_id"].to_numpy() >= onset_wi

    # ---- Plot 1: Anomaly score time series ----
    ax = ax_anomaly
    ax.set_title(title, fontsize=10)
    # Shade post-onset
    if post_onset.any():
        xmin = window_id[post_onset].min()
        xmax = window_id.max()
        ax.axvspan(xmin, xmax, color="orange", alpha=0.2, label="post-onset")
    # Anomaly score
    valid = np.isfinite(df["anomaly_score"].to_numpy())
    if valid.any():
        ax.plot(window_id[valid], df.loc[valid, "anomaly_score"], "b-", label="anomaly score", linewidth=0.8)
    # Threshold
    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label="threshold")
    # Flagged windows (red dots)
    flagged = df["flagged"].to_numpy().astype(bool)
    if flagged.any():
        f_windows = window_id[flagged]
        f_scores = pd.to_numeric(df.loc[flagged, "anomaly_score"], errors="coerce").to_numpy()
        valid_f = np.isfinite(f_scores)
        if valid_f.any():
            ax.scatter(f_windows[valid_f], f_scores[valid_f], color="red", s=14, zorder=5, label="flagged")
    ax.set_ylabel("anomaly score")
    ax.set_xlabel("window index")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: H1 feature time series (3 subplots) ----
    for ax, feat in zip(ax_h1_list, H1_FEATURES):
        if feat not in df.columns:
            ax.set_visible(False)
            continue
        vals = pd.to_numeric(df[feat], errors="coerce").to_numpy()
        ax.plot(window_id, vals, "b-", linewidth=0.7)
        if post_onset.any():
            xmin = window_id[post_onset].min()
            xmax = window_id.max()
            ax.axvspan(xmin, xmax, color="orange", alpha=0.2)
        if onset_wi is not None:
            ax.axvline(onset_wi, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_ylabel(feat.replace("h1_", ""), fontsize=8)
        ax.grid(True, alpha=0.3)
    ax_h1_list[-1].set_xlabel("window index")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # One figure: two diagnostic plots side by side (one per run)
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex="col")
    for col, (val_dir, dims_dir, title) in enumerate(RUNS):
        path = BASE / val_dir / dims_dir / "seed_0" / "window_metrics.csv"
        if not path.exists():
            for row in range(4):
                axes[row, col].text(0.5, 0.5, "Missing data", ha="center", va="center", transform=axes[row, col].transAxes)
            continue
        df = pd.read_csv(path)
        ax_anomaly = axes[0, col]
        ax_h1_list = [axes[1, col], axes[2, col], axes[3, col]]
        plot_one_run(ax_anomaly, ax_h1_list, df, title)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "trigger_dims_5_diagnostics.png", dpi=150)
    plt.close(fig)
    print("Saved outputs/trigger/diagnostics/trigger_dims_5_diagnostics.png")

    # Per-run figures for easier inspection
    for val_dir, dims_dir, title in RUNS:
        path = BASE / val_dir / dims_dir / "seed_0" / "window_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        fig2, (ax_anom, ax_h1_wa, ax_h1_la, ax_h1_bc) = plt.subplots(4, 1, figsize=(6, 9), sharex=True)
        plot_one_run(ax_anom, [ax_h1_wa, ax_h1_la, ax_h1_bc], df, title)
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        name = f"{val_dir}_{dims_dir}_diagnostics.png"
        fig2.savefig(OUT_DIR / name, dpi=150)
        plt.close(fig2)
        print(f"Saved outputs/trigger/diagnostics/{name}")


if __name__ == "__main__":
    main()
