"""Compute metrics for trigger runs with multiple trigger_value and trigger_dims.
Output table grouped by trigger_value, then trigger_dims."""
from __future__ import annotations

import json
import pandas as pd
from pathlib import Path

POISON_START_T = 4000
BASE = Path(__file__).resolve().parent.parent / "outputs" / "trigger"

CONFIGS = [
    (3.0, "[0]", "val_3", "dims_1"),
    (3.0, "[0,1,2]", "val_3", "dims_3"),
    (3.0, "[0,1,2,3,4]", "val_3", "dims_5"),
    (5.0, "[0]", "val_5", "dims_1"),
    (5.0, "[0,1,2]", "val_5", "dims_3"),
    (5.0, "[0,1,2,3,4]", "val_5", "dims_5"),
]


def compute_metrics(csv_path: Path, config_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["in_poison_region"] = df["in_poison_region"].astype(bool)
    df["flagged"] = df["flagged"].astype(bool)
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")

    pre = df[~df["in_poison_region"]]
    post = df[df["in_poison_region"]]

    fpr_pre_onset = float("nan")
    if len(pre) > 0:
        fpr_pre_onset = pre["flagged"].mean()

    detection_rate = float("nan")
    if len(post) > 0:
        detection_rate = post["flagged"].mean()

    # detection_delay: windows from poison onset to first flagged window after onset
    onset_rows = df[df["t"] >= POISON_START_T]
    if onset_rows.empty:
        detection_delay = "not detected"
    else:
        poison_onset_window_idx = int(onset_rows.iloc[0]["window_id"])
        flagged_after = df[df["in_poison_region"] & df["flagged"]]
        if flagged_after.empty:
            detection_delay = "not detected"
        else:
            first_flagged_idx = int(flagged_after.iloc[0]["window_id"])
            detection_delay = first_flagged_idx - poison_onset_window_idx

    mean_pre = float("nan")
    mean_post = float("nan")
    if len(pre) > 0:
        mean_pre = pre["anomaly_score"].dropna().mean()
    if len(post) > 0:
        mean_post = post["anomaly_score"].dropna().mean()

    flip_count = None
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        flip_count = cfg.get("run_summary", {}).get("flip_count")

    return {
        "detection_delay": detection_delay,
        "detection_rate": detection_rate,
        "fpr_pre_onset": fpr_pre_onset,
        "flip_count": flip_count,
        "mean_pre_onset": mean_pre,
        "mean_post_onset": mean_post,
    }


def main():
    print("Trigger poisoning: trigger_value=3.0 and 5.0 × dims_1, dims_3, dims_5")
    print("=" * 100)
    for trigger_val in (3.0, 5.0):
        print(f"\ntrigger_value = {trigger_val}")
        print("-" * 100)
        print(f"{'trigger_dims':<20} | {'detection_delay':<16} | {'detection_rate':<14} | {'fpr_pre_onset':<12} | {'flip_count':<10} | {'mean_pre':<10} | {'mean_post':<10} | post vs pre")
        print("-" * 100)
        for _tv, dims_label, val_dir, dims_dir in CONFIGS:
            if _tv != trigger_val:
                continue
            run_dir = BASE / val_dir / dims_dir / "seed_0"
            csv_path = run_dir / "window_metrics.csv"
            config_path = run_dir / "config.json"
            m = compute_metrics(csv_path, config_path)
            if m is None:
                print(f"{dims_label:<20} | (missing)")
                continue
            dd = m["detection_delay"]
            dr = m["detection_rate"]
            fpr = m["fpr_pre_onset"]
            fc = m["flip_count"]
            mean_pre = m["mean_pre_onset"]
            mean_post = m["mean_post_onset"]
            dd_str = str(dd) if isinstance(dd, str) else f"{int(dd)}"
            dr_str = f"{dr:.4f}" if pd.notna(dr) else "—"
            fpr_str = f"{fpr:.4f}" if pd.notna(fpr) else "—"
            fc_str = str(fc) if fc is not None else "—"
            mp_str = f"{mean_pre:.4f}" if pd.notna(mean_pre) else "—"
            mpost_str = f"{mean_post:.4f}" if pd.notna(mean_post) else "—"
            diff = (mean_post - mean_pre) if (pd.notna(mean_post) and pd.notna(mean_pre)) else float("nan")
            diff_str = f"{diff:+.4f}" if pd.notna(diff) else "—"
            print(f"{dims_label:<20} | {dd_str:<16} | {dr_str:<14} | {fpr_str:<12} | {fc_str:<10} | {mp_str:<10} | {mpost_str:<10} | {diff_str}")
    print()


if __name__ == "__main__":
    main()
