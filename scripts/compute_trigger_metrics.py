"""Compute detection_delay, detection_rate, fpr_pre_onset, flip_count for trigger runs."""
from __future__ import annotations

import json
import pandas as pd
from pathlib import Path

POISON_START_T = 4000
CONFIGS = [
    ("[0]", "dims_1"),
    ("[0,1,2]", "dims_3"),
    ("[0,1,2,3,4]", "dims_5"),
]
BASE = Path(__file__).resolve().parent.parent / "outputs" / "trigger"


def compute_metrics(csv_path: Path, config_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df["in_poison_region"] = df["in_poison_region"].astype(bool)
    df["flagged"] = df["flagged"].astype(bool)

    pre = df[~df["in_poison_region"]]
    post = df[df["in_poison_region"]]

    fpr_pre_onset = float("nan")
    if len(pre) > 0:
        fpr_pre_onset = pre["flagged"].mean()

    detection_rate = float("nan")
    if len(post) > 0:
        detection_rate = post["flagged"].mean()

    # Poison onset window index = first window with t >= POISON_START_T
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

    flip_count = None
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        run_summary = cfg.get("run_summary", {})
        flip_count = run_summary.get("flip_count")

    return {
        "detection_delay": detection_delay,
        "detection_rate": detection_rate,
        "fpr_pre_onset": fpr_pre_onset,
        "flip_count": flip_count,
    }


def main():
    print("| trigger_dims      | detection_delay | detection_rate | fpr_pre_onset | flip_count |")
    print("|-------------------|-----------------|----------------|---------------|------------|")
    for label, dir_name in CONFIGS:
        run_dir = BASE / dir_name / "seed_0"
        csv_path = run_dir / "window_metrics.csv"
        config_path = run_dir / "config.json"
        if not csv_path.exists():
            print(f"| {label:17} | (missing)       | (missing)      | (missing)      | (missing)   |")
            continue
        m = compute_metrics(csv_path, config_path)
        dd = m["detection_delay"]
        dr = m["detection_rate"]
        fpr = m["fpr_pre_onset"]
        fc = m["flip_count"]
        dd_str = str(dd) if isinstance(dd, str) else f"{int(dd)}"
        dr_str = f"{dr:.4f}" if pd.notna(dr) else "—"
        fpr_str = f"{fpr:.4f}" if pd.notna(fpr) else "—"
        fc_str = str(fc) if fc is not None else "—"
        print(f"| {label:17} | {dd_str:15} | {dr_str:14} | {fpr_str:13} | {fc_str:10} |")


if __name__ == "__main__":
    main()
