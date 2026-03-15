"""Compute detection_delay, detection_rate, fpr_pre_onset from label-flip window_metrics.csv."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

POISON_START_T = 4000
RATES = ["0.05", "0.10", "0.20"]
BASE = Path(__file__).resolve().parent.parent / "outputs" / "label_flip"


def compute_metrics(csv_path: Path) -> dict:
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
        detection_delay = float("nan")
    else:
        poison_onset_window_idx = int(onset_rows.iloc[0]["window_id"])
        flagged_after = df[df["in_poison_region"] & df["flagged"]]
        if flagged_after.empty:
            detection_delay = float("nan")  # no detection after onset
        else:
            first_flagged_idx = int(flagged_after.iloc[0]["window_id"])
            detection_delay = first_flagged_idx - poison_onset_window_idx

    return {
        "detection_delay": detection_delay,
        "detection_rate": detection_rate,
        "fpr_pre_onset": fpr_pre_onset,
    }


def main():
    print("| poison_rate | detection_delay | detection_rate | fpr_pre_onset |")
    print("|-------------|------------------|----------------|---------------|")
    for rate in RATES:
        path = BASE / f"rate_{rate}" / "seed_0" / "window_metrics.csv"
        if not path.exists():
            print(f"| {rate} | (missing) | (missing) | (missing) |")
            continue
        m = compute_metrics(path)
        dd = m["detection_delay"]
        dr = m["detection_rate"]
        fpr = m["fpr_pre_onset"]
        dd_str = f"{int(dd)}" if pd.notna(dd) else "—"
        dr_str = f"{dr:.4f}" if pd.notna(dr) else "—"
        fpr_str = f"{fpr:.4f}" if pd.notna(fpr) else "—"
        print(f"| {rate} | {dd_str} | {dr_str} | {fpr_str} |")


if __name__ == "__main__":
    main()
