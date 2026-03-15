"""Consolidated trigger results: all configs × h1_then_h0 and h1_extended.
Group by trigger_value, then trigger_dims; scorer side by side."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ONSET = 395
BASE = Path(__file__).resolve().parent.parent / "outputs" / "trigger"

# (trigger_value, trigger_dims_label, val_dir, dims_dir for h1_then_h0, dims_dir for h1_extended)
CONFIGS = [
    (3.0, "[0]", "val_3", "dims_1", "dims_1"),           # extended in seed_0_extended
    (3.0, "[0,1,2]", "val_3", "dims_3", "dims_3"),
    (3.0, "[0,1,2,3,4]", "val_3", "dims_5", "dims_5_h1ext"),
    (5.0, "[0]", "val_5", "dims_1", "dims_1"),
    (5.0, "[0,1,2]", "val_5", "dims_3", "dims_3"),
    (5.0, "[0,1,2,3,4]", "val_5", "dims_5", "dims_5_h1ext"),
]


def metrics(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["flagged"] = df["flagged"].astype(bool)
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")
    pre = df[df["window_id"] < ONSET]
    post = df[df["window_id"] >= ONSET]
    fpr = float(pre["flagged"].mean()) if len(pre) else float("nan")
    dr = float(post["flagged"].mean()) if len(post) else float("nan")
    onset_rows = df[df["t"] >= 4000]
    if onset_rows.empty:
        dd = "not detected"
    else:
        poi = int(onset_rows.iloc[0]["window_id"])
        flagged_after = df[(df["window_id"] >= poi) & df["flagged"]]
        if flagged_after.empty:
            dd = "not detected"
        else:
            dd = int(flagged_after.iloc[0]["window_id"]) - poi
    mean_pre = float(pre["anomaly_score"].dropna().mean()) if len(pre) else float("nan")
    mean_post = float(post["anomaly_score"].dropna().mean()) if len(post) else float("nan")
    post_pre = (mean_post - mean_pre) if (pd.notna(mean_post) and pd.notna(mean_pre)) else float("nan")
    return {"detection_delay": dd, "detection_rate": dr, "fpr_pre_onset": fpr, "post_pre_score": post_pre}


def main():
    rows = []
    for trigger_value, dims_label, val_dir, dims_orig, dims_ext in CONFIGS:
        # h1_then_h0: seed_0
        csv_orig = BASE / val_dir / dims_orig / "seed_0" / "window_metrics.csv"
        m_orig = metrics(csv_orig)
        # h1_extended: seed_0_extended for dims_1/dims_3, seed_0 in dims_5_h1ext
        if dims_ext == "dims_5_h1ext":
            csv_ext = BASE / val_dir / dims_ext / "seed_0" / "window_metrics.csv"
        else:
            csv_ext = BASE / val_dir / dims_ext / "seed_0_extended" / "window_metrics.csv"
        m_ext = metrics(csv_ext)
        dd_orig = m_orig["detection_delay"] if m_orig else "—"
        dd_ext = m_ext["detection_delay"] if m_ext else "—"
        dr_orig = f"{m_orig['detection_rate']:.4f}" if m_orig and pd.notna(m_orig["detection_rate"]) else "—"
        dr_ext = f"{m_ext['detection_rate']:.4f}" if m_ext and pd.notna(m_ext["detection_rate"]) else "—"
        fpr_orig = f"{m_orig['fpr_pre_onset']:.4f}" if m_orig and pd.notna(m_orig["fpr_pre_onset"]) else "—"
        fpr_ext = f"{m_ext['fpr_pre_onset']:.4f}" if m_ext and pd.notna(m_ext["fpr_pre_onset"]) else "—"
        pp_orig = f"{m_orig['post_pre_score']:.4f}" if m_orig and pd.notna(m_orig.get("post_pre_score")) else "—"
        pp_ext = f"{m_ext['post_pre_score']:.4f}" if m_ext and pd.notna(m_ext.get("post_pre_score")) else "—"
        rows.append({
            "trigger_value": trigger_value,
            "trigger_dims": dims_label,
            "detection_delay_h1_then_h0": dd_orig,
            "detection_delay_h1_extended": dd_ext,
            "detection_rate_h1_then_h0": dr_orig,
            "detection_rate_h1_extended": dr_ext,
            "fpr_pre_onset_h1_then_h0": fpr_orig,
            "fpr_pre_onset_h1_extended": fpr_ext,
            "post_pre_score_h1_then_h0": pp_orig,
            "post_pre_score_h1_extended": pp_ext,
        })

    # Print grouped by trigger_value, then trigger_dims
    print("Trigger experiment matrix: h1_then_h0 vs h1_extended")
    print("=" * 100)
    for tv in (3.0, 5.0):
        print(f"\ntrigger_value = {tv}")
        print("-" * 100)
        print(f"{'trigger_dims':<18} | {'delay (h1_th_h0)':<16} | {'delay (h1_ext)':<16} | {'rate (h1_th_h0)':<14} | {'rate (h1_ext)':<14} | {'fpr (th_h0)':<10} | {'fpr (ext)':<10} | {'post-pre (th_h0)':<14} | {'post-pre (ext)':<14}")
        print("-" * 100)
        for r in rows:
            if r["trigger_value"] != tv:
                continue
            print(f"{r['trigger_dims']:<18} | {str(r['detection_delay_h1_then_h0']):<16} | {str(r['detection_delay_h1_extended']):<16} | {r['detection_rate_h1_then_h0']:<14} | {r['detection_rate_h1_extended']:<14} | {r['fpr_pre_onset_h1_then_h0']:<10} | {r['fpr_pre_onset_h1_extended']:<10} | {r['post_pre_score_h1_then_h0']:<14} | {r['post_pre_score_h1_extended']:<14}")
    print()


if __name__ == "__main__":
    main()
