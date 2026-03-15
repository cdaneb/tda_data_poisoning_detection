"""Pre vs post onset H1 feature statistics for val_5/dims_5 and val_3/dims_5.
Computes pre_mean, post_mean, post_std, and shift = (post_mean - pre_mean) / pre_std
for all 6 H1 features. Prints tables to terminal."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ONSET_WINDOW_ID = 395
BASE = Path(__file__).resolve().parent.parent / "outputs" / "trigger"

H1_FEATURES = [
    "h1_max_persistence",
    "h1_count",
    "h1_entropy",
    "h1_wasserstein_amplitude",
    "h1_landscape_amplitude",
    "h1_betti_curve_mean",
]

RUNS = [
    ("val_5", "dims_5", "trigger_value=5.0, dims_5"),
    ("val_3", "dims_5", "trigger_value=3.0, dims_5"),
]


def compute_shift_table(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df["window_id"] < ONSET_WINDOW_ID]
    post = df[df["window_id"] >= ONSET_WINDOW_ID]
    rows = []
    for feat in H1_FEATURES:
        if feat not in df.columns:
            rows.append({"feature": feat, "pre_mean": None, "post_mean": None, "post_std": None, "shift": None})
            continue
        pre_vals = pd.to_numeric(pre[feat], errors="coerce").dropna()
        post_vals = pd.to_numeric(post[feat], errors="coerce").dropna()
        pre_mean = float(pre_vals.mean()) if len(pre_vals) else float("nan")
        post_mean = float(post_vals.mean()) if len(post_vals) else float("nan")
        post_std = float(post_vals.std()) if len(post_vals) > 1 else 0.0
        pre_std = float(pre_vals.std()) if len(pre_vals) > 1 else float("nan")
        if pre_std and pre_std > 0 and pd.notna(post_mean) and pd.notna(pre_mean):
            shift = (post_mean - pre_mean) / pre_std
        else:
            shift = float("nan")
        rows.append({
            "feature": feat,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "post_std": post_std,
            "shift": shift,
        })
    return pd.DataFrame(rows)


def main():
    for val_dir, dims_dir, label in RUNS:
        path = BASE / val_dir / dims_dir / "seed_0" / "window_metrics.csv"
        if not path.exists():
            print(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        tab = compute_shift_table(df)
        print(f"\n{'='*90}")
        print(label)
        print(f"{'='*90}")
        print(f"{'feature':<32} | {'pre_mean':>12} | {'post_mean':>12} | {'post_std':>10} | {'shift':>10}")
        print("-" * 90)
        for _, r in tab.iterrows():
            pre_s = f"{r['pre_mean']:.6g}" if pd.notna(r["pre_mean"]) else "—"
            post_s = f"{r['post_mean']:.6g}" if pd.notna(r["post_mean"]) else "—"
            pstd_s = f"{r['post_std']:.6g}" if pd.notna(r["post_std"]) else "—"
            shift_s = f"{r['shift']:.4f}" if pd.notna(r["shift"]) else "—"
            print(f"{r['feature']:<32} | {pre_s:>12} | {post_s:>12} | {pstd_s:>10} | {shift_s:>10}")
    print()


if __name__ == "__main__":
    main()
