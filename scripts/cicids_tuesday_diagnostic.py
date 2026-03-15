"""Diagnostic for CICIDS Tuesday: load with max_rows=20000, report rows, labels, attack types, shape."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# Use same root as experiments
ROOT = Path(__file__).resolve().parent.parent / "data" / "cicids2017"
LABEL_COL = "Label"
BENIGN = "BENIGN"
MAX_ROWS = 20000


def main():
    files = sorted(ROOT.rglob("*.csv"), key=lambda p: p.name)
    tuesday = [f for f in files if "Tuesday" in f.name]
    if not tuesday:
        print("No Tuesday CSV found under", ROOT)
        print("Available:", [f.name for f in files[:20]])
        return
    path = tuesday[0]
    print("File:", path)
    print()

    # Raw CSV for attack types and total rows (before dropna)
    df_raw = pd.read_csv(path, nrows=MAX_ROWS)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    if LABEL_COL not in df_raw.columns:
        print("Label column", LABEL_COL, "not found. Columns:", list(df_raw.columns[:15]))
        return

    total_raw = len(df_raw)
    label_counts = df_raw[LABEL_COL].astype(str).value_counts()
    benign_count = int(label_counts.get(BENIGN, 0))
    attack_count = total_raw - benign_count
    print("Total rows loaded (raw, max_rows=%d):" % MAX_ROWS, total_raw)
    print("Label distribution (before binarization):")
    print("  BENIGN:", benign_count, "(%s%%)" % round(100 * benign_count / total_raw, 2))
    print("  attack (total):", attack_count, "(%s%%)" % round(100 * attack_count / total_raw, 2))
    print("Attack types present (unique values in Label):")
    for val, cnt in label_counts.items():
        if str(val).upper() != BENIGN:
            print("  ", val, ":", cnt)
    print()

    # Use loader to get feature shape and post-clean counts
    from src.streaming.cicids2017 import load_cicids_csv
    X, y, ts, feature_names = load_cicids_csv(
        str(path), label_col=LABEL_COL, benign_label=BENIGN, max_rows=MAX_ROWS
    )
    n_after = len(y)
    benign_after = int(np.sum(y == 0))
    attack_after = int(np.sum(y == 1))
    print("After loader (dropna, etc.):")
    print("  Total rows:", n_after)
    print("  Benign:", benign_after, "(%s%%)" % round(100 * benign_after / n_after, 2))
    print("  Attack:", attack_after, "(%s%%)" % round(100 * attack_after / n_after, 2))
    print("Feature shape:", X.shape)
    print()

    # Attack onset in training stream (first index where y==1 in time-ordered train split)
    from src.streaming.cicids2017 import make_cicids_stream
    stream, (X_test, y_test), meta = make_cicids_stream(
        str(ROOT), file_glob="Tuesday", benign_label=BENIGN, test_size=0.2,
        random_state=0, max_rows=MAX_ROWS
    )
    y_train = stream.y
    first_attack_idx = next((i for i, yv in enumerate(y_train) if yv == 1), None)
    if first_attack_idx is not None:
        print("Attack onset (first attack in training stream, 0-based index):", first_attack_idx)
    else:
        print("Attack onset: no attack in training split (all benign in train portion)")


if __name__ == "__main__":
    main()
