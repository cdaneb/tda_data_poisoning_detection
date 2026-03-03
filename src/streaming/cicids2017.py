"""Utilities for loading the CICIDS2017 flow-based intrusion detection dataset.

This module provides helpers to list CSV files, load individual CSVs with
CICIDS-specific cleaning, and construct a time-ordered :class:`DataStream`
for streaming experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .stream import DataStream


def list_cicids_files(root_dir: str) -> List[str]:
    """Return all CSV file paths under ``root_dir`` (recursively), sorted.

    Files are sorted by filename to provide a deterministic ordering.
    Non-CSV files are ignored.
    """
    root = Path(root_dir)
    if not root.exists():
        return []

    files = sorted(root.rglob("*.csv"), key=lambda p: p.name)
    return [str(p) for p in files]


def load_cicids_csv(
    path: str,
    *,
    label_col: str = "Label",
    time_col: str = "Timestamp",
    benign_label: str = "BENIGN",
    drop_cols: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """Load and clean a single CICIDS2017 CSV file.

    Returns
    -------
    X : np.ndarray
        Numeric feature matrix of shape (n_samples, n_features), dtype float32.
    y : np.ndarray
        Binary labels of shape (n_samples,), where 0 = benign, 1 = attack.
    timestamps : np.ndarray or None
        Array of dtype datetime64[ns] if valid timestamps could be parsed,
        otherwise ``None``.
    feature_names : list[str]
        Names of the numeric feature columns corresponding to X.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if max_rows is not None and max_rows > 0:
        df = df.head(int(max_rows))

    # Replace +/-inf with NaN then drop any rows containing NaN.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")

    # Default non-feature columns commonly present in CICIDS2017.
    default_drop = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Source Port",
        "Destination Port",
        "Protocol",
    ]
    if drop_cols is not None:
        default_drop = drop_cols

    # Labels: 0 for benign, 1 for any non-benign label.
    if label_col in df.columns:
        labels = (df[label_col].astype(str) != str(benign_label)).astype(int).to_numpy()
    else:
        raise KeyError(f"Label column {label_col!r} not found in {path!r}")

    # Timestamps: best-effort conversion, fall back to None if any NaT.
    timestamps_arr: Optional[np.ndarray]
    if time_col in df.columns:
        try:
            ts = pd.to_datetime(df[time_col], errors="coerce")
            if ts.isna().any():
                timestamps_arr = None
            else:
                timestamps_arr = ts.to_numpy()
        except Exception:
            timestamps_arr = None
    else:
        timestamps_arr = None

    # Drop label, time, and obvious non-feature columns.
    cols_to_drop = list(default_drop)
    cols_to_drop.append(label_col)
    if time_col is not None:
        cols_to_drop.append(time_col)

    feature_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Keep only numeric feature columns.
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_names = list(feature_df.columns)

    X = feature_df.to_numpy(dtype=np.float32, copy=True)
    y = labels.astype(int)

    if X.shape[0] != y.shape[0]:
        raise ValueError("Feature and label arrays must have the same number of rows.")

    return X, y, timestamps_arr, feature_names


def make_cicids_stream(
    root_dir: str,
    *,
    file_glob: Optional[str] = None,
    label_col: str = "Label",
    time_col: str = "Timestamp",
    benign_label: str = "BENIGN",
    test_size: float = 0.2,
    random_state: int = 0,
    max_rows: Optional[int] = None,
    max_files: Optional[int] = None,
) -> Tuple[DataStream, Tuple[np.ndarray, np.ndarray], Dict[str, object]]:
    """Construct a time-ordered DataStream from CICIDS2017 CSV files."""
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    files = list_cicids_files(root_dir)
    if file_glob:
        files = [f for f in files if file_glob in Path(f).name]

    if max_files is not None and max_files > 0:
        files = files[: int(max_files)]

    if not files:
        raise ValueError(f"No CICIDS2017 CSV files found under {root_dir!r}")

    loaded: List[Dict[str, object]] = []
    has_timestamps = True

    for f in files:
        X, y, ts, feature_names = load_cicids_csv(
            f,
            label_col=label_col,
            time_col=time_col,
            benign_label=benign_label,
            max_rows=max_rows,
        )
        if ts is None:
            has_timestamps = False
        loaded.append(
            {
                "X": X,
                "y": y,
                "timestamps": ts,
                "feature_names": feature_names,
            }
        )

    # Align feature columns across files by intersection of names.
    common = set(loaded[0]["feature_names"])  # type: ignore[arg-type]
    for item in loaded[1:]:
        common &= set(item["feature_names"])  # type: ignore[arg-type]

    if not common:
        raise ValueError("No common feature columns across CICIDS files.")

    # Preserve ordering from the first file for determinism.
    first_names: List[str] = loaded[0]["feature_names"]  # type: ignore[assignment]
    feature_names = [c for c in first_names if c in common]

    X_all_list: List[np.ndarray] = []
    y_all_list: List[np.ndarray] = []
    ts_all_list: List[np.ndarray] = []

    for item in loaded:
        X_i: np.ndarray = item["X"]  # type: ignore[assignment]
        y_i: np.ndarray = item["y"]  # type: ignore[assignment]
        ts_i: Optional[np.ndarray] = item["timestamps"]  # type: ignore[assignment]
        names_i: List[str] = item["feature_names"]  # type: ignore[assignment]

        # Reorder / subset columns to the common feature set.
        idxs = [names_i.index(c) for c in feature_names]
        X_i_aligned = X_i[:, idxs]

        X_all_list.append(X_i_aligned)
        y_all_list.append(y_i)
        if has_timestamps and ts_i is not None:
            ts_all_list.append(ts_i)

    X_all = np.vstack(X_all_list)
    y_all = np.concatenate(y_all_list)

    timestamps_all: Optional[np.ndarray]
    if has_timestamps and ts_all_list:
        timestamps_all = np.concatenate(ts_all_list)
        # Stable sort by timestamps if present.
        order = np.argsort(timestamps_all, kind="mergesort")
        X_all = X_all[order]
        y_all = y_all[order]
        timestamps_all = timestamps_all[order]
    else:
        timestamps_all = None

    n_samples = X_all.shape[0]
    split_idx = int(n_samples * (1.0 - float(test_size)))
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError("test_size results in empty train or test split.")

    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_test = X_all[split_idx:]
    y_test = y_all[split_idx:]

    stream = DataStream(X_train, y_train, shuffle=False, random_state=random_state)

    meta: Dict[str, object] = {
        "feature_names": feature_names,
        "files_used": files,
        "n_rows": int(n_samples),
        "has_timestamps": bool(timestamps_all is not None),
        "root_dir": str(root_dir),
    }

    return stream, (X_test, y_test), meta

