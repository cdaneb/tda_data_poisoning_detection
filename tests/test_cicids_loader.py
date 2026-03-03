"""Tests for the CICIDS2017 CSV loader and helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.streaming.cicids2017 import list_cicids_files, load_cicids_csv


def test_load_cicids_csv_basic_cleaning(tmp_path):
    # Construct a tiny CICIDS-like DataFrame.
    df = pd.DataFrame(
        {
            "Flow ID": ["f1", "f2", "f3"],
            "Source IP": ["1.1.1.1"] * 3,
            "Destination IP": ["2.2.2.2"] * 3,
            "Source Port": [80, 80, 80],
            "Destination Port": [8080, 8080, 8080],
            "Protocol": [6, 6, 6],
            "f1": [0.1, np.inf, 0.3],
            "f2": [1.0, 2.0, 3.0],
            "Label": ["BENIGN", "DoS", "BENIGN"],
            "Timestamp": [
                "2017-01-01 00:00:01",
                "2017-01-01 00:00:03",
                "2017-01-01 00:00:02",
            ],
        }
    )

    csv_path = tmp_path / "cicids_small.csv"
    df.to_csv(csv_path, index=False)

    X, y, timestamps, feature_names = load_cicids_csv(str(csv_path))

    # Inf row should be dropped, leaving 2 samples.
    assert X.shape[0] == 2
    assert y.shape[0] == 2

    # Labels: 0 for BENIGN, 1 for anything else.
    # Remaining labels should be BENIGN and BENIGN (attack row with inf dropped).
    assert set(y.tolist()) == {0}

    # Only numeric feature columns remain, excluding label/time and non-features.
    assert set(feature_names) == {"f1", "f2"}
    assert X.shape[1] == 2
    assert X.dtype == np.float32

    # Timestamps either None (if parsing failed) or sortable datetime64 array.
    if timestamps is not None:
        # Check that sorting works and matches expected chronological order.
        sorted_ts = np.sort(timestamps)
        expected = np.array(
            [
                np.datetime64("2017-01-01T00:00:01"),
                np.datetime64("2017-01-01T00:00:02"),
            ]
        )
        assert np.array_equal(sorted_ts, expected)


def test_list_cicids_files_returns_csvs(tmp_path):
    # Create a mix of CSV and non-CSV files.
    (tmp_path / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x,y\n3,4\n", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("not csv", encoding="utf-8")

    subdir = tmp_path / "nested"
    subdir.mkdir()
    (subdir / "c.csv").write_text("x,y\n5,6\n", encoding="utf-8")

    files = list_cicids_files(str(tmp_path))
    names = {Path(f).name for f in files}

    assert "a.csv" in names
    assert "b.csv" in names
    assert "c.csv" in names
    assert "ignore.txt" not in names

