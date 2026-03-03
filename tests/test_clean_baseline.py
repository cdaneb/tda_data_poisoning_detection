"""Tests for the clean baseline entrypoint."""

from pathlib import Path

import pandas as pd

from src.experiments.run_clean_baseline import CleanBaselineConfig, run_clean_baseline


def test_run_clean_baseline_writes_artifacts_synthetic(tmp_path):
    """run_clean_baseline should write window_metrics.csv and baseline_params.json."""
    cfg = CleanBaselineConfig(
        use_cicids=False,
        n_steps=300,
        n_features=5,
        window_size=30,
        stride=5,
        warmup_windows=3,
        calibration_windows=3,
        output_dir=str(tmp_path),
        seed=0,
        random_state=0,
    )

    result = run_clean_baseline(cfg, use_cicids=False)
    out_dir = Path(result["output_dir"])

    window_metrics_path = out_dir / "window_metrics.csv"
    baseline_params_path = out_dir / "baseline_params.json"

    assert window_metrics_path.exists()
    assert baseline_params_path.exists()

    df = pd.read_csv(window_metrics_path)
    assert len(df) > 0

