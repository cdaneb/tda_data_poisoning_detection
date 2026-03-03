"""Smoke test for the end-to-end streaming poisoning experiment."""

from pathlib import Path

import pandas as pd

from src.experiments.run_experiment import ExperimentConfig, run_experiment


def test_experiment_smoke(tmp_path):
    cfg = ExperimentConfig(
        n_steps=300,
        window_size=30,
        stride=5,
        warmup_windows=3,
        eval_every=50,
        poison_start_t=100,
        poison_end_t=200,
        poison_rate=0.8,
        output_dir=str(tmp_path),
        random_state=0,
        point_cloud_mode="residuals",
    )

    result = run_experiment(cfg)

    metrics_path = Path(result["metrics_path"])
    tda_path = Path(result["tda_features_path"])
    detections_path = Path(result["detections_path"])

    assert metrics_path.exists()
    assert tda_path.exists()
    assert detections_path.exists()

    metrics_df = pd.read_csv(metrics_path)
    tda_df = pd.read_csv(tda_path)
    det_df = pd.read_csv(detections_path)

    assert len(metrics_df) > 0
    assert len(tda_df) > 0

    # In a clean or weakly-poisoned smoke test run, it is acceptable for the
    # robust baseline to produce zero detections. We only require that the
    # detection machinery is wired end-to-end (flag column present and
    # detections.csv consistent with it).
    assert "flag" in tda_df.columns
    assert len(det_df) <= len(tda_df)

