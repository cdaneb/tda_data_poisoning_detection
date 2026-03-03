"""Tests for WindowBuffer and TDAMonitor."""

import numpy as np

from src.streaming.window_buffer import WindowBuffer
from src.streaming.tda_monitor import TDAMonitor


def test_windowbuffer_point_cloud_shapes():
    n_features = 3
    wb = WindowBuffer(window_size=10, n_features=n_features)

    for t in range(5):
        x = np.array([t, t + 1, t + 2], dtype=float)
        wb.add(t, x, y=0, y_prob=0.2 * t, err=0.1 * t, poisoned=False)

    assert len(wb) == 5

    pc_features = wb.get_point_cloud(mode="features")
    assert pc_features.shape == (5, n_features)

    pc_residuals = wb.get_point_cloud(mode="residuals")
    assert pc_residuals.shape == (5, n_features + 2)


def test_tda_monitor_no_crash_empty_h1():
    """Random blob-like points may have trivial or empty H1; monitor should not crash."""
    n_features = 2
    wb = WindowBuffer(window_size=30, n_features=n_features)

    rng = np.random.default_rng(0)
    for t in range(30):
        x = rng.normal(size=n_features)
        wb.add(t, x, y=0)

    monitor = TDAMonitor(
        window_buffer=wb,
        threshold=3.0,
        warmup_windows=3,
        k_consecutive=2,
        point_cloud_mode="features",
        tda_threshold=0.1,
        maxdim=1,
    )

    for t in range(5):
        row = monitor.update(t)
        # Required TDA feature columns should always be present.
        for key in [
            "h0_max_persistence",
            "h0_count",
            "h0_entropy",
            "h1_max_persistence",
            "h1_count",
            "h1_entropy",
        ]:
            assert key in row


def test_detection_score_finite_after_warmup():
    """After warmup, detection score should be finite."""
    n_features = 2
    wb = WindowBuffer(window_size=30, n_features=n_features)

    rng = np.random.default_rng(1)
    for t in range(30):
        x = rng.normal(size=n_features)
        wb.add(t, x, y=0)

    monitor = TDAMonitor(
        window_buffer=wb,
        threshold=3.0,
        warmup_windows=3,
        k_consecutive=2,
        point_cloud_mode="features",
        tda_threshold=0.1,
        maxdim=1,
    )

    rows = []
    for t in range(10):
        row = monitor.update(t)
        rows.append(row)

    # Rows after warmup should have finite scores.
    for row in rows[4:]:
        assert "score" in row
        assert np.isfinite(row["score"])

