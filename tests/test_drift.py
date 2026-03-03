"""Tests for natural drift transform."""

import numpy as np

from src.streaming.drift import apply_natural_drift


def test_no_change_outside_window():
    x = np.array([1.0, 2.0, 3.0])
    params = dict(drift_start=10, drift_end=20, dims=[0, 1], magnitude=1.0)

    # Before the window.
    y_before = apply_natural_drift(x, t=0, **params)
    assert np.allclose(y_before, x)

    # After the window.
    y_after = apply_natural_drift(x, t=25, **params)
    assert np.allclose(y_after, x)


def test_linear_ramp_inside_window():
    x = np.array([0.0, 1.0, 2.0])
    drift_start = 10
    drift_end = 20
    magnitude = 1.0
    dims = [0]

    # At the start of the window: alpha = 0.
    y_start = apply_natural_drift(
        x,
        t=drift_start,
        drift_start=drift_start,
        drift_end=drift_end,
        dims=dims,
        magnitude=magnitude,
    )
    assert np.allclose(y_start, x)

    # Midpoint: alpha = 0.5.
    mid_t = (drift_start + drift_end) // 2
    y_mid = apply_natural_drift(
        x,
        t=mid_t,
        drift_start=drift_start,
        drift_end=drift_end,
        dims=dims,
        magnitude=magnitude,
    )
    expected_mid = x.copy()
    expected_mid[0] += 0.5 * magnitude
    assert np.allclose(y_mid, expected_mid)

    # End of window: alpha = 1.0.
    y_end = apply_natural_drift(
        x,
        t=drift_end,
        drift_start=drift_start,
        drift_end=drift_end,
        dims=dims,
        magnitude=magnitude,
    )
    expected_end = x.copy()
    expected_end[0] += 1.0 * magnitude
    assert np.allclose(y_end, expected_end)


def test_dims_out_of_bounds_safe():
    x = np.array([1.0, 2.0])

    # Include an out-of-bounds index; only valid dims should be affected.
    y = apply_natural_drift(
        x,
        t=5,
        drift_start=0,
        drift_end=10,
        dims=[0, 5],
        magnitude=2.0,
    )

    # Drift applied to dim 0 only.
    alpha = (5 - 0) / max(1, 10 - 0)
    expected = x.copy()
    expected[0] += alpha * 2.0

    assert np.allclose(y, expected)

