"""Sliding window and Takens embedding for streaming/time series TDA."""

from typing import Tuple

import numpy as np

from .homology import compute_persistence


def sliding_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Extract sliding windows from a 1D signal.

    Args:
        signal: 1D array of shape (n_samples,)
        window_size: Length of each window
        stride: Step between consecutive windows

    Returns:
        Windows array of shape (n_windows, window_size)
    """
    n = len(signal)
    if window_size > n:
        return np.array([]).reshape(0, window_size)
    n_windows = (n - window_size) // stride + 1
    windows = np.array([
        signal[i * stride : i * stride + window_size]
        for i in range(n_windows)
    ])
    return windows


def takens_embedding(
    signal: np.ndarray,
    dimension: int,
    time_delay: int,
    stride: int = 1,
) -> np.ndarray:
    """Time-delay (Takens) embedding of a 1D signal.

    Each point is [x(t), x(t+tau), x(t+2*tau), ..., x(t+(d-1)*tau)].

    Args:
        signal: 1D array
        dimension: Embedding dimension d
        time_delay: Time delay tau
        stride: Step between consecutive embedding points

    Returns:
        Point cloud of shape (n_points, dimension)
    """
    n = len(signal)
    max_idx = n - (dimension - 1) * time_delay
    if max_idx <= 0:
        return np.array([]).reshape(0, dimension)

    indices = np.arange(0, max_idx, stride)
    embedded = np.array([
        [signal[i + k * time_delay] for k in range(dimension)]
        for i in indices
    ])
    return embedded


def sliding_window_persistence(
    signal: np.ndarray,
    dimension: int,
    time_delay: int,
    stride: int = 1,
    maxdim: int = 1,
) -> Tuple[np.ndarray, dict]:
    """Takens embedding + persistence for a 1D time series.

    Returns:
        embedded: Point cloud from Takens embedding
        result: Full ripser result (with 'dgms' key)
    """
    embedded = takens_embedding(signal, dimension, time_delay, stride)
    if len(embedded) == 0:
        return embedded, {"dgms": []}
    result = compute_persistence(embedded, maxdim=maxdim)
    return embedded, result
