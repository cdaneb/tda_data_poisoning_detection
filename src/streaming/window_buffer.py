"""Windowed buffer over recent streaming samples for TDA point clouds."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np


class WindowBuffer:
    """Ring buffer that stores the most recent W samples from a stream.

    The concrete point-cloud construction logic is implemented in later steps.
    """

    def __init__(self, window_size: int, n_features: int, *, maxlen: Optional[int] = None) -> None:
        self.window_size = int(window_size)
        self.n_features = int(n_features)
        # maxlen parameter is accepted for future flexibility, but the primary
        # capacity is controlled by window_size as specified.
        self._buffer: Deque[Dict[str, object]] = deque(maxlen=self.window_size)

    def add(
        self,
        t: int,
        x: np.ndarray,
        y: int,
        y_prob: Optional[float] = None,
        err: Optional[float] = None,
        poisoned: Optional[bool] = None,
    ) -> None:
        """Add a single sample and associated metadata to the buffer."""
        x_arr = np.asarray(x, dtype=float).ravel()
        if x_arr.shape[0] != self.n_features:
            raise ValueError(
                f"Expected x with {self.n_features} features, got {x_arr.shape[0]}"
            )
        self._buffer.append(
            {
                "t": int(t),
                "x": x_arr,
                "y": int(y),
                "y_prob": None if y_prob is None else float(y_prob),
                "err": None if err is None else float(err),
                "poisoned": bool(poisoned) if poisoned is not None else None,
            }
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)

    def get_point_cloud(self, mode: str = "features") -> np.ndarray:
        """Return a point cloud built from the current window contents.

        Args:
            mode: Either ``\"features\"`` or ``\"residuals\"``.

        Returns:
            A 2D NumPy array where each row is a point in the cloud.
        """
        if not self._buffer:
            raise ValueError("WindowBuffer is empty; cannot build point cloud.")

        mode = str(mode)
        if mode not in ("features", "residuals"):
            raise ValueError(f"Unsupported point_cloud mode: {mode!r}")

        if mode == "features":
            xs = [np.asarray(item["x"], dtype=float).ravel() for item in self._buffer]
            return np.stack(xs, axis=0)

        # mode == "residuals"
        rows = []
        for item in self._buffer:
            x = np.asarray(item["x"], dtype=float).ravel()
            y_prob = 0.0 if item.get("y_prob") is None else float(item["y_prob"])
            err = 0.0 if item.get("err") is None else float(item["err"])
            row = np.concatenate([x, np.array([y_prob, err], dtype=float)])
            rows.append(row)
        return np.stack(rows, axis=0)

    @property
    def data(self) -> List[Dict[str, object]]:
        """Expose the raw buffer contents for debugging and testing."""
        return list(self._buffer)


