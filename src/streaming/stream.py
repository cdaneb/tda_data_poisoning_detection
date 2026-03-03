"""Streaming data utilities: time-ordered sample generators and helpers."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class DataStream:
    """Simple time-ordered data stream over (X, y) pairs.

    Iterating over the stream yields triples ``(t, x, y)`` where ``t`` is a
    monotonically increasing timestep starting at 0, ``x`` is a feature
    vector of shape ``(n_features,)``, and ``y`` is an integer label.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        shuffle: bool = False,
        random_state: int = 0,
    ) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.X = np.asarray(X)
        self.y = np.asarray(y, dtype=int)
        self.n_samples = int(self.X.shape[0])

        indices = np.arange(self.n_samples)
        if shuffle:
            rng = np.random.default_rng(random_state)
            indices = rng.permutation(indices)
        self._indices = indices

        self._pos: int = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.n_samples

    def __iter__(self) -> "DataStream":
        self._pos = 0
        return self

    def __next__(self) -> Tuple[int, np.ndarray, int]:
        """Return the next (t, x, y) triple from the stream."""
        if self._pos >= self.n_samples:
            raise StopIteration
        idx = int(self._indices[self._pos])
        t = self._pos
        x = self.X[idx]
        y = int(self.y[idx])
        self._pos += 1
        return t, x, y

    def reset(self) -> None:
        """Reset the internal pointer to the start of the stream."""
        self._pos = 0


def make_classification_stream(
    n_steps: int,
    n_features: int,
    random_state: int = 0,
    class_sep: float = 1.0,
    flip_y: float = 0.0,
    test_size: float = 0.2,
) -> Tuple[DataStream, Tuple[np.ndarray, np.ndarray]]:
    """Construct a synthetic classification stream and held-out test set.

    The training portion is wrapped in a :class:`DataStream` while the
    held-out set is returned as arrays ``(X_test, y_test)``.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    X, y = make_classification(
        n_samples=n_steps,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    stream = DataStream(X_train, y_train, shuffle=False)
    return stream, (X_test, y_test)

