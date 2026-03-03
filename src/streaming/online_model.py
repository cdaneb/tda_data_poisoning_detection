"""Online learning models for streaming experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier


class OnlineLearner:
    """Online binary classifier using SGD with logistic loss.

    This is a light wrapper around :class:`sklearn.linear_model.SGDClassifier`
    configured for online learning via :meth:`partial_fit`.
    """

    def __init__(
        self,
        n_features: int,
        classes: Sequence[int] = (0, 1),
        random_state: int = 0,
        **sgd_kwargs: Any,
    ) -> None:
        self.n_features = int(n_features)
        self.classes = tuple(int(c) for c in classes)
        self.random_state = int(random_state)
        self.history: List[Dict[str, Any]] = []
        self._model = SGDClassifier(
            loss="log_loss",
            random_state=self.random_state,
            **sgd_kwargs,
        )
        self._initialized: bool = False

    def update(self, x: np.ndarray, y: int, t: Optional[int] = None) -> None:
        """Ingest a single sample and update the online model.

        Args:
            x: Feature vector of shape (n_features,) or (1, n_features)
            y: Integer class label
            t: Optional timestep metadata stored in history
        """
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"Expected x with {self.n_features} features, got {x_arr.shape[1]}"
            )

        y_arr = np.asarray([int(y)], dtype=int)

        # Initialize with known class set on first update.
        if not self._initialized:
            self._model.partial_fit(x_arr, y_arr, classes=np.asarray(self.classes))
            self._initialized = True
        else:
            self._model.partial_fit(x_arr, y_arr)

        y_pred = int(self._model.predict(x_arr)[0])
        y_prob = float(self.predict_proba(x_arr[0]))
        correct = bool(y_pred == int(y))

        self.history.append(
            {
                "t": t,
                "y_true": int(y),
                "y_pred": y_pred,
                "y_prob": y_prob,
                "correct": correct,
            }
        )

    def predict(self, x: np.ndarray) -> int:
        """Predict the label for a single feature vector."""
        if not self._initialized:
            raise RuntimeError("OnlineLearner has not been updated yet.")
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"Expected x with {self.n_features} features, got {x_arr.shape[1]}"
            )
        return int(self._model.predict(x_arr)[0])

    def predict_proba(self, x: np.ndarray) -> float:
        """Return the probability of the positive class for a single feature vector."""
        if not self._initialized:
            raise RuntimeError("OnlineLearner has not been updated yet.")

        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"Expected x with {self.n_features} features, got {x_arr.shape[1]}"
            )

        proba = self._model.predict_proba(x_arr)[0]

        # By convention the "positive" class is the largest class label.
        positive_class = max(self.classes)
        try:
            idx = int(np.where(self._model.classes_ == positive_class)[0][0])
        except Exception:
            # Fallback: if we cannot find the positive class, use the last column.
            idx = proba.shape[0] - 1
        p = float(proba[idx])
        # Clip for numerical safety.
        return float(np.clip(p, 0.0, 1.0))

