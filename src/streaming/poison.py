"""Local data poisoning attacks applied to streaming samples."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class PoisoningAttack:
    """Configurable local poisoning attack for streaming data."""

    def __init__(
        self,
        mode: str = "label_flip",
        *,
        start_t: Optional[int] = None,
        end_t: Optional[int] = None,
        poison_rate: float = 0.0,
        # label_flip params
        target_class: Optional[int] = None,
        # trigger params
        trigger_value: float = 0.5,
        trigger_dims: Optional[List[int]] = None,
        target_label: int = 1,
        random_state: int = 0,
    ) -> None:
        if poison_rate < 0.0 or poison_rate > 1.0:
            raise ValueError("poison_rate must be in [0, 1].")
        if start_t is not None and end_t is not None and start_t > end_t:
            raise ValueError("start_t must be <= end_t when both are provided.")

        self.mode = mode
        self.start_t = start_t
        self.end_t = end_t
        self.poison_rate = float(poison_rate)
        self.target_class = target_class
        self.trigger_value = float(trigger_value)
        self.trigger_dims = trigger_dims if trigger_dims is not None else [0]
        self.target_label = int(target_label)
        self.random_state = int(random_state)
        self._rng = np.random.default_rng(self.random_state)

        # Basic validation of trigger dimensions (non-negative integers).
        for d in self.trigger_dims:
            if not isinstance(d, int) or d < 0:
                raise ValueError("trigger_dims must contain non-negative integers.")

    def is_active(self, t: int) -> bool:
        """Return True if the attack is active at timestep ``t``.

        When end_t is None, the attack is active for all t >= start_t (one-way).
        """
        if self.start_t is None:
            return False
        t = int(t)
        start = int(self.start_t)
        if self.end_t is None:
            return t >= start
        return start <= t <= int(self.end_t)

    def _maybe_poison(self) -> bool:
        """Return True with probability poison_rate."""
        if self.poison_rate <= 0.0:
            return False
        u = float(self._rng.random())
        return u < self.poison_rate

    def apply(self, x: np.ndarray, y: int, t: int) -> Tuple[np.ndarray, int, bool]:
        """Apply the configured attack to a single sample.

        Returns:
            (x_poisoned, y_poisoned, was_poisoned)
        """
        x_arr = np.asarray(x, dtype=float)
        y_int = int(y)

        if not self.is_active(t) or self.poison_rate <= 0.0:
            return x_arr, y_int, False

        # Decide whether to poison this sample at all.
        if not self._maybe_poison():
            return x_arr, y_int, False

        if self.mode == "label_flip":
            # Optionally only flip a target class.
            if self.target_class is not None and y_int != int(self.target_class):
                return x_arr, y_int, False

            if y_int not in (0, 1):
                raise ValueError("label_flip mode currently supports binary labels {0, 1}.")

            y_flipped = 1 - y_int
            return x_arr, int(y_flipped), True

        if self.mode == "trigger":
            x2 = x_arr.copy()
            n_features = x2.shape[0]
            for d in self.trigger_dims:
                if 0 <= d < n_features:
                    x2[d] = x2[d] + self.trigger_value
            y2 = self.target_label
            return x2, int(y2), True

        raise ValueError(f"Unknown poisoning mode: {self.mode!r}")

