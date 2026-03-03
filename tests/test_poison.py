"""Tests for the PoisoningAttack class."""

import numpy as np

from src.streaming.poison import PoisoningAttack


def test_inactive_by_default():
    """With default start/end (None), the attack is never active."""
    attack = PoisoningAttack()
    x = np.array([1.0, 2.0])
    y = 0
    for t in range(0, 10):
        assert not attack.is_active(t)
        x2, y2, poisoned = attack.apply(x, y, t)
        assert np.allclose(x2, x)
        assert y2 == y
        assert poisoned is False


def test_label_flip_window_only():
    """Label-flip attack should only apply within the configured time window."""
    attack = PoisoningAttack(
        mode="label_flip",
        start_t=5,
        end_t=10,
        poison_rate=1.0,
        random_state=0,
    )
    x = np.array([0.0, 0.0])

    for t in range(0, 15):
        # Alternate labels to ensure we cover both 0 and 1.
        y = t % 2
        x2, y2, poisoned = attack.apply(x, y, t)
        if 5 <= t <= 10:
            assert poisoned is True
            # Binary flip: 0->1, 1->0
            assert y2 == 1 - y
            assert np.allclose(x2, x)
        else:
            assert poisoned is False
            assert y2 == y
            assert np.allclose(x2, x)


def test_trigger_applies_and_sets_label():
    """Trigger attack should modify selected dims and force the target label."""
    attack = PoisoningAttack(
        mode="trigger",
        start_t=0,
        end_t=100,
        poison_rate=1.0,
        trigger_value=0.5,
        trigger_dims=[0, 2],
        target_label=7,
        random_state=0,
    )

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 0
    x2, y2, poisoned = attack.apply(x, y, t=5)

    assert poisoned is True
    # Dims 0 and 2 are shifted; others unchanged.
    assert np.isclose(x2[0], x[0] + 0.5)
    assert np.isclose(x2[2], x[2] + 0.5)
    assert np.isclose(x2[1], x[1])
    assert np.isclose(x2[3], x[3])
    # Label forced to target_label.
    assert y2 == 7


def test_poison_rate_probabilistic():
    """With fixed RNG and poison_rate ~0.5, fraction of poisoned samples is reasonable."""
    N = 200
    attack = PoisoningAttack(
        mode="label_flip",
        start_t=0,
        end_t=N - 1,
        poison_rate=0.5,
        random_state=123,
    )
    x = np.array([0.0, 0.0])
    y = 0

    poisoned_count = 0
    for t in range(N):
        _, _, poisoned = attack.apply(x, y, t)
        if poisoned:
            poisoned_count += 1

    # Loose bounds to remain robust across RNG implementations.
    assert 0.3 * N <= poisoned_count <= 0.7 * N

