"""Tests for DataStream and make_classification_stream."""

import numpy as np

from src.streaming.stream import DataStream, make_classification_stream


def test_stream_shapes_and_len():
    n_steps = 200
    n_features = 4
    stream, (X_test, y_test) = make_classification_stream(
        n_steps=n_steps,
        n_features=n_features,
        random_state=0,
        class_sep=1.5,
        flip_y=0.0,
        test_size=0.25,
    )

    # Training length matches the underlying training set size.
    xs = []
    ys = []
    for t, x, y in stream:
        xs.append(x)
        ys.append(y)
        assert isinstance(t, int)
        assert x.shape == (n_features,)
        assert isinstance(int(y), int)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    assert len(stream) == xs.shape[0] == ys.shape[0]
    # Basic sanity check on test shapes as well.
    assert X_test.shape[1] == n_features
    assert X_test.shape[0] == y_test.shape[0]


def test_stream_determinism():
    n_steps = 100
    n_features = 3

    stream1, _ = make_classification_stream(
        n_steps=n_steps,
        n_features=n_features,
        random_state=123,
        class_sep=1.0,
        flip_y=0.0,
        test_size=0.2,
    )
    stream2, _ = make_classification_stream(
        n_steps=n_steps,
        n_features=n_features,
        random_state=123,
        class_sep=1.0,
        flip_y=0.0,
        test_size=0.2,
    )

    k = 10
    samples1 = [next(iter(stream1)) for _ in range(k)]
    samples2 = [next(iter(stream2)) for _ in range(k)]

    for (t1, x1, y1), (t2, x2, y2) in zip(samples1, samples2):
        assert t1 == t2
        assert np.allclose(x1, x2)
        assert y1 == y2


def test_reset():
    n_steps = 50
    n_features = 2
    stream, _ = make_classification_stream(
        n_steps=n_steps,
        n_features=n_features,
        random_state=7,
        class_sep=1.0,
        flip_y=0.0,
        test_size=0.2,
    )

    # Take a few samples, then reset and ensure we see the same prefix again.
    prefix1 = []
    it = iter(stream)
    for _ in range(5):
        prefix1.append(next(it))

    stream.reset()
    prefix2 = []
    it2 = iter(stream)
    for _ in range(5):
        prefix2.append(next(it2))

    for (t1, x1, y1), (t2, x2, y2) in zip(prefix1, prefix2):
        assert t1 == t2
        assert np.allclose(x1, x2)
        assert y1 == y2

