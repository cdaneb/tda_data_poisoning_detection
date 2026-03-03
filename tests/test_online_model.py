"""Tests for the OnlineLearner wrapper around SGDClassifier."""

import numpy as np
from sklearn.datasets import make_classification

from src.streaming.online_model import OnlineLearner


def test_online_learner_learns_separable_data():
    """OnlineLearner should achieve reasonable accuracy on a simple problem."""
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        flip_y=0.0,
        random_state=0,
    )

    learner = OnlineLearner(n_features=5, classes=(0, 1), random_state=0)

    # Train on the first half of the stream.
    for t in range(100):
        learner.update(X[t], int(y[t]), t=t)

    # Evaluate on a held-out slice.
    X_test = X[100:150]
    y_test = y[100:150]
    preds = np.array([learner.predict(x) for x in X_test])
    accuracy = np.mean(preds == y_test)

    # Loose threshold to remain robust across sklearn versions.
    assert accuracy >= 0.7


def test_online_learner_history_and_probabilities():
    """History entries should contain the expected fields and valid probabilities."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0, 1, 1])

    learner = OnlineLearner(n_features=2, classes=(0, 1), random_state=0)
    for t, (x_i, y_i) in enumerate(zip(X, y)):
        learner.update(x_i, int(y_i), t=t)

    assert len(learner.history) == 3

    last = learner.history[-1]
    assert set(last.keys()) == {"t", "y_true", "y_pred", "y_prob", "correct"}
    assert last["t"] == 2
    assert isinstance(last["correct"], bool)
    assert 0.0 <= last["y_prob"] <= 1.0

    # predict_proba should agree with the stored probability for the same point.
    p = learner.predict_proba(X[-1])
    assert np.isclose(p, last["y_prob"], atol=1e-6)

