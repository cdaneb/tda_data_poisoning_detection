"""Minimal tests for TDA pipeline."""

import numpy as np

from src.data import generate_circles, generate_random, generate_blobs
from src.homology import compute_persistence
from src.summaries import max_persistence, persistence_count, persistence_entropy

# Relative comparison factor: circles should have meaningfully more H1 than controls
K = 2.0
RANDOM_STATE = 42


def test_circles_h1_dominates_random():
    """maxH1(circles) > k * maxH1(random) — circles have stronger loop structure."""
    X_c, _ = generate_circles(n_samples=200, noise=0.05, factor=0.5, random_state=RANDOM_STATE)
    X_r, _ = generate_random(n_samples=200, random_state=RANDOM_STATE)

    dgms_c = compute_persistence(X_c, maxdim=1)["dgms"]
    dgms_r = compute_persistence(X_r, maxdim=1)["dgms"]

    max_h1_c = max_persistence(dgms_c[1]) if len(dgms_c[1]) > 0 else 0.0
    max_h1_r = max_persistence(dgms_r[1]) if len(dgms_r[1]) > 0 else 0.0

    assert max_h1_c > K * max_h1_r, f"circles maxH1={max_h1_c} should exceed {K}*random maxH1={max_h1_r}"


def test_circles_h1_dominates_blobs():
    """maxH1(circles) > k * maxH1(blobs) — circles have stronger loop structure."""
    X_c, _ = generate_circles(n_samples=200, noise=0.05, factor=0.5, random_state=RANDOM_STATE)
    X_b, _ = generate_blobs(n_samples=200, random_state=RANDOM_STATE)

    dgms_c = compute_persistence(X_c, maxdim=1)["dgms"]
    dgms_b = compute_persistence(X_b, maxdim=1)["dgms"]

    max_h1_c = max_persistence(dgms_c[1]) if len(dgms_c[1]) > 0 else 0.0
    max_h1_b = max_persistence(dgms_b[1]) if len(dgms_b[1]) > 0 else 0.0

    assert max_h1_c > K * max_h1_b, f"circles maxH1={max_h1_c} should exceed {K}*blobs maxH1={max_h1_b}"


def test_circles_has_h1_points():
    """Circles yield at least one H1 point (robust absolute check)."""
    X, _ = generate_circles(n_samples=200, noise=0.05, factor=0.5, random_state=RANDOM_STATE)
    result = compute_persistence(X, maxdim=1)
    h1 = result["dgms"][1]
    assert len(h1) >= 1, "circles should have at least one H1 point"


def test_empty_diagram_summaries():
    """Empty diagrams return max_persistence=0.0, count=0, entropy=0.0."""
    empty = np.array([]).reshape(0, 2)
    assert max_persistence(empty) == 0.0
    assert persistence_count(empty, threshold=0.1) == 0
    assert persistence_entropy(empty) == 0.0


def test_all_infinite_bars_summaries():
    """Diagrams with only infinite bars (death=inf) return same as empty."""
    inf_bars = np.array([[0.0, np.inf], [0.1, np.inf]])
    assert max_persistence(inf_bars) == 0.0
    assert persistence_count(inf_bars, threshold=0.1) == 0
    assert persistence_entropy(inf_bars) == 0.0


def test_persistence_entropy_finite_diagram():
    """persistence_entropy on finite H1 diagram returns float, no TypeError."""
    dgm = np.array([[0.0, 1.0], [0.2, 0.9]])
    result = persistence_entropy(dgm)
    assert isinstance(result, float)
    try:
        import persim  # noqa: F401
        assert np.isfinite(result)
    except ImportError:
        pass
