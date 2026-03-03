"""Synthetic data generation and I/O for TDA pipeline."""

from typing import Optional, Tuple

import numpy as np
from pathlib import Path
from sklearn.datasets import make_circles, make_blobs
from sklearn.preprocessing import StandardScaler


def preprocess_point_cloud(X: np.ndarray, standardize: bool = False) -> np.ndarray:
    """Optionally standardize point cloud (zero mean, unit variance per feature)."""
    if not standardize:
        return X
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def generate_circles(
    n_samples: int = 200,
    noise: float = 0.05,
    factor: float = 0.5,
    random_state: Optional[int] = 42,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate concentric circles point cloud with known annulus topology.

    Returns:
        X: Point cloud of shape (n_samples, 2)
        y: Labels (0 or 1) for inner/outer circle
    """
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        shuffle=True,
        random_state=random_state,
    )
    X = preprocess_point_cloud(X, standardize=standardize)
    return X, y


def generate_random(
    n_samples: int = 200,
    random_state: Optional[int] = 42,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random points in [0, 1]^2 (negative control: no persistent loops)."""
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, 1, (n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    X = preprocess_point_cloud(X, standardize=standardize)
    return X, y


def generate_blobs(
    n_samples: int = 200,
    random_state: Optional[int] = 42,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate single Gaussian blob (negative control: contractible, trivial H1).

    Uses centers=1 and cluster_std=0.5 to produce a compact blob with no
    persistent loops. Intended for negative-control comparisons in TDA tests.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=1,
        n_features=2,
        cluster_std=0.5,
        random_state=random_state,
    )
    X = preprocess_point_cloud(X, standardize=standardize)
    return X, y


def save_point_cloud(X: np.ndarray, y: np.ndarray, path: str) -> None:
    """Save point cloud and labels to .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)


def load_point_cloud(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load point cloud and labels from .npz file."""
    data = np.load(path)
    return data["X"], data["y"]
