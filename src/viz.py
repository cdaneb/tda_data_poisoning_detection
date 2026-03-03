"""Visualization for point clouds and persistence diagrams."""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import persim


def plot_point_cloud(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot a 2D point cloud, optionally colored by labels."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    if y is not None:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.7, **kwargs)
    else:
        ax.scatter(X[:, 0], X[:, 1], alpha=0.7, **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    return ax


def plot_persistence_diagram(
    dgms: list,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot persistence diagrams using persim."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    persim.plot_diagrams(dgms, ax=ax, show=False, **kwargs)
    return ax
