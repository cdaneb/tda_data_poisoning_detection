"""TDA pipeline: data generation, persistent homology, visualization, summaries."""

from .data import (
    generate_circles,
    generate_random,
    generate_blobs,
    preprocess_point_cloud,
    save_point_cloud,
    load_point_cloud,
)
from .homology import compute_persistence
from .viz import plot_point_cloud, plot_persistence_diagram
from .summaries import (
    max_persistence,
    persistence_count,
    persistence_entropy,
    summarize_by_dimension,
)
from .sliding_window import sliding_windows, takens_embedding, sliding_window_persistence

__all__ = [
    "generate_circles",
    "generate_random",
    "generate_blobs",
    "preprocess_point_cloud",
    "save_point_cloud",
    "load_point_cloud",
    "compute_persistence",
    "plot_point_cloud",
    "plot_persistence_diagram",
    "max_persistence",
    "persistence_count",
    "persistence_entropy",
    "summarize_by_dimension",
    "sliding_windows",
    "takens_embedding",
    "sliding_window_persistence",
]
