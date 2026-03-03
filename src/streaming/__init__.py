"""Streaming utilities for online learning and TDA-based monitoring.

This subpackage provides:
- DataStream abstractions for time-ordered samples
- OnlineLearner wrappers around incremental sklearn models
- PoisoningAttack utilities for local data corruption
- WindowBuffer and TDAMonitor for TDA-based drift detection
"""

from .stream import DataStream, make_classification_stream
from .poison import PoisoningAttack
from .online_model import OnlineLearner
from .window_buffer import WindowBuffer
from .tda_monitor import TDAMonitor

__all__ = [
    "DataStream",
    "make_classification_stream",
    "PoisoningAttack",
    "OnlineLearner",
    "WindowBuffer",
    "TDAMonitor",
]

