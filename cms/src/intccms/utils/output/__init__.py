"""
Output management utilities.

This package provides centralized management of output directories and
histogram I/O with filtering capabilities.
"""

from .directories import OutputDirectoryManager
from .histograms import (
    filter_empty_histograms,
    filter_invalid_systematics,
    load_histograms_from_pickle,
    save_histograms_to_pickle,
    save_histograms_to_root,
)

__all__ = [
    # Directory management
    "OutputDirectoryManager",
    # Histogram I/O (public convenience functions)
    "load_histograms_from_pickle",
    "save_histograms_to_pickle",
    "save_histograms_to_root",
    # Filtering functions
    "filter_invalid_systematics",
    "filter_empty_histograms",
]
