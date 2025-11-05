"""File inspection for input data characterization.

This module provides tools for extracting metadata from ROOT files using
distributed processing with Dask.

Example Usage
-------------
>>> from dask.distributed import Client
>>> from intccms.metrics.inspector import inspect_dataset_distributed, aggregate_statistics
>>>
>>> client = Client()
>>> results, errors = inspect_dataset_distributed(client, file_list)
>>> print(f"Success: {errors['successful']}/{errors['total_files']} files")
>>> stats = aggregate_statistics(results)
>>> print(f"Total events: {stats['total_events']:,}")
"""

from intccms.metrics.inspector.core import (
    inspect_file,
    inspect_dataset_distributed,
    get_total_events_distributed,
)
from intccms.metrics.inspector.aggregator import (
    aggregate_statistics,
    get_top_branches,
    group_by_dataset,
    compute_dataset_statistics,
    compute_compression_stats,
)
from intccms.metrics.inspector.integration import (
    extract_files_from_dataset_manager,
    get_dataset_file_counts,
)
from intccms.metrics.inspector import plot

__all__ = [
    # Core inspection
    "inspect_file",
    "inspect_dataset_distributed",
    "get_total_events_distributed",
    # Aggregation
    "aggregate_statistics",
    "get_top_branches",
    "group_by_dataset",
    "compute_dataset_statistics",
    "compute_compression_stats",
    # Integration
    "extract_files_from_dataset_manager",
    "get_dataset_file_counts",
    # Plotting
    "plot",
]
