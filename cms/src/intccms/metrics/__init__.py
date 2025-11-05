"""Performance benchmarking and metrics collection.

This module provides tools for tracking performance metrics during CMS analysis workflows,
including data throughput, worker utilization, and I/O performance. Supports both full
physics analysis benchmarking and idap-200gbps style I/O throughput tests.

Key Components
--------------
- measurements: Save/load benchmark measurement results for reanalysis
- worker_tracker: Background thread monitoring Dask worker count over time
- collector: Aggregate and calculate derived performance metrics
- reporter: Generate human-readable summary reports
- visualization: Create performance plots and dashboards

Example Usage
-------------
>>> from intccms.metrics import save_measurement, start_tracking
>>>
>>> # Start worker tracking
>>> tracker = start_tracking(dask_client, measurement_path)
>>>
>>> # Run analysis with metrics enabled
>>> output, report, metrics = run_processor_workflow(...)
>>>
>>> # Stop tracking and save
>>> stop_tracking(tracker)
>>> save_measurement(output, t0, t1, measurement_path)
"""

from intccms.metrics.measurements import (
    save_measurement,
    load_measurement,
    list_measurements,
)
from intccms.metrics.worker_tracker import (
    start_tracking,
    stop_tracking,
    load_worker_timeline,
    calculate_time_averaged_workers,
    load_memory_timeline,
    calculate_peak_memory,
    calculate_average_memory_per_worker,
)

__all__ = [
    # Measurements
    "save_measurement",
    "load_measurement",
    "list_measurements",
    # Worker tracking
    "start_tracking",
    "stop_tracking",
    "load_worker_timeline",
    "calculate_time_averaged_workers",
    # Memory tracking
    "load_memory_timeline",
    "calculate_peak_memory",
    "calculate_average_memory_per_worker",
]
