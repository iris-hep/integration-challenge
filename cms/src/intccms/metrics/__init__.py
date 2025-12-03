"""Performance benchmarking and metrics collection.

This module provides tools for tracking performance metrics during CMS analysis workflows,
including data throughput, worker utilization, and I/O performance. Supports both full
physics analysis benchmarking and idap-200gbps style I/O throughput tests.

Key Components
--------------
- measurements: Save/load benchmark measurement results for reanalysis
- worker_tracker: Scheduler-based tracking of Dask worker count and memory over time
- collector: Aggregate and calculate derived performance metrics
- reporter: Generate human-readable summary reports
- visualization: Create performance plots and dashboards

Example Usage
-------------
>>> from dask.distributed import Client
>>> from intccms.metrics import (
...     save_measurement,
...     start_tracking,
...     stop_tracking,
...     save_worker_timeline,
... )
>>>
>>> client = Client()
>>>
>>> # Start worker tracking on scheduler
>>> client.run_on_scheduler(start_tracking, interval=1.0)
>>>
>>> # Run analysis with metrics enabled
>>> output, report, metrics = run_processor_workflow(...)
>>>
>>> # Stop tracking and save
>>> tracking_data = client.run_on_scheduler(stop_tracking)
>>> save_worker_timeline(tracking_data, measurement_path)
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
    save_worker_timeline,
    load_worker_timeline,
    calculate_time_averaged_workers,
    calculate_average_workers,
    calculate_peak_memory,
    calculate_average_memory_per_worker,
)
from intccms.metrics.collector import (
    collect_processing_metrics,
    format_bytes,
    format_time,
)
from intccms.metrics.reporter import (
    format_throughput_table,
    format_event_processing_table,
    format_resources_table,
    format_timing_table,
)
from intccms.metrics.visualization import (
    plot_worker_count_timeline,
    plot_memory_utilization_timeline,
    plot_cpu_utilization_timeline,
    plot_scaling_efficiency,
    plot_summary_dashboard,
)

__all__ = [
    # Measurements
    "save_measurement",
    "load_measurement",
    "list_measurements",
    # Worker tracking
    "start_tracking",
    "stop_tracking",
    "save_worker_timeline",
    "load_worker_timeline",
    "calculate_time_averaged_workers",
    "calculate_average_workers",
    "calculate_peak_memory",
    "calculate_average_memory_per_worker",
    # Collector
    "collect_processing_metrics",
    "format_bytes",
    "format_time",
    # Reporter
    "format_throughput_table",
    "format_event_processing_table",
    "format_resources_table",
    "format_timing_table",
    # Visualization
    "plot_worker_count_timeline",
    "plot_memory_utilization_timeline",
    "plot_cpu_utilization_timeline",
    "plot_scaling_efficiency",
    "plot_summary_dashboard",
]
