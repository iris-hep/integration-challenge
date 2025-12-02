"""Metrics collection and calculation from processing runs.

This module aggregates metrics from coffea reports, timing data, and worker
tracking data to calculate throughput, event rates, and efficiency metrics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from intccms.metrics.worker_tracker import (
    calculate_time_averaged_workers,
    load_worker_timeline,
)

logger = logging.getLogger(__name__)


def collect_processing_metrics(
    coffea_report: Dict[str, Any],
    t_start: float,
    t_end: float,
    custom_metrics: Optional[Dict[str, Any]] = None,
    measurement_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calculate all processing metrics from workflow run.

    Combines metrics from coffea's aggregated report and optional per-dataset
    custom metrics. The coffea report is converted to a "total" dataset entry,
    while custom metrics provide per-dataset breakdowns.

    Parameters
    ----------
    coffea_report : dict
        Coffea's aggregated report from Runner with keys:
        - bytesread: Total compressed bytes read
        - entries: Total events processed
        - processtime: Aggregated CPU time across chunks
        - chunks: Number of chunks processed
    t_start : float
        Start time (from time.perf_counter())
    t_end : float
        End time (from time.perf_counter())
    custom_metrics : dict, optional
        Per-dataset metrics from output["_metrics"] with structure:
        {dataset_name: {performance_counters: {...}, entries: N, duration: T}}
        If None, only coffea_report is used
    measurement_path : Path, optional
        Path to measurement directory containing worker_timeline.json
        If None, worker metrics will be skipped. cores_per_worker is
        auto-detected from worker tracking data.

    Returns
    -------
    metrics : dict
        Dictionary containing all calculated metrics:

        **Throughput Metrics:**
        - overall_rate_gbps: Overall data rate in Gbps
        - overall_rate_mbps: Overall data rate in MB/s
        - compression_ratio: Uncompressed / compressed ratio

        **Event Processing Metrics:**
        - event_rate_wall_khz: Events/sec from wall clock perspective (kHz)
        - event_rate_agg_khz: Events/sec from aggregated CPU time (kHz)
        - event_rate_core_hz: Events/sec/core (Hz per core)

        **Timing Metrics:**
        - wall_time: Real elapsed time (seconds)
        - total_cpu_time: Sum of all task durations (seconds)
        - num_chunks: Number of chunks processed
        - avg_cpu_time_per_chunk: Average CPU time per chunk (seconds)

        **Worker/Resource Metrics:**
        - avg_workers: Time-averaged worker count
        - peak_workers: Maximum workers observed
        - total_cores: Average cores available (auto-detected from workers)

        **Efficiency Metrics:**
        - core_efficiency: Fraction of cores actually used (0-1)
        - speedup_factor: Parallel speedup achieved

    Examples
    --------
    >>> from pathlib import Path
    >>> # With coffea report only (aggregated totals)
    >>> metrics = collect_processing_metrics(
    ...     coffea_report={'bytesread': 1e9, 'entries': 100000, 'processtime': 50.0},
    ...     t_start=0.0,
    ...     t_end=50.0,
    ...     measurement_path=Path("measurements/test")
    ... )
    >>>
    >>> # With both coffea report and custom per-dataset metrics
    >>> metrics = collect_processing_metrics(
    ...     coffea_report={'bytesread': 1e9, 'entries': 100000, 'processtime': 50.0},
    ...     t_start=0.0,
    ...     t_end=50.0,
    ...     custom_metrics={'TTbar': {...}, 'WJets': {...}},
    ...     measurement_path=Path("measurements/test")
    ... )
    >>> print(f"Throughput: {metrics['overall_rate_gbps']:.1f} Gbps")
    >>> print(f"Core Efficiency: {metrics['core_efficiency']*100:.1f}%")
    """
    # Calculate wall time
    wall_time = t_end - t_start

    # Extract number of chunks from coffea report
    num_chunks = coffea_report.get("chunks", 0)

    # Combine coffea report and custom metrics into unified structure
    # Coffea report (flat) → "total" dataset entry
    # Custom metrics (nested) → per-dataset entries
    combined_report = {}

    # Convert coffea report to "total" dataset if present
    if "bytesread" in coffea_report:
        combined_report["total"] = {
            "entries": coffea_report.get("entries", 0),
            "duration": coffea_report.get("processtime", wall_time),
            "performance_counters": {
                "num_requested_bytes": coffea_report.get("bytesread", 0)
            },
        }

    # Add custom per-dataset metrics if provided
    if custom_metrics:
        combined_report.update(custom_metrics)

    # Extract and aggregate metrics from combined report
    # Report structure: {dataset_name: {performance_counters: {...}, entries: N, duration: T}}
    total_bytes_compressed = 0
    total_bytes_uncompressed = 0
    total_events = 0
    total_cpu_time = 0

    for dataset_name, dataset_data in combined_report.items():
        # Skip non-dataset entries
        if not isinstance(dataset_data, dict):
            continue

        # Get performance counters
        perf_counters = dataset_data.get("performance_counters", {})
        total_bytes_compressed += perf_counters.get("num_requested_bytes", 0)
        # Note: Coffea doesn't track uncompressed bytes, we'd need to add this
        # For now, estimate from compression ratio of ~2.5x for NanoAOD

        # Get events and duration
        total_events += dataset_data.get("entries", 0)
        total_cpu_time += dataset_data.get("duration", 0)

    # Estimate uncompressed bytes (typical NanoAOD compression ~2.5x)
    # This is approximate - for exact values, we'd need to track in processor
    estimated_compression_ratio = 2.5
    total_bytes_uncompressed = total_bytes_compressed * estimated_compression_ratio

    # Calculate throughput metrics
    overall_rate_gbps = (total_bytes_compressed * 8 / 1e9) / wall_time if wall_time > 0 else 0
    overall_rate_mbps = (total_bytes_compressed / 1e6) / wall_time if wall_time > 0 else 0
    compression_ratio = estimated_compression_ratio

    # Calculate event rate metrics
    event_rate_wall_khz = (total_events / wall_time) / 1000 if wall_time > 0 else 0
    event_rate_agg_khz = (total_events / total_cpu_time) / 1000 if total_cpu_time > 0 else 0

    # Load worker tracking data if available
    avg_workers = None
    peak_workers = None
    total_cores = None
    core_efficiency = None
    event_rate_core_hz = None
    speedup_factor = None

    if measurement_path is not None:
        try:
            tracking_data = load_worker_timeline(measurement_path)
            worker_counts = tracking_data["worker_counts"]
            cores_per_worker = tracking_data.get("cores_per_worker")

            # Calculate worker metrics
            avg_workers = calculate_time_averaged_workers(worker_counts)
            peak_workers = max(worker_counts.values()) if worker_counts else 0

            # Calculate total cores if cores_per_worker was captured
            if cores_per_worker is not None and avg_workers is not None:
                total_cores = avg_workers * cores_per_worker

                # Calculate efficiency metrics
                if total_cores and wall_time > 0:
                    total_available_time = total_cores * wall_time
                    core_efficiency = total_cpu_time / total_available_time if total_available_time > 0 else 0

                    # Event rate per core
                    event_rate_core_hz = total_events / (wall_time * total_cores)
            else:
                logger.warning("cores_per_worker not found in tracking data - core metrics will be unavailable")

            # Calculate speedup factor
            speedup_factor = total_cpu_time / wall_time if wall_time > 0 else 0

        except FileNotFoundError:
            logger.warning(f"Worker timeline not found at {measurement_path}")
        except Exception as e:
            logger.warning(f"Failed to load worker tracking data: {e}")

    # Calculate chunk-level metrics
    avg_cpu_time_per_chunk = total_cpu_time / num_chunks if num_chunks > 0 else 0

    # Build metrics dictionary
    metrics = {
        # Throughput metrics
        "overall_rate_gbps": overall_rate_gbps,
        "overall_rate_mbps": overall_rate_mbps,
        "compression_ratio": compression_ratio,
        # Event processing metrics
        "total_events": total_events,
        "event_rate_wall_khz": event_rate_wall_khz,
        "event_rate_agg_khz": event_rate_agg_khz,
        "event_rate_core_hz": event_rate_core_hz,
        # Timing metrics
        "wall_time": wall_time,
        "total_cpu_time": total_cpu_time,
        "num_chunks": num_chunks,
        "avg_cpu_time_per_chunk": avg_cpu_time_per_chunk,
        # Data volume metrics
        "total_bytes_compressed": total_bytes_compressed,
        "total_bytes_uncompressed": total_bytes_uncompressed,
        # Worker/resource metrics
        "avg_workers": avg_workers,
        "peak_workers": peak_workers,
        "total_cores": total_cores,
        # Efficiency metrics
        "core_efficiency": core_efficiency,
        "speedup_factor": speedup_factor,
    }

    return metrics


def format_bytes(num_bytes: float) -> str:
    """Format bytes in human-readable units.

    Parameters
    ----------
    num_bytes : float
        Number of bytes

    Returns
    -------
    formatted : str
        Formatted string (e.g., "1.23 GB", "456.7 MB")

    Examples
    --------
    >>> format_bytes(1234567890)
    '1.15 GB'
    >>> format_bytes(1234567)
    '1.18 MB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    """Format time in human-readable units.

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    formatted : str
        Formatted string (e.g., "1h 23m 45s", "45.2s")

    Examples
    --------
    >>> format_time(45.2)
    '45.2s'
    >>> format_time(3723)
    '1h 2m 3s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60

    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
