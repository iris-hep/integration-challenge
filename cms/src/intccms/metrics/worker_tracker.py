"""Scheduler-based tracking of Dask worker count and memory over time.

Enables core efficiency calculations, worker scaling analysis, and memory
usage tracking by recording worker count and memory usage on the scheduler
using async tasks.
"""

import asyncio
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Scheduler-Side Functions (Run via client.run_on_scheduler)
# =============================================================================


def start_tracking(dask_scheduler, interval: float = 1.0):
    """Start tracking worker count and memory on scheduler.

    Creates an async task on the scheduler that records worker count and
    memory usage every `interval` seconds. Data is stored in-memory on
    the scheduler. Also captures cores_per_worker from the first worker.

    Parameters
    ----------
    dask_scheduler : distributed.Scheduler
        Dask scheduler object
    interval : float
        Seconds between samples (default: 1.0)

    Examples
    --------
    >>> from dask.distributed import Client
    >>> client = Client()
    >>> client.run_on_scheduler(start_tracking, interval=1.0)
    """
    # Initialize tracking state on scheduler
    dask_scheduler.worker_counts = {}
    dask_scheduler.worker_memory = {}
    dask_scheduler.worker_memory_limit = {}  # Track memory limits
    dask_scheduler.worker_active_tasks = {}  # Track active tasks
    dask_scheduler.track_count = True

    # Capture cores_per_worker from first worker
    if dask_scheduler.workers:
        first_worker = list(dask_scheduler.workers.values())[0]
        dask_scheduler.cores_per_worker = first_worker.nthreads
    else:
        dask_scheduler.cores_per_worker = None

    async def track_worker_metrics():
        """Async task to track worker metrics."""
        while dask_scheduler.track_count:
            timestamp = datetime.datetime.now()

            # Record worker count
            num_workers = len(dask_scheduler.workers)
            dask_scheduler.worker_counts[timestamp] = num_workers

            # Record memory, memory limit, and active tasks for each worker
            for worker_id, worker_state in dask_scheduler.workers.items():
                # Get memory from worker metrics (safe dict access)
                memory_bytes = worker_state.metrics.get("memory", 0)

                # Get memory limit (direct attribute, fallback to 0)
                memory_limit = getattr(worker_state, "memory_limit", 0)

                # Get active tasks (processing is a set, fallback to empty set)
                processing = getattr(worker_state, "processing", set())
                active_tasks = len(processing) if processing else 0

                # Initialize worker-specific lists if not present
                if worker_id not in dask_scheduler.worker_memory:
                    dask_scheduler.worker_memory[worker_id] = []
                if worker_id not in dask_scheduler.worker_memory_limit:
                    dask_scheduler.worker_memory_limit[worker_id] = []
                if worker_id not in dask_scheduler.worker_active_tasks:
                    dask_scheduler.worker_active_tasks[worker_id] = []

                # Append timestamped data
                dask_scheduler.worker_memory[worker_id].append(
                    (timestamp, memory_bytes)
                )
                dask_scheduler.worker_memory_limit[worker_id].append(
                    (timestamp, memory_limit)
                )
                dask_scheduler.worker_active_tasks[worker_id].append(
                    (timestamp, active_tasks)
                )

            # Sleep for interval
            await asyncio.sleep(interval)

    # Create and start the tracking task
    asyncio.create_task(track_worker_metrics())


def stop_tracking(dask_scheduler) -> Dict:
    """Stop tracking and return collected data.

    Stops the tracking task and returns all collected worker count,
    memory data, memory limits, active tasks, and cores_per_worker information.

    Parameters
    ----------
    dask_scheduler : distributed.Scheduler
        Dask scheduler object

    Returns
    -------
    tracking_data : dict
        Dictionary containing:
        - worker_counts: {datetime -> count} mapping
        - worker_memory: {worker_id -> [(datetime, memory_bytes), ...]}
        - worker_memory_limit: {worker_id -> [(datetime, memory_limit_bytes), ...]}
        - worker_active_tasks: {worker_id -> [(datetime, num_active_tasks), ...]}
        - cores_per_worker: int or None, number of threads per worker

    Examples
    --------
    >>> from dask.distributed import Client
    >>> client = Client()
    >>> client.run_on_scheduler(start_tracking)
    >>> # ... run workload ...
    >>> data = client.run_on_scheduler(stop_tracking)
    """
    # Stop tracking
    dask_scheduler.track_count = False

    # Retrieve and return data
    tracking_data = {
        "worker_counts": dask_scheduler.worker_counts,
        "worker_memory": dask_scheduler.worker_memory,
        "worker_memory_limit": getattr(dask_scheduler, "worker_memory_limit", {}),
        "worker_active_tasks": getattr(dask_scheduler, "worker_active_tasks", {}),
        "cores_per_worker": getattr(dask_scheduler, "cores_per_worker", None),
    }

    return tracking_data


# =============================================================================
# Client-Side Functions (Save/Load/Calculate)
# =============================================================================


def save_worker_timeline(
    tracking_data: Dict,
    measurement_path: Path,
) -> None:
    """Save worker tracking data to JSON file.

    Saves worker count and memory timelines to a structured JSON file
    on the client side.

    Parameters
    ----------
    tracking_data : dict
        Data returned from stop_tracking()
    measurement_path : Path
        Directory to save worker_timeline.json

    Examples
    --------
    >>> data = client.run_on_scheduler(stop_tracking)
    >>> save_worker_timeline(data, Path("measurements/test"))
    """
    measurement_path = Path(measurement_path)
    measurement_path.mkdir(parents=True, exist_ok=True)

    # Convert datetime keys to ISO format strings
    worker_counts = tracking_data["worker_counts"]
    worker_memory = tracking_data["worker_memory"]
    worker_memory_limit = tracking_data.get("worker_memory_limit", {})
    worker_active_tasks = tracking_data.get("worker_active_tasks", {})

    # Build JSON-serializable structure
    timeline_data = {
        "metadata": {
            "num_samples": len(worker_counts),
            "start_time": None,
            "end_time": None,
            "num_workers_range": None,
            "cores_per_worker": tracking_data.get("cores_per_worker"),
        },
        "worker_counts": [],
        "worker_memory": {},
        "worker_memory_limit": {},
        "worker_active_tasks": {},
    }

    # Convert worker counts timeline
    if worker_counts:
        sorted_timestamps = sorted(worker_counts.keys())
        timeline_data["metadata"]["start_time"] = sorted_timestamps[0].isoformat()
        timeline_data["metadata"]["end_time"] = sorted_timestamps[-1].isoformat()
        timeline_data["metadata"]["num_workers_range"] = [
            min(worker_counts.values()),
            max(worker_counts.values()),
        ]

        for timestamp in sorted_timestamps:
            timeline_data["worker_counts"].append({
                "timestamp": timestamp.isoformat(),
                "worker_count": worker_counts[timestamp],
            })

    # Convert worker memory timeline
    for worker_id, memory_timeline in worker_memory.items():
        timeline_data["worker_memory"][worker_id] = [
            {
                "timestamp": timestamp.isoformat(),
                "memory_bytes": memory_bytes,
            }
            for timestamp, memory_bytes in memory_timeline
        ]

    # Convert worker memory limit timeline
    for worker_id, limit_timeline in worker_memory_limit.items():
        timeline_data["worker_memory_limit"][worker_id] = [
            {
                "timestamp": timestamp.isoformat(),
                "memory_limit_bytes": memory_limit_bytes,
            }
            for timestamp, memory_limit_bytes in limit_timeline
        ]

    # Convert worker active tasks timeline
    for worker_id, tasks_timeline in worker_active_tasks.items():
        timeline_data["worker_active_tasks"][worker_id] = [
            {
                "timestamp": timestamp.isoformat(),
                "active_tasks": num_tasks,
            }
            for timestamp, num_tasks in tasks_timeline
        ]

    # Save to JSON
    output_file = measurement_path / "worker_timeline.json"
    with open(output_file, "w") as f:
        json.dump(timeline_data, f, indent=2)


def load_worker_timeline(measurement_path: Path) -> Dict:
    """Load worker tracking data from JSON file.

    Loads and deserializes worker count and memory timelines from JSON.

    Parameters
    ----------
    measurement_path : Path
        Directory containing worker_timeline.json

    Returns
    -------
    tracking_data : dict
        Dictionary containing:
        - worker_counts: {datetime -> count} mapping
        - worker_memory: {worker_id -> [(datetime, memory_bytes), ...]}
        - worker_memory_limit: {worker_id -> [(datetime, memory_limit_bytes), ...]}
        - worker_active_tasks: {worker_id -> [(datetime, num_active_tasks), ...]}
        - cores_per_worker: int or None, number of threads per worker

    Raises
    ------
    FileNotFoundError
        If worker_timeline.json doesn't exist

    Examples
    --------
    >>> from pathlib import Path
    >>> data = load_worker_timeline(Path("measurements/test"))
    >>> avg_workers = calculate_time_averaged_workers(data["worker_counts"])
    """
    timeline_file = measurement_path / "worker_timeline.json"

    if not timeline_file.exists():
        raise FileNotFoundError(f"Worker timeline not found: {timeline_file}")

    with open(timeline_file, "r") as f:
        timeline_data = json.load(f)

    # Convert ISO strings back to datetime objects
    worker_counts = {}
    for entry in timeline_data.get("worker_counts", []):
        timestamp = datetime.datetime.fromisoformat(entry["timestamp"])
        worker_counts[timestamp] = entry["worker_count"]

    # Convert worker memory timeline
    worker_memory = {}
    for worker_id, memory_timeline in timeline_data.get("worker_memory", {}).items():
        worker_memory[worker_id] = [
            (
                datetime.datetime.fromisoformat(entry["timestamp"]),
                entry["memory_bytes"],
            )
            for entry in memory_timeline
        ]

    # Convert worker memory limit timeline
    worker_memory_limit = {}
    for worker_id, limit_timeline in timeline_data.get("worker_memory_limit", {}).items():
        worker_memory_limit[worker_id] = [
            (
                datetime.datetime.fromisoformat(entry["timestamp"]),
                entry["memory_limit_bytes"],
            )
            for entry in limit_timeline
        ]

    # Convert worker active tasks timeline
    worker_active_tasks = {}
    for worker_id, tasks_timeline in timeline_data.get("worker_active_tasks", {}).items():
        worker_active_tasks[worker_id] = [
            (
                datetime.datetime.fromisoformat(entry["timestamp"]),
                entry["active_tasks"],
            )
            for entry in tasks_timeline
        ]

    # Load cores_per_worker from metadata
    cores_per_worker = timeline_data.get("metadata", {}).get("cores_per_worker")

    return {
        "worker_counts": worker_counts,
        "worker_memory": worker_memory,
        "worker_memory_limit": worker_memory_limit,
        "worker_active_tasks": worker_active_tasks,
        "cores_per_worker": cores_per_worker,
    }


def calculate_time_averaged_workers(worker_counts: Dict[datetime.datetime, int]) -> float:
    """Calculate time-weighted average worker count.

    Uses trapezoidal integration to compute the average number of workers
    weighted by the time each count was active.

    Parameters
    ----------
    worker_counts : dict
        Mapping from datetime to worker count

    Returns
    -------
    avg_workers : float
        Time-averaged worker count

    Examples
    --------
    >>> data = load_worker_timeline(measurement_path)
    >>> avg_workers = calculate_time_averaged_workers(data["worker_counts"])
    """
    if not worker_counts:
        return 0.0

    if len(worker_counts) < 2:
        return float(list(worker_counts.values())[0])

    # Sort by timestamp
    sorted_items = sorted(worker_counts.items())
    timestamps = [t for t, _ in sorted_items]
    counts = [c for _, c in sorted_items]

    # Convert to seconds since first sample
    t0 = timestamps[0]
    times = np.array([(t - t0).total_seconds() for t in timestamps])
    worker_array = np.array(counts, dtype=float)

    # Calculate time intervals
    delta_t = np.diff(times)

    # Trapezoidal integration: area = (y1 + y2) / 2 * delta_t
    workers_times_time = [
        (worker_array[i] + worker_array[i + 1]) / 2 * delta_t[i]
        for i in range(len(delta_t))
    ]

    # Time-weighted average
    total_time = times[-1] - times[0]
    if total_time == 0:
        return worker_array[0]

    return sum(workers_times_time) / total_time


def calculate_average_workers(worker_counts: Dict[datetime.datetime, int]) -> float:
    """Calculate average number of workers using trapezoidal integration.

    This is the implementation from Mo's example. Identical to
    calculate_time_averaged_workers() but kept as separate function
    for compatibility.

    Parameters
    ----------
    worker_counts : dict
        Mapping from datetime to worker count

    Returns
    -------
    avg_workers : float
        Time-averaged worker count

    Examples
    --------
    >>> data = client.run_on_scheduler(stop_tracking)
    >>> avg = calculate_average_workers(data["worker_counts"])
    """
    if not worker_counts:
        return 0.0

    worker_info = sorted(worker_counts.items())

    if len(worker_info) < 2:
        return float(worker_info[0][1])

    nworker_dt = 0
    for (t0, nw0), (t1, nw1) in zip(worker_info[:-1], worker_info[1:]):
        nworker_dt += (nw1 + nw0) / 2 * (t1 - t0).total_seconds()

    total_seconds = (worker_info[-1][0] - worker_info[0][0]).total_seconds()
    return nworker_dt / total_seconds


def calculate_peak_memory(worker_memory: Dict[str, List[Tuple]]) -> float:
    """Calculate peak memory usage across all workers.

    Parameters
    ----------
    worker_memory : dict
        Dictionary from tracking data: worker_id -> [(timestamp, memory_bytes), ...]

    Returns
    -------
    peak_memory_bytes : float
        Maximum memory usage observed across all workers

    Examples
    --------
    >>> data = load_worker_timeline(measurement_path)
    >>> peak_gb = calculate_peak_memory(data["worker_memory"]) / 1e9
    """
    if not worker_memory:
        return 0.0

    all_memory_values = []
    for worker_id, timeline in worker_memory.items():
        for timestamp, memory_bytes in timeline:
            all_memory_values.append(memory_bytes)

    return max(all_memory_values) if all_memory_values else 0.0


def calculate_average_memory_per_worker(worker_memory: Dict[str, List[Tuple]]) -> float:
    """Calculate time-weighted average memory per worker.

    Computes time-weighted average for each worker, then averages across workers.

    Parameters
    ----------
    worker_memory : dict
        Dictionary from tracking data: worker_id -> [(timestamp, memory_bytes), ...]

    Returns
    -------
    avg_memory_bytes : float
        Average memory per worker (averaged across workers and time)

    Examples
    --------
    >>> data = load_worker_timeline(measurement_path)
    >>> avg_gb = calculate_average_memory_per_worker(data["worker_memory"]) / 1e9
    """
    if not worker_memory:
        return 0.0

    worker_averages = []

    for worker_id, timeline in worker_memory.items():
        if len(timeline) < 2:
            if timeline:
                worker_averages.append(timeline[0][1])
            continue

        # Sort by timestamp
        timeline = sorted(timeline, key=lambda x: x[0])

        # Extract timestamps and memory values
        timestamps = [t for t, m in timeline]
        memory_values = [m for t, m in timeline]

        # Convert to seconds since first sample
        t0 = timestamps[0]
        times = np.array([(t - t0).total_seconds() for t in timestamps])
        memory = np.array(memory_values, dtype=float)

        # Calculate time intervals
        delta_t = np.diff(times)

        # Trapezoidal integration
        memory_times_time = [
            (memory[i] + memory[i + 1]) / 2 * delta_t[i]
            for i in range(len(delta_t))
        ]

        # Time-weighted average for this worker
        total_time = times[-1] - times[0]
        if total_time > 0:
            worker_avg = sum(memory_times_time) / total_time
            worker_averages.append(worker_avg)
        else:
            worker_averages.append(memory[0])

    # Average across all workers
    return np.mean(worker_averages) if worker_averages else 0.0
