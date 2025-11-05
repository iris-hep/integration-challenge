"""Background thread tracking Dask worker count and memory over time.

Enables core efficiency calculations, worker scaling analysis, and memory
usage tracking by recording worker count and memory usage every N seconds
during execution.
"""

import datetime
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def start_tracking(
    client,
    measurement_path: Path,
    interval: float = 1.0,
) -> Tuple[threading.Thread, Path]:
    """Start background thread tracking worker count and memory.

    Creates a signal file and spawns a daemon thread that writes worker
    count and memory usage to disk every `interval` seconds. Thread
    automatically stops when signal file is removed.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask distributed client
    measurement_path : Path
        Directory to write num_workers.txt and worker_memory.txt
    interval : float
        Seconds between samples (default: 1.0)

    Returns
    -------
    thread : threading.Thread
        Background tracking thread (already started)
    signal_file : Path
        Signal file path (remove to stop tracking)

    Examples
    --------
    >>> from dask.distributed import Client
    >>> from pathlib import Path
    >>>
    >>> client = Client()
    >>> measurement_path = Path("measurements/test")
    >>> tracker = start_tracking(client, measurement_path)
    >>>
    >>> # Run workload...
    >>>
    >>> stop_tracking(tracker)
    """
    measurement_path = Path(measurement_path)
    measurement_path.mkdir(parents=True, exist_ok=True)

    signal_file = measurement_path / "DASK_RUNNING"
    worker_count_file = measurement_path / "num_workers.txt"
    memory_file = measurement_path / "worker_memory.txt"

    # Create signal file
    signal_file.touch()

    # Start background thread
    thread = threading.Thread(
        target=_write_worker_metrics,
        args=(client, worker_count_file, memory_file, signal_file, interval),
        daemon=True,
    )
    thread.start()

    return thread, signal_file


def stop_tracking(tracker: Tuple[threading.Thread, Path]) -> None:
    """Stop worker tracking thread gracefully.

    Removes signal file and waits for thread to complete. Thread will
    finish its current sleep interval before stopping.

    Parameters
    ----------
    tracker : Tuple[threading.Thread, Path]
        Tuple of (thread, signal_file) returned by start_tracking()

    Examples
    --------
    >>> tracker = start_tracking(client, measurement_path)
    >>> # ... run workload ...
    >>> stop_tracking(tracker)
    """
    thread, signal_file = tracker

    # Remove signal file to stop thread
    if signal_file.exists():
        signal_file.unlink()

    # Wait for thread to finish (with timeout)
    thread.join(timeout=5.0)


def _write_worker_metrics(
    client,
    worker_count_file: Path,
    memory_file: Path,
    signal_file: Path,
    interval: float,
) -> None:
    """Background thread worker function (internal use only).

    Writes worker count and memory usage to files every `interval` seconds
    while signal file exists. Daemon thread - automatically stops when main exits.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask distributed client
    worker_count_file : Path
        File to write worker counts
    memory_file : Path
        File to write memory usage per worker
    signal_file : Path
        Signal file (thread stops when removed)
    interval : float
        Seconds between samples
    """
    with open(worker_count_file, "w") as f_workers, open(memory_file, "w") as f_memory:
        while signal_file.exists():
            try:
                # Get current scheduler info
                scheduler_info = client.scheduler_info()
                workers_dict = scheduler_info.get("workers", {})
                nworkers = len(workers_dict)

                timestamp = datetime.datetime.now()

                # Write worker count
                f_workers.write(f"{timestamp}, {nworkers}\n")
                f_workers.flush()

                # Write memory usage for each worker
                # Format: timestamp, worker_id, memory_bytes
                for worker_id, worker_info in workers_dict.items():
                    # Get memory from worker metrics
                    metrics = worker_info.get("metrics", {})
                    memory_bytes = metrics.get("memory", 0)

                    f_memory.write(f"{timestamp}, {worker_id}, {memory_bytes}\n")

                f_memory.flush()

            except Exception as e:
                # Log error but keep running
                error_msg = f"{datetime.datetime.now()}, ERROR: {e}\n"
                f_workers.write(error_msg)
                f_memory.write(error_msg)
                f_workers.flush()
                f_memory.flush()

            # Sleep for interval
            time.sleep(interval)


def load_worker_timeline(measurement_path: Path) -> Tuple[List[datetime.datetime], List[int]]:
    """Load worker count timeline from measurement.

    Parameters
    ----------
    measurement_path : Path
        Directory containing num_workers.txt

    Returns
    -------
    timestamps : List[datetime.datetime]
        List of sample timestamps
    worker_counts : List[int]
        Number of workers at each timestamp

    Raises
    ------
    FileNotFoundError
        If num_workers.txt doesn't exist

    Examples
    --------
    >>> from pathlib import Path
    >>>
    >>> timestamps, workers = load_worker_timeline(Path("measurements/test"))
    >>> avg_workers = calculate_time_averaged_workers(timestamps, workers)
    """
    timeline_file = measurement_path / "num_workers.txt"

    if not timeline_file.exists():
        raise FileNotFoundError(f"Worker timeline not found: {timeline_file}")

    timestamps = []
    worker_counts = []

    with open(timeline_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "ERROR" in line:
                continue

            try:
                timestamp_str, count_str = line.rsplit(",", 1)
                timestamp = datetime.datetime.fromisoformat(timestamp_str.strip())
                count = int(count_str.strip())

                timestamps.append(timestamp)
                worker_counts.append(count)
            except (ValueError, AttributeError):
                # Skip malformed lines
                continue

    return timestamps, worker_counts


def load_memory_timeline(measurement_path: Path) -> Tuple[List[datetime.datetime], dict]:
    """Load worker memory timeline from measurement.

    Parameters
    ----------
    measurement_path : Path
        Directory containing worker_memory.txt

    Returns
    -------
    timestamps : List[datetime.datetime]
        List of unique sample timestamps
    worker_memory : dict
        Dictionary mapping worker_id to list of (timestamp, memory_bytes) tuples

    Raises
    ------
    FileNotFoundError
        If worker_memory.txt doesn't exist

    Examples
    --------
    >>> from pathlib import Path
    >>>
    >>> timestamps, worker_memory = load_memory_timeline(Path("measurements/test"))
    >>> peak_memory_gb = calculate_peak_memory(worker_memory) / 1e9
    """
    memory_file = measurement_path / "worker_memory.txt"

    if not memory_file.exists():
        raise FileNotFoundError(f"Memory timeline not found: {memory_file}")

    timestamps_set = set()
    worker_memory = {}  # worker_id -> [(timestamp, memory_bytes), ...]

    with open(memory_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "ERROR" in line:
                continue

            try:
                parts = line.rsplit(",", 2)
                if len(parts) != 3:
                    continue

                timestamp_str, worker_id, memory_str = parts
                timestamp = datetime.datetime.fromisoformat(timestamp_str.strip())
                worker_id = worker_id.strip()
                memory_bytes = float(memory_str.strip())

                timestamps_set.add(timestamp)

                if worker_id not in worker_memory:
                    worker_memory[worker_id] = []

                worker_memory[worker_id].append((timestamp, memory_bytes))

            except (ValueError, AttributeError):
                # Skip malformed lines
                continue

    timestamps = sorted(timestamps_set)
    return timestamps, worker_memory


def calculate_time_averaged_workers(
    timestamps: List[datetime.datetime],
    worker_counts: List[int],
) -> float:
    """Calculate time-weighted average worker count.

    Uses trapezoidal integration to compute the average number of workers
    weighted by the time each count was active.

    Parameters
    ----------
    timestamps : List[datetime.datetime]
        Sample timestamps
    worker_counts : List[int]
        Worker count at each timestamp

    Returns
    -------
    avg_workers : float
        Time-averaged worker count

    Examples
    --------
    >>> timestamps = [t0, t1, t2]  # 0s, 100s, 300s
    >>> workers = [10, 50, 50]
    >>> avg = calculate_time_averaged_workers(timestamps, workers)
    >>> # avg ≈ (10*100 + 50*200) / 300 = 36.67
    """
    if len(timestamps) < 2:
        return float(worker_counts[0]) if worker_counts else 0.0

    # Convert timestamps to seconds since first sample
    t0 = timestamps[0]
    times = np.array([(t - t0).total_seconds() for t in timestamps])
    counts = np.array(worker_counts, dtype=float)

    # Calculate time intervals
    delta_t = np.diff(times)

    # Calculate workers * time for each interval (using midpoint)
    workers_times_time = [(counts[i] + counts[i + 1]) / 2 * delta_t[i]
                          for i in range(len(delta_t))]

    # Time-weighted average
    total_time = times[-1] - times[0]
    if total_time == 0:
        return counts[0]

    return sum(workers_times_time) / total_time


def calculate_peak_memory(worker_memory: dict) -> float:
    """Calculate peak memory usage across all workers.

    Parameters
    ----------
    worker_memory : dict
        Dictionary from load_memory_timeline: worker_id -> [(timestamp, memory_bytes), ...]

    Returns
    -------
    peak_memory_bytes : float
        Maximum memory usage observed across all workers

    Examples
    --------
    >>> timestamps, worker_memory = load_memory_timeline(measurement_path)
    >>> peak_gb = calculate_peak_memory(worker_memory) / 1e9
    """
    if not worker_memory:
        return 0.0

    all_memory_values = []
    for worker_id, timeline in worker_memory.items():
        for timestamp, memory_bytes in timeline:
            all_memory_values.append(memory_bytes)

    return max(all_memory_values) if all_memory_values else 0.0


def calculate_average_memory_per_worker(worker_memory: dict) -> float:
    """Calculate time-weighted average memory per worker.

    Parameters
    ----------
    worker_memory : dict
        Dictionary from load_memory_timeline: worker_id -> [(timestamp, memory_bytes), ...]

    Returns
    -------
    avg_memory_bytes : float
        Average memory per worker (averaged across workers and time)

    Examples
    --------
    >>> timestamps, worker_memory = load_memory_timeline(measurement_path)
    >>> avg_gb = calculate_average_memory_per_worker(worker_memory) / 1e9
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

        # Calculate memory * time for each interval (using midpoint)
        memory_times_time = [(memory[i] + memory[i + 1]) / 2 * delta_t[i]
                            for i in range(len(delta_t))]

        # Time-weighted average for this worker
        total_time = times[-1] - times[0]
        if total_time > 0:
            worker_avg = sum(memory_times_time) / total_time
            worker_averages.append(worker_avg)
        else:
            worker_averages.append(memory[0])

    # Average across all workers
    return np.mean(worker_averages) if worker_averages else 0.0
