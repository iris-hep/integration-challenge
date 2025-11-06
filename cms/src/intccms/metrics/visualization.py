"""Visualization functions for metrics data.

Provides plotting functions for worker timelines, memory utilization,
scaling efficiency, and summary dashboards.
"""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from intccms.metrics.worker_tracker import load_worker_timeline


def plot_worker_count_timeline(
    tracking_data: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Worker Count Over Time",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot worker count over time.

    Parameters
    ----------
    tracking_data : dict
        Data from load_worker_timeline()
    output_path : Path, optional
        Path to save figure (if None, doesn't save)
    figsize : tuple
        Figure size (width, height) in inches
    title : str
        Plot title

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Examples
    --------
    >>> from pathlib import Path
    >>> data = load_worker_timeline(Path("benchmarks/latest"))
    >>> fig, ax = plot_worker_count_timeline(data)
    >>> plt.show()
    """
    worker_counts = tracking_data["worker_counts"]

    if not worker_counts:
        raise ValueError("No worker count data available")

    # Sort by timestamp
    sorted_items = sorted(worker_counts.items())
    timestamps = [t for t, _ in sorted_items]
    counts = [c for _, c in sorted_items]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(timestamps, counts, marker='o', linestyle='-', linewidth=2, markersize=4)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Workers")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_memory_utilization_timeline(
    tracking_data: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Memory Utilization Over Time",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot memory utilization percentage over time (averaged across workers).

    Parameters
    ----------
    tracking_data : dict
        Data from load_worker_timeline() with worker_memory and worker_memory_limit
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Examples
    --------
    >>> from pathlib import Path
    >>> data = load_worker_timeline(Path("benchmarks/latest"))
    >>> fig, ax = plot_memory_utilization_timeline(data)
    >>> plt.show()
    """
    worker_memory = tracking_data.get("worker_memory", {})
    worker_memory_limit = tracking_data.get("worker_memory_limit", {})

    if not worker_memory or not worker_memory_limit:
        raise ValueError("Memory or memory limit data not available")

    # Collect all unique timestamps
    all_timestamps = set()
    for worker_id in worker_memory.keys():
        for timestamp, _ in worker_memory[worker_id]:
            all_timestamps.add(timestamp)

    sorted_timestamps = sorted(all_timestamps)

    # Calculate memory utilization % at each timestamp
    utilization_pct = []
    utilization_min = []
    utilization_max = []

    for timestamp in sorted_timestamps:
        worker_utils = []

        for worker_id in worker_memory.keys():
            # Find memory usage at this timestamp
            mem_data = worker_memory[worker_id]
            limit_data = worker_memory_limit.get(worker_id, [])

            # Find closest timestamp (should be exact match)
            mem_value = None
            for t, m in mem_data:
                if t == timestamp:
                    mem_value = m
                    break

            limit_value = None
            for t, l in limit_data:
                if t == timestamp:
                    limit_value = l
                    break

            if mem_value is not None and limit_value is not None and limit_value > 0:
                util_pct = (mem_value / limit_value) * 100
                worker_utils.append(util_pct)

        if worker_utils:
            utilization_pct.append(np.mean(worker_utils))
            utilization_min.append(np.min(worker_utils))
            utilization_max.append(np.max(worker_utils))
        else:
            utilization_pct.append(0)
            utilization_min.append(0)
            utilization_max.append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot mean with confidence band
    ax.plot(sorted_timestamps, utilization_pct, linewidth=2, label='Mean', color='C0')
    ax.fill_between(sorted_timestamps, utilization_min, utilization_max,
                     alpha=0.3, label='Min-Max Range', color='C0')

    ax.set_xlabel("Time")
    ax.set_ylabel("Memory Utilization (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_cpu_utilization_timeline(
    tracking_data: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 4),
    title: str = "CPU Utilization Over Time",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot CPU utilization over time (based on active tasks / available threads).

    Parameters
    ----------
    tracking_data : dict
        Data from load_worker_timeline() with worker_active_tasks and cores_per_worker
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Examples
    --------
    >>> from pathlib import Path
    >>> data = load_worker_timeline(Path("benchmarks/latest"))
    >>> fig, ax = plot_cpu_utilization_timeline(data)
    >>> plt.show()
    """
    worker_active_tasks = tracking_data.get("worker_active_tasks", {})
    cores_per_worker = tracking_data.get("cores_per_worker")

    if not worker_active_tasks or cores_per_worker is None:
        raise ValueError("Active tasks or cores_per_worker data not available")

    # Collect all unique timestamps
    all_timestamps = set()
    for worker_id in worker_active_tasks.keys():
        for timestamp, _ in worker_active_tasks[worker_id]:
            all_timestamps.add(timestamp)

    sorted_timestamps = sorted(all_timestamps)

    # Calculate CPU utilization % at each timestamp
    cpu_util_pct = []
    cpu_util_min = []
    cpu_util_max = []

    for timestamp in sorted_timestamps:
        worker_utils = []

        for worker_id in worker_active_tasks.keys():
            # Find active tasks at this timestamp
            tasks_data = worker_active_tasks[worker_id]

            active_tasks = None
            for t, tasks in tasks_data:
                if t == timestamp:
                    active_tasks = tasks
                    break

            if active_tasks is not None:
                # CPU utilization = active_tasks / cores_per_worker
                util_pct = (active_tasks / cores_per_worker) * 100
                worker_utils.append(min(util_pct, 100))  # Cap at 100%

        if worker_utils:
            cpu_util_pct.append(np.mean(worker_utils))
            cpu_util_min.append(np.min(worker_utils))
            cpu_util_max.append(np.max(worker_utils))
        else:
            cpu_util_pct.append(0)
            cpu_util_min.append(0)
            cpu_util_max.append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot mean with confidence band
    ax.plot(sorted_timestamps, cpu_util_pct, linewidth=2, label='Mean', color='C1')
    ax.fill_between(sorted_timestamps, cpu_util_min, cpu_util_max,
                     alpha=0.3, label='Min-Max Range', color='C1')

    ax.set_xlabel("Time")
    ax.set_ylabel("CPU Utilization (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_scaling_efficiency(
    tracking_data: Dict,
    metrics: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (6, 5),
    title: str = "Scaling Efficiency",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot throughput vs number of workers to show scaling efficiency.

    Parameters
    ----------
    tracking_data : dict
        Data from load_worker_timeline()
    metrics : dict
        Metrics dict from collect_processing_metrics()
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Examples
    --------
    >>> fig, ax = plot_scaling_efficiency(tracking_data, metrics)
    >>> plt.show()
    """
    worker_counts = tracking_data["worker_counts"]
    throughput_gbps = metrics.get("overall_rate_gbps", 0)

    if not worker_counts:
        raise ValueError("No worker count data available")

    # Get average and peak worker counts
    counts = list(worker_counts.values())
    avg_workers = np.mean(counts)
    peak_workers = max(counts)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot single point for this run
    ax.scatter([avg_workers], [throughput_gbps], s=100, color='C0',
               label=f'This run (avg workers)', zorder=3)
    ax.scatter([peak_workers], [throughput_gbps], s=100, color='C1', marker='s',
               label=f'This run (peak workers)', zorder=3)

    # Plot ideal linear scaling line from origin
    max_x = peak_workers * 1.2
    ideal_y = throughput_gbps * (max_x / avg_workers)
    ax.plot([0, max_x], [0, ideal_y], '--', color='gray', alpha=0.5,
            label='Ideal linear scaling')

    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Throughput (Gbps)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, ideal_y * 1.1)

    # Add efficiency annotation
    efficiency = (throughput_gbps / avg_workers) / (throughput_gbps / avg_workers)  # Normalize to 1.0
    ax.text(0.05, 0.95, f'Avg Workers: {avg_workers:.1f}\nThroughput: {throughput_gbps:.2f} Gbps',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_summary_dashboard(
    tracking_data: Dict,
    metrics: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Create a summary dashboard with 4 plots.

    Layout:
    ┌─────────────────────┬─────────────────────┐
    │ Worker Count        │ Memory Utilization  │
    ├─────────────────────┼─────────────────────┤
    │ CPU Utilization     │ Scaling Efficiency  │
    └─────────────────────┴─────────────────────┘

    Parameters
    ----------
    tracking_data : dict
        Data from load_worker_timeline()
    metrics : dict
        Metrics dict from collect_processing_metrics()
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib Figure

    Examples
    --------
    >>> fig = plot_summary_dashboard(tracking_data, metrics)
    >>> plt.show()
    """
    fig = plt.figure(figsize=figsize)

    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Worker Count (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    worker_counts = tracking_data["worker_counts"]
    sorted_items = sorted(worker_counts.items())
    timestamps = [t for t, _ in sorted_items]
    counts = [c for _, c in sorted_items]
    ax1.plot(timestamps, counts, marker='o', linestyle='-', linewidth=2, markersize=4)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Number of Workers")
    ax1.set_title("Worker Count Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Memory Utilization (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        worker_memory = tracking_data.get("worker_memory", {})
        worker_memory_limit = tracking_data.get("worker_memory_limit", {})

        if worker_memory and worker_memory_limit:
            all_timestamps = set()
            for worker_id in worker_memory.keys():
                for timestamp, _ in worker_memory[worker_id]:
                    all_timestamps.add(timestamp)
            sorted_timestamps = sorted(all_timestamps)

            utilization_pct = []
            for timestamp in sorted_timestamps:
                worker_utils = []
                for worker_id in worker_memory.keys():
                    mem_value = None
                    for t, m in worker_memory[worker_id]:
                        if t == timestamp:
                            mem_value = m
                            break
                    limit_value = None
                    for t, l in worker_memory_limit.get(worker_id, []):
                        if t == timestamp:
                            limit_value = l
                            break
                    if mem_value is not None and limit_value is not None and limit_value > 0:
                        worker_utils.append((mem_value / limit_value) * 100)
                utilization_pct.append(np.mean(worker_utils) if worker_utils else 0)

            ax2.plot(sorted_timestamps, utilization_pct, linewidth=2, color='C0')
            ax2.set_ylim(0, 100)
        else:
            ax2.text(0.5, 0.5, 'No memory limit data', ha='center', va='center',
                    transform=ax2.transAxes)
    except Exception as e:
        ax2.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                transform=ax2.transAxes)

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Memory Utilization (%)")
    ax2.set_title("Memory Utilization Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. CPU Utilization (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    try:
        worker_active_tasks = tracking_data.get("worker_active_tasks", {})
        cores_per_worker = tracking_data.get("cores_per_worker")

        if worker_active_tasks and cores_per_worker:
            all_timestamps = set()
            for worker_id in worker_active_tasks.keys():
                for timestamp, _ in worker_active_tasks[worker_id]:
                    all_timestamps.add(timestamp)
            sorted_timestamps = sorted(all_timestamps)

            cpu_util_pct = []
            for timestamp in sorted_timestamps:
                worker_utils = []
                for worker_id in worker_active_tasks.keys():
                    active_tasks = None
                    for t, tasks in worker_active_tasks[worker_id]:
                        if t == timestamp:
                            active_tasks = tasks
                            break
                    if active_tasks is not None:
                        util_pct = (active_tasks / cores_per_worker) * 100
                        worker_utils.append(min(util_pct, 100))
                cpu_util_pct.append(np.mean(worker_utils) if worker_utils else 0)

            ax3.plot(sorted_timestamps, cpu_util_pct, linewidth=2, color='C1')
            ax3.set_ylim(0, 100)
        else:
            ax3.text(0.5, 0.5, 'No CPU utilization data', ha='center', va='center',
                    transform=ax3.transAxes)
    except Exception as e:
        ax3.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                transform=ax3.transAxes)

    ax3.set_xlabel("Time")
    ax3.set_ylabel("CPU Utilization (%)")
    ax3.set_title("CPU Utilization Over Time")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Scaling Efficiency (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    throughput_gbps = metrics.get("overall_rate_gbps", 0)
    counts = list(worker_counts.values())
    avg_workers = np.mean(counts)
    peak_workers = max(counts)

    ax4.scatter([avg_workers], [throughput_gbps], s=100, color='C0',
               label='Avg workers', zorder=3)
    ax4.scatter([peak_workers], [throughput_gbps], s=100, color='C1', marker='s',
               label='Peak workers', zorder=3)

    # Ideal scaling line
    max_x = peak_workers * 1.2
    ideal_y = throughput_gbps * (max_x / avg_workers) if avg_workers > 0 else 0
    ax4.plot([0, max_x], [0, ideal_y], '--', color='gray', alpha=0.5,
            label='Ideal linear scaling')

    ax4.set_xlabel("Number of Workers")
    ax4.set_ylabel("Throughput (Gbps)")
    ax4.set_title("Scaling Efficiency")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(0, max_x)
    ax4.set_ylim(0, ideal_y * 1.1 if ideal_y > 0 else 1)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
