"""Rich table formatting for processing metrics.

This module provides Rich table formatters for displaying processing metrics
in Jupyter notebooks and terminal output.
"""

from typing import Any, Dict

from rich.table import Table

from intccms.metrics.collector import format_bytes, format_time


def format_throughput_table(metrics: Dict[str, Any]) -> Table:
    """Format throughput metrics as Rich table.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from collect_processing_metrics()

    Returns
    -------
    table : rich.table.Table
        Formatted Rich table

    Examples
    --------
    >>> from rich.console import Console
    >>> console = Console()
    >>> table = format_throughput_table(metrics)
    >>> console.print(table)
    """
    table = Table(title="Throughput Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Data rate
    table.add_row(
        "Data Rate",
        f"{metrics.get('overall_rate_gbps', 0):.2f} Gbps ({metrics.get('overall_rate_mbps', 0):.1f} MB/s)"
    )

    # Compression
    ratio = metrics.get('compression_ratio', 0)
    table.add_row("Compression Ratio", f"{ratio:.2f}x")

    # Data volume
    compressed = metrics.get('total_bytes_compressed', 0)
    uncompressed = metrics.get('total_bytes_uncompressed', 0)
    if compressed or uncompressed:
        table.add_row(
            "Total Data Read",
            f"{format_bytes(compressed)} compressed, {format_bytes(uncompressed)} uncompressed"
        )

    return table


def format_event_processing_table(metrics: Dict[str, Any]) -> Table:
    """Format event processing metrics as Rich table.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from collect_processing_metrics()

    Returns
    -------
    table : rich.table.Table
        Formatted Rich table

    Examples
    --------
    >>> from rich.console import Console
    >>> console = Console()
    >>> table = format_event_processing_table(metrics)
    >>> console.print(table)
    """
    table = Table(title="Event Processing Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Total events
    total_events = metrics.get('total_events', 0)
    table.add_row("Total Events", f"{total_events:,}")

    # Event rates
    wall_khz = metrics.get('event_rate_wall_khz', 0)
    table.add_row("Event Rate (Wall Clock)", f"{wall_khz:.1f} kHz")

    agg_khz = metrics.get('event_rate_agg_khz', 0)
    table.add_row("Event Rate (Aggregated)", f"{agg_khz:.1f} kHz")

    # Core-averaged rate (may be None if no worker data)
    core_hz = metrics.get('event_rate_core_hz')
    if core_hz is not None:
        table.add_row("Event Rate (Core-Averaged)", f"{core_hz:.1f} Hz/core")
    else:
        table.add_row("Event Rate (Core-Averaged)", "[dim]N/A (no worker data)[/dim]")

    # Efficiency ratio
    if wall_khz and agg_khz:
        efficiency_ratio = wall_khz / agg_khz
        table.add_row("Efficiency Ratio", f"{efficiency_ratio:.1%}")

    return table


def format_resources_table(metrics: Dict[str, Any]) -> Table:
    """Format resource utilization metrics as Rich table.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from collect_processing_metrics()

    Returns
    -------
    table : rich.table.Table
        Formatted Rich table

    Examples
    --------
    >>> from rich.console import Console
    >>> console = Console()
    >>> table = format_resources_table(metrics)
    >>> console.print(table)
    """
    table = Table(title="Resource Utilization", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Worker metrics
    avg_workers = metrics.get('avg_workers')
    if avg_workers is not None:
        table.add_row("Workers (Time-Averaged)", f"{avg_workers:.1f}")
    else:
        table.add_row("Workers (Time-Averaged)", "[dim]N/A (no worker tracking)[/dim]")

    peak_workers = metrics.get('peak_workers')
    if peak_workers is not None:
        table.add_row("Peak Workers", f"{peak_workers}")
    else:
        table.add_row("Peak Workers", "[dim]N/A (no worker tracking)[/dim]")

    # Core metrics
    total_cores = metrics.get('total_cores')
    if total_cores is not None:
        table.add_row("Total Cores", f"{total_cores:.0f}")
    else:
        table.add_row("Total Cores", "[dim]N/A (no worker tracking)[/dim]")

    # Efficiency
    core_efficiency = metrics.get('core_efficiency')
    if core_efficiency is not None:
        table.add_row("Core Efficiency", f"{core_efficiency:.1%}")
    else:
        table.add_row("Core Efficiency", "[dim]N/A (no worker tracking)[/dim]")

    # Speedup
    speedup = metrics.get('speedup_factor')
    if speedup is not None:
        table.add_row("Speedup Factor", f"{speedup:.1f}x")
    else:
        table.add_row("Speedup Factor", "[dim]N/A (no worker tracking)[/dim]")

    return table


def format_timing_table(metrics: Dict[str, Any]) -> Table:
    """Format timing metrics as Rich table.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from collect_processing_metrics()

    Returns
    -------
    table : rich.table.Table
        Formatted Rich table

    Examples
    --------
    >>> from rich.console import Console
    >>> console = Console()
    >>> table = format_timing_table(metrics)
    >>> console.print(table)
    """
    table = Table(title="Timing Breakdown", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Wall time
    wall_time = metrics.get('wall_time', 0)
    table.add_row("Wall Time", format_time(wall_time))

    # CPU time
    cpu_time = metrics.get('total_cpu_time', 0)
    table.add_row("Total CPU Time", format_time(cpu_time))

    # Chunk metrics
    num_chunks = metrics.get('num_chunks', 0)
    if num_chunks > 0:
        table.add_row("Number of Chunks", f"{num_chunks:,}")
        avg_cpu_per_chunk = metrics.get('avg_cpu_time_per_chunk', 0)
        table.add_row("Avg CPU Time/Chunk", format_time(avg_cpu_per_chunk))

    return table
