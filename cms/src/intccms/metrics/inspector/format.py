"""Rich formatting for inspection statistics display."""

from typing import Dict
from rich.table import Table
from rich.console import Console


def _format_bytes(value: float) -> str:
    """Format byte quantity with appropriate unit."""
    if value is None or value <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    magnitude = 0
    while value >= 1024 and magnitude < len(units) - 1:
        value /= 1024
        magnitude += 1
    return f"{value:.2f} {units[magnitude]}"


def format_overall_stats_table(stats: Dict) -> Table:
    """Format overall statistics as a rich table.

    Parameters
    ----------
    stats : dict
        Statistics from aggregate_statistics()

    Returns
    -------
    table : rich.table.Table
        Formatted rich table

    Examples
    --------
    >>> from rich.console import Console
    >>> console = Console(force_jupyter=False)
    >>> table = format_overall_stats_table(stats)
    >>> console.print(table)
    """
    table = Table(title="Overall Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")

    # File and event counts
    table.add_row("Total Files", f"{stats['total_files']:,}")
    table.add_row("Total Events", f"{stats['total_events']:,}")
    table.add_row("Total Size", f"{stats['total_size_bytes'] / 1024**3:.2f} GB")

    table.add_section()

    # Events per file
    table.add_row("Avg Events/File", f"{stats['avg_events_per_file']:,.0f}")
    table.add_row("Median Events/File", f"{stats['median_events_per_file']:,.0f}")
    table.add_row("P90 Events/File", f"{stats['p90_events_per_file']:,.0f}")

    table.add_section()

    # File sizes
    table.add_row("Avg File Size", f"{stats['avg_file_size_bytes'] / 1024**3:.2f} GB")
    table.add_row("Median File Size", f"{stats['median_file_size_bytes'] / 1024**3:.2f} GB")

    table.add_section()

    table.add_row("Avg Bytes/Event", _format_bytes(stats.get('avg_bytes_per_event')))
    table.add_row("Avg Bytes/Event/Branch", _format_bytes(stats.get('avg_bytes_per_event_per_branch')))

    table.add_section()

    # Branch info
    table.add_row("Total Branches", f"{stats['total_branches']}")

    return table


def format_branch_stats_table(branch_stats: Dict) -> Table:
    """Format branch statistics as a rich table.

    Parameters
    ----------
    branch_stats : dict
        Statistics from compute_branch_statistics()

    Returns
    -------
    table : rich.table.Table
        Formatted rich table

    Examples
    --------
    >>> from rich.console import Console
    >>> console = Console(force_jupyter=False)
    >>> table = format_branch_stats_table(branch_stats)
    >>> console.print(table)
    """
    import numpy as np

    table = Table(title="Branch Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="magenta", no_wrap=True)
    table.add_column("Value", justify="right", style="yellow")

    # Number of branches
    table.add_row("Number of Branches", f"{branch_stats['num_branches']:,}")

    table.add_section()

    # Branch sizes (convert to MB)
    sizes_mb = [s / 1024 / 1024 for s in branch_stats['branch_sizes']]
    table.add_row("Median Branch Size", f"{np.median(sizes_mb):.2f} MB")
    table.add_row("Mean Branch Size", f"{np.mean(sizes_mb):.2f} MB")
    table.add_row("Max Branch Size", f"{max(sizes_mb):.2f} MB")

    table.add_section()

    # Compression ratios
    if branch_stats['compression_ratios']:
        table.add_row("Median Compression", f"{np.median(branch_stats['compression_ratios']):.2f}x")
        table.add_row("Mean Compression", f"{np.mean(branch_stats['compression_ratios']):.2f}x")
    else:
        table.add_row("Compression Data", "Not Available")

    return table


def format_dataset_stats_table(dataset_stats: Dict[str, Dict]) -> Table:
    """Format per-dataset statistics as a rich table.

    Parameters
    ----------
    dataset_stats : Dict[str, dict]
        Statistics from compute_dataset_statistics()

    Returns
    -------
    table : rich.table.Table
        Formatted rich table

    Examples
    --------
    >>> from rich.console import Console
    >>> console = Console(force_jupyter=False)
    >>> table = format_dataset_stats_table(dataset_stats)
    >>> console.print(table)
    """
    table = Table(title="Per-Dataset Statistics", show_header=True, header_style="bold blue")
    table.add_column("Dataset", style="blue", no_wrap=True)
    table.add_column("Files", justify="right", style="cyan")
    table.add_column("Total Events", justify="right", style="green")
    table.add_column("Avg Events/File", justify="right", style="yellow")
    table.add_column("Total Size", justify="right", style="magenta")
    table.add_column("Avg File Size", justify="right", style="red")
    table.add_column("Bytes/Event", justify="right", style="yellow")
    table.add_column("Bytes/Event/Branch", justify="right", style="yellow")

    for dataset_name, stats in dataset_stats.items():
        table.add_row(
            dataset_name,
            f"{stats['num_files']:,}",
            f"{stats['total_events']:,}",
            f"{stats['avg_events_per_file']:,.0f}",
            f"{stats['total_size_bytes'] / 1024**3:.2f} GB",
            f"{stats['avg_file_size_bytes'] / 1024**3:.2f} GB",
            _format_bytes(stats.get('bytes_per_event')),
            _format_bytes(stats.get('bytes_per_event_per_branch')),
        )

    return table


def format_compression_stats_table(comp_stats: Dict) -> Table:
    """Format compression statistics as a rich table.

    Parameters
    ----------
    comp_stats : dict
        Statistics from compute_compression_stats()

    Returns
    -------
    table : rich.table.Table
        Formatted rich table

    Examples
    --------
    >>> from rich.console import Console
    >>> console = Console(force_jupyter=False)
    >>> table = format_compression_stats_table(comp_stats)
    >>> console.print(table)
    """
    table = Table(title="Compression Statistics", show_header=True, header_style="bold red")
    table.add_column("Metric", style="red", no_wrap=True)
    table.add_column("Value", justify="right", style="yellow")

    # Files with compression data
    table.add_row("Files with Compression", f"{comp_stats['files_with_compression']:,}")

    table.add_section()

    # Tree-level compression
    table.add_row("Avg Tree Compression", f"{comp_stats['avg_tree_compression_ratio']:.2f}x")
    table.add_row("Median Tree Compression", f"{comp_stats['median_tree_compression_ratio']:.2f}x")
    table.add_row("Overall Compression", f"{comp_stats['overall_compression_ratio']:.2f}x")

    table.add_section()

    # Sizes
    table.add_row("Total Compressed", f"{comp_stats['total_compressed_bytes'] / 1024**3:.2f} GB")
    table.add_row("Total Uncompressed", f"{comp_stats['total_uncompressed_bytes'] / 1024**3:.2f} GB")
    table.add_row(
        "Space Savings",
        f"{(comp_stats['total_uncompressed_bytes'] - comp_stats['total_compressed_bytes']) / 1024**3:.2f} GB"
    )

    return table
