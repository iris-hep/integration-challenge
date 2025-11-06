"""Visualization tools for input file inspection results."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from intccms.metrics.inspector.aggregator import compute_branch_statistics, group_by_dataset


def _apply_modern_style(ax):
    """Apply modern plotting style to axes.

    Applies consistent styling:
    - Internal tick marks on all sides
    - Minor ticks enabled
    - Light grid
    - Removes top/right spines for cleaner look

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to style
    """
    # Internal tick marks on all sides
    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
        labelsize=10,
    )

    # Enable minor ticks
    ax.minorticks_on()

    # Light grid
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _apply_scientific_notation(ax, axis="x", threshold=1e4):
    """Apply scientific notation with LaTeX when tick magnitudes are large."""
    axis_obj = ax.xaxis if axis == "x" else ax.yaxis
    ticks = axis_obj.get_ticklocs()
    finite_ticks = [abs(t) for t in ticks if np.isfinite(t)]
    if not finite_ticks:
        return
    if max(finite_ticks) < threshold:
        return

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    axis_obj.set_major_formatter(formatter)
    axis_obj.get_offset_text().set_fontsize(10)


def _label_with_units(ax, axis: str, base_label: str, unit: Optional[str] = None):
    """Append unit text to an axis label if provided."""
    label = base_label.strip()
    if unit:
        label = f"{label} ({unit})"

    if axis == "x":
        ax.set_xlabel(label, fontsize=12)
    else:
        ax.set_ylabel(label, fontsize=12)


def plot_event_distribution(
    results: List[Dict],
    title: str = "Event Distribution Across Files",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
):
    """Plot histogram of events per file.

    Parameters
    ----------
    results : List[dict]
        File inspection results
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure (if None, just displays)

    Examples
    --------
    >>> plot_event_distribution(results)
    >>> plot_event_distribution(results, save_path="events_distribution.png")
    """
    event_counts = [r["num_events"] for r in results]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(event_counts, bins=30, edgecolor="none", alpha=0.6, color="steelblue")
    _label_with_units(ax, "x", "Events per File", "events")
    _label_with_units(ax, "y", "Number of Files")
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Apply modern style
    _apply_modern_style(ax)
    _apply_scientific_notation(ax, axis="x")

    # Add stats text
    stats_text = f"Files: {len(event_counts)}\n"
    stats_text += f"Mean: {np.mean(event_counts):,.0f}\n"
    stats_text += f"Median: {np.median(event_counts):,.0f}\n"
    stats_text += f"Total: {sum(event_counts):,.0f}"

    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_dataset_comparison(
    dataset_stats: Dict[str, Dict],
    title: str = "Dataset Comparison",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    """Plot comparison of datasets (files and events).

    Parameters
    ----------
    dataset_stats : Dict[str, dict]
        Per-dataset statistics from compute_dataset_statistics()
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> plot_dataset_comparison(dataset_stats)
    """
    datasets = list(dataset_stats.keys())
    num_files = [stats["num_files"] for stats in dataset_stats.values()]
    total_events = [stats["total_events"] for stats in dataset_stats.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Files per dataset
    ax1.barh(datasets, num_files, color="steelblue", edgecolor="none", alpha=0.6)
    _label_with_units(ax1, "x", "Number of Files")
    ax1.set_title("Files per Dataset", fontsize=13, fontweight='bold')

    # Add values on bars
    for i, v in enumerate(num_files):
        ax1.text(v, i, f" {v:,}", va='center', fontsize=9)

    # Apply modern style
    _apply_modern_style(ax1)

    # Events per dataset
    ax2.barh(datasets, total_events, color="coral", edgecolor="none", alpha=0.6)
    _label_with_units(ax2, "x", "Total Events", "events")
    ax2.set_title("Events per Dataset", fontsize=13, fontweight='bold')

    # Add values on bars
    for i, v in enumerate(total_events):
        ax2.text(v, i, f" {v:,}", va='center', fontsize=9)

    # Apply modern style
    _apply_modern_style(ax2)
    _apply_scientific_notation(ax2, axis="x")

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, (ax1, ax2)


def plot_events_per_file_by_dataset(
    dataset_stats: Dict[str, Dict],
    title: str = "Events per File by Dataset",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
):
    """Plot events/file ratio comparison across datasets.

    Parameters
    ----------
    dataset_stats : Dict[str, dict]
        Per-dataset statistics from compute_dataset_statistics()
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> plot_events_per_file_by_dataset(dataset_stats)
    """
    datasets = list(dataset_stats.keys())
    events_per_file = [stats["avg_events_per_file"] for stats in dataset_stats.values()]

    fig, ax = plt.subplots(figsize=figsize)

    # Horizontal bar chart showing events/file ratio
    ax.barh(datasets, events_per_file, color="mediumseagreen", edgecolor="none", alpha=0.6)

    _label_with_units(ax, "x", "Average Events per File", "events")
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add values on bars
    for i, v in enumerate(events_per_file):
        ax.text(v, i, f" {v:,.0f}", va='center', fontsize=9)

    # Apply modern style
    _apply_modern_style(ax)
    _apply_scientific_notation(ax, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_branch_size_distribution(
    results: List[Dict],
    title: str = "Branch Size Distribution",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
):
    """Plot box plot of branch sizes across all branches.

    Parameters
    ----------
    results : List[dict]
        File inspection results
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> plot_branch_size_distribution(results)
    """
    # Compute branch statistics
    stats = compute_branch_statistics(results)
    sizes_mb = [s / 1024 / 1024 for s in stats["branch_sizes"]]  # Convert to MB

    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot with 5th-95th percentile whiskers
    bp = ax.boxplot(
        sizes_mb,
        orientation='vertical',
        patch_artist=True,
        widths=0.5,
        whis=[5, 95],
    )
    bp['boxes'][0].set_facecolor('mediumseagreen')
    bp['boxes'][0].set_alpha(0.6)

    _label_with_units(ax, "y", "Branch Size", "MB")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])

    # Apply modern style
    _apply_modern_style(ax)

    # Add stats text
    stats_text = f"Total branches: {stats['num_branches']}\n"
    stats_text += f"Median: {np.median(sizes_mb):.2f} MB\n"
    stats_text += f"Mean: {np.mean(sizes_mb):.2f} MB"

    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_branch_compression_distribution(
    results: List[Dict],
    title: str = "Branch Compression Ratio Distribution",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
):
    """Plot box plot of compression ratios across all branches.

    Parameters
    ----------
    results : List[dict]
        File inspection results
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> plot_branch_compression_distribution(results)
    """
    # Compute branch statistics
    stats = compute_branch_statistics(results)
    compression_ratios = stats["compression_ratios"]

    if not compression_ratios:
        print("No compression data available")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot with 5th-95th percentile whiskers
    bp = ax.boxplot(
        compression_ratios,
        orientation='vertical',
        patch_artist=True,
        widths=0.5,
        whis=[5, 95],
    )
    bp['boxes'][0].set_facecolor('coral')
    bp['boxes'][0].set_alpha(0.6)

    _label_with_units(ax, "y", "Compression Ratio")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])

    # Apply modern style
    _apply_modern_style(ax)

    # Add reference lines
    ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Good (>2x)')
    ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='OK (>1.5x)')
    ax.legend(fontsize=9, loc='upper right')

    # Add stats text
    stats_text = f"Median: {np.median(compression_ratios):.2f}x\n"
    stats_text += f"Mean: {np.mean(compression_ratios):.2f}x"

    ax.text(0.02, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_branch_distributions_by_dataset(
    results: List[Dict],
    dataset_map: Dict[str, str],
    title: str = "Branch Distributions by Dataset",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    """Plot side-by-side box plots of branch characteristics per dataset.

    Parameters
    ----------
    results : List[dict]
        File inspection results
    dataset_map : Dict[str, str]
        Mapping from filepath to dataset name
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> plot_branch_distributions_by_dataset(results, dataset_map)
    """
    # Group results by dataset
    grouped = group_by_dataset(results, dataset_map)

    # Compute stats for each dataset
    dataset_sizes = {}
    dataset_compressions = {}

    for dataset_name, dataset_results in grouped.items():
        stats = compute_branch_statistics(dataset_results)
        dataset_sizes[dataset_name] = [s / 1024 / 1024 for s in stats["branch_sizes"]]  # MB
        dataset_compressions[dataset_name] = stats["compression_ratios"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Branch sizes per dataset
    dataset_names = list(dataset_sizes.keys())
    size_data = [dataset_sizes[name] for name in dataset_names]

    bp1 = ax1.boxplot(size_data, tick_labels=dataset_names, patch_artist=True, whis=[5, 95])
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)

    _label_with_units(ax1, "y", "Branch Size", "MB")
    ax1.set_title("Branch Sizes", fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Apply modern style
    _apply_modern_style(ax1)

    # Compression ratios per dataset
    compression_data = [dataset_compressions[name] for name in dataset_names if dataset_compressions[name]]
    compression_labels = [name for name in dataset_names if dataset_compressions[name]]

    if compression_data:
        bp2 = ax2.boxplot(compression_data, tick_labels=compression_labels, patch_artist=True, whis=[5, 95])
        for patch in bp2['boxes']:
            patch.set_facecolor('coral')
            patch.set_alpha(0.6)

        ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    _label_with_units(ax2, "y", "Compression Ratio")
    ax2.set_title("Compression Ratios", fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # Apply modern style
    _apply_modern_style(ax2)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, (ax1, ax2)


def plot_file_size_distribution(
    results: List[Dict],
    title: str = "File Size Distribution",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
):
    """Plot histogram of file sizes.

    Parameters
    ----------
    results : List[dict]
        File inspection results
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> plot_file_size_distribution(results)
    """
    # Filter out files with zero size (remote files)
    file_sizes_gb = [
        r["file_size_bytes"] / 1024**3
        for r in results
        if r["file_size_bytes"] > 0
    ]

    if not file_sizes_gb:
        print("No file size data available (all files are remote)")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(file_sizes_gb, bins=30, edgecolor='none', alpha=0.6, color='purple')
    _label_with_units(ax, "x", "File Size", "GB")
    _label_with_units(ax, "y", "Number of Files")
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Apply modern style
    _apply_modern_style(ax)
    _apply_scientific_notation(ax, axis="x")

    # Add stats text
    stats_text = f"Files: {len(file_sizes_gb)}\n"
    stats_text += f"Mean: {np.mean(file_sizes_gb):.2f} GB\n"
    stats_text += f"Median: {np.median(file_sizes_gb):.2f} GB\n"
    stats_text += f"Total: {sum(file_sizes_gb):.1f} GB"

    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_summary_dashboard(
    results: List[Dict],
    dataset_stats: Dict[str, Dict],
    dataset_map: Dict[str, str],
    save_path: Optional[str] = None,
):
    """Create a comprehensive summary dashboard with all key plots.

    Parameters
    ----------
    results : List[dict]
        All file inspection results
    dataset_stats : Dict[str, dict]
        Per-dataset statistics
    dataset_map : Dict[str, str]
        Mapping from filepath to dataset name
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> # After running full inspection
    >>> plot_summary_dashboard(results, dataset_stats, dataset_map,
    ...                        save_path="input_summary.png")
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # 1. Event distribution
    ax1 = fig.add_subplot(gs[0, 0])
    event_counts = [r["num_events"] for r in results]
    ax1.hist(event_counts, bins=30, edgecolor='none', alpha=0.6, color='steelblue')
    ax1.set_xlabel("Events per File", fontsize=10)
    ax1.set_ylabel("Number of Files", fontsize=10)
    ax1.set_title("Event Distribution", fontweight='bold', fontsize=11)
    _apply_modern_style(ax1)

    # 2. Dataset comparison - events
    ax2 = fig.add_subplot(gs[0, 1])
    datasets = list(dataset_stats.keys())
    total_events = [stats["total_events"] for stats in dataset_stats.values()]
    ax2.barh(datasets, total_events, color='coral', edgecolor='none', alpha=0.6)
    ax2.set_xlabel("Total Events", fontsize=10)
    ax2.set_title("Events per Dataset", fontweight='bold', fontsize=11)
    _apply_modern_style(ax2)

    # 3. Dataset comparison - files
    ax3 = fig.add_subplot(gs[1, 0])
    num_files = [stats["num_files"] for stats in dataset_stats.values()]
    ax3.barh(datasets, num_files, color='steelblue', edgecolor='none', alpha=0.6)
    ax3.set_xlabel("Number of Files", fontsize=10)
    ax3.set_title("Files per Dataset", fontweight='bold', fontsize=11)
    _apply_modern_style(ax3)

    # 4. Branch size distribution (aggregated)
    ax4 = fig.add_subplot(gs[1, 1])
    stats = compute_branch_statistics(results)
    sizes_mb = [s / 1024 / 1024 for s in stats["branch_sizes"]]
    bp = ax4.boxplot(sizes_mb, orientation='vertical', patch_artist=True, widths=0.5, whis=[5, 95])
    bp['boxes'][0].set_facecolor('mediumseagreen')
    bp['boxes'][0].set_alpha(0.6)
    ax4.set_ylabel("Branch Size (MB)", fontsize=10)
    ax4.set_title("Branch Size Distribution", fontweight='bold', fontsize=11)
    ax4.set_xticks([])
    _apply_modern_style(ax4)

    # 5. Branch compression distribution (aggregated)
    ax5 = fig.add_subplot(gs[2, 0])
    compression_ratios = stats["compression_ratios"]
    if compression_ratios:
        bp2 = ax5.boxplot(compression_ratios, orientation='vertical', patch_artist=True, widths=0.5, whis=[5, 95])
        bp2['boxes'][0].set_facecolor('coral')
        bp2['boxes'][0].set_alpha(0.6)
        ax5.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax5.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax5.set_ylabel("Compression Ratio", fontsize=10)
    ax5.set_title("Compression Distribution", fontweight='bold', fontsize=11)
    ax5.set_xticks([])
    _apply_modern_style(ax5)

    # 6. Per-dataset branch size comparison
    ax6 = fig.add_subplot(gs[2, 1])
    grouped = group_by_dataset(results, dataset_map)
    dataset_sizes = {}
    for dataset_name, dataset_results in grouped.items():
        ds_stats = compute_branch_statistics(dataset_results)
        dataset_sizes[dataset_name] = [s / 1024 / 1024 for s in ds_stats["branch_sizes"]]

    dataset_names = list(dataset_sizes.keys())
    size_data = [dataset_sizes[name] for name in dataset_names]
    if size_data:
        bp3 = ax6.boxplot(size_data, tick_labels=dataset_names, patch_artist=True, whis=[5, 95])
        for patch in bp3['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)
        ax6.tick_params(axis='x', rotation=45)
    ax6.set_ylabel("Branch Size (MB)", fontsize=10)
    ax6.set_title("Branch Sizes by Dataset", fontweight='bold', fontsize=11)
    _apply_modern_style(ax6)

    fig.suptitle("Input Data Characterization Summary", fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig
