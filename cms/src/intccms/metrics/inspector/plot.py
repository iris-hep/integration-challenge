"""Visualization tools for input file inspection results."""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


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

    ax.hist(event_counts, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Events per File", fontsize=12)
    ax.set_ylabel("Number of Files", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

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
    ax1.barh(datasets, num_files, color='steelblue', edgecolor='black')
    ax1.set_xlabel("Number of Files", fontsize=12)
    ax1.set_title("Files per Dataset", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Add values on bars
    for i, v in enumerate(num_files):
        ax1.text(v, i, f' {v:,}', va='center', fontsize=9)

    # Events per dataset
    ax2.barh(datasets, total_events, color='coral', edgecolor='black')
    ax2.set_xlabel("Total Events", fontsize=12)
    ax2.set_title("Events per Dataset", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add values on bars
    for i, v in enumerate(total_events):
        ax2.text(v, i, f' {v:,}', va='center', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, (ax1, ax2)


def plot_top_branches(
    top_branches: List[tuple],
    title: str = "Top Branches by Size",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
):
    """Plot top N largest branches.

    Parameters
    ----------
    top_branches : List[tuple]
        List of (branch_name, size_bytes, compression_ratio) from get_top_branches()
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> top = get_top_branches(results, top_n=20)
    >>> plot_top_branches(top)
    """
    branch_names = [b[0] for b in top_branches]
    sizes_mb = [b[1] / 1024 / 1024 for b in top_branches]  # Convert to MB
    compression_ratios = [b[2] for b in top_branches]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Branch sizes
    ax1.barh(range(len(branch_names)), sizes_mb, color='mediumseagreen', edgecolor='black')
    ax1.set_yticks(range(len(branch_names)))
    ax1.set_yticklabels(branch_names, fontsize=9)
    ax1.set_xlabel("Size (MB)", fontsize=12)
    ax1.set_title("Branch Size", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Add values on bars
    for i, v in enumerate(sizes_mb):
        ax1.text(v, i, f' {v:.1f}', va='center', fontsize=8)

    # Compression ratios
    colors = ['red' if r < 1.5 else 'orange' if r < 2.0 else 'green' for r in compression_ratios]
    ax2.barh(range(len(branch_names)), compression_ratios, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_yticks(range(len(branch_names)))
    ax2.set_yticklabels(branch_names, fontsize=9)
    ax2.set_xlabel("Compression Ratio", fontsize=12)
    ax2.set_title("Compression Efficiency", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=2.0, color='green', linestyle='--', alpha=0.5, label='Good (>2x)')
    ax2.axvline(x=1.5, color='orange', linestyle='--', alpha=0.5, label='OK (>1.5x)')
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()

    # Add values on bars
    for i, v in enumerate(compression_ratios):
        ax2.text(v, i, f' {v:.2f}x', va='center', fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
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

    ax.hist(file_sizes_gb, bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel("File Size (GB)", fontsize=12)
    ax.set_ylabel("Number of Files", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

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
    top_branches: List[tuple],
    save_path: Optional[str] = None,
):
    """Create a comprehensive summary dashboard with all key plots.

    Parameters
    ----------
    results : List[dict]
        All file inspection results
    dataset_stats : Dict[str, dict]
        Per-dataset statistics
    top_branches : List[tuple]
        Top branches by size
    save_path : str, optional
        Path to save figure

    Examples
    --------
    >>> # After running full inspection
    >>> plot_summary_dashboard(results, dataset_stats, top_branches,
    ...                        save_path="input_summary.png")
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Event distribution
    ax1 = fig.add_subplot(gs[0, 0])
    event_counts = [r["num_events"] for r in results]
    ax1.hist(event_counts, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Events per File")
    ax1.set_ylabel("Number of Files")
    ax1.set_title("Event Distribution", fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Dataset comparison - events
    ax2 = fig.add_subplot(gs[0, 1])
    datasets = list(dataset_stats.keys())
    total_events = [stats["total_events"] for stats in dataset_stats.values()]
    ax2.barh(datasets, total_events, color='coral', edgecolor='black')
    ax2.set_xlabel("Total Events")
    ax2.set_title("Events per Dataset", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Dataset comparison - files
    ax3 = fig.add_subplot(gs[1, 0])
    num_files = [stats["num_files"] for stats in dataset_stats.values()]
    ax3.barh(datasets, num_files, color='steelblue', edgecolor='black')
    ax3.set_xlabel("Number of Files")
    ax3.set_title("Files per Dataset", fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Top branches
    ax4 = fig.add_subplot(gs[1, 1])
    top_10 = top_branches[:10]
    branch_names = [b[0] for b in top_10]
    sizes_mb = [b[1] / 1024 / 1024 for b in top_10]
    ax4.barh(range(len(branch_names)), sizes_mb, color='mediumseagreen', edgecolor='black')
    ax4.set_yticks(range(len(branch_names)))
    ax4.set_yticklabels(branch_names, fontsize=9)
    ax4.set_xlabel("Size (MB)")
    ax4.set_title("Top 10 Branches by Size", fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()

    # 5. Compression ratios
    ax5 = fig.add_subplot(gs[2, :])
    compression_ratios = [b[2] for b in top_10]
    colors = ['red' if r < 1.5 else 'orange' if r < 2.0 else 'green' for r in compression_ratios]
    ax5.barh(range(len(branch_names)), compression_ratios, color=colors, edgecolor='black', alpha=0.7)
    ax5.set_yticks(range(len(branch_names)))
    ax5.set_yticklabels(branch_names, fontsize=9)
    ax5.set_xlabel("Compression Ratio")
    ax5.set_title("Compression Efficiency (Top 10 Branches)", fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.axvline(x=2.0, color='green', linestyle='--', alpha=0.5, label='Good (>2x)')
    ax5.axvline(x=1.5, color='orange', linestyle='--', alpha=0.5, label='OK (>1.5x)')
    ax5.legend()
    ax5.invert_yaxis()

    fig.suptitle("Input Data Characterization Summary", fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig
