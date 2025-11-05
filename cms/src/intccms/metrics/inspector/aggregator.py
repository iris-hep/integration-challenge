"""Aggregation and statistics for file inspection results."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def aggregate_statistics(results: List[Dict]) -> Dict:
    """Compute aggregate statistics from file inspection results.

    Parameters
    ----------
    results : List[dict]
        List of file metadata dicts from inspect_file() or inspect_dataset_distributed()

    Returns
    -------
    stats : dict
        Dictionary containing aggregate statistics

    Examples
    --------
    >>> stats = aggregate_statistics(results)
    >>> print(f"Total events: {stats['total_events']:,}")
    """
    event_counts = [r["num_events"] for r in results]
    file_sizes = [r["file_size_bytes"] for r in results if r["file_size_bytes"] > 0]

    # Get all unique branches (assume all files have same branches)
    all_branches = set()
    for r in results:
        all_branches.update(r["branches"].keys())

    # Compute histogram
    if event_counts:
        hist, bin_edges = np.histogram(event_counts, bins=10)
    else:
        hist, bin_edges = np.array([]), np.array([])

    stats = {
        "total_files": len(results),
        "total_events": sum(event_counts),
        "total_size_bytes": sum(file_sizes),
        "avg_events_per_file": np.mean(event_counts) if event_counts else 0,
        "std_events_per_file": np.std(event_counts) if event_counts else 0,
        "median_events_per_file": np.median(event_counts) if event_counts else 0,
        "p90_events_per_file": np.percentile(event_counts, 90) if event_counts else 0,
        "avg_file_size_bytes": np.mean(file_sizes) if file_sizes else 0,
        "median_file_size_bytes": np.median(file_sizes) if file_sizes else 0,
        "p90_file_size_bytes": np.percentile(file_sizes, 90) if file_sizes else 0,
        "total_branches": len(all_branches),
        "event_histogram": {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist(),
        },
    }

    return stats


def get_top_branches(results: List[Dict], top_n: int = 20) -> List[Tuple[str, int, float]]:
    """Find the largest branches across all files.

    Parameters
    ----------
    results : List[dict]
        List of file metadata dicts
    top_n : int
        Number of top branches to return

    Returns
    -------
    top_branches : List[Tuple[str, int, float]]
        List of (branch_name, total_bytes, avg_compression_ratio) tuples

    Examples
    --------
    >>> top = get_top_branches(results, top_n=10)
    >>> for name, size, ratio in top[:5]:
    ...     print(f"{name}: {size/1024:.1f} KB (ratio: {ratio:.2f}x)")
    """
    # Aggregate branch sizes across all files
    branch_totals = defaultdict(int)
    branch_compression = defaultdict(list)

    for result in results:
        for branch_name, info in result["branches"].items():
            # info is tuple: (num_bytes, compressed_bytes, uncompressed_bytes)
            num_bytes, compressed_bytes, uncompressed_bytes = info
            branch_totals[branch_name] += num_bytes

            # Track compression ratios if available
            if compressed_bytes > 0 and uncompressed_bytes > 0:
                ratio = uncompressed_bytes / compressed_bytes
                branch_compression[branch_name].append(ratio)

    # Compute average compression ratio per branch
    branch_avg_compression = {}
    for branch_name, ratios in branch_compression.items():
        branch_avg_compression[branch_name] = np.mean(ratios) if ratios else 1.0

    # Sort by total size
    sorted_branches = sorted(branch_totals.items(), key=lambda x: x[1], reverse=True)

    # Return top N with compression info
    top_branches = [
        (name, size, branch_avg_compression.get(name, 1.0))
        for name, size in sorted_branches[:top_n]
    ]

    return top_branches


def group_by_dataset(
    results: List[Dict], dataset_map: Optional[Dict[str, str]] = None
) -> Dict[str, List[Dict]]:
    """Group file results by dataset name.

    Parameters
    ----------
    results : List[dict]
        List of file metadata dicts
    dataset_map : Dict[str, str], optional
        Mapping from filepath to dataset name
        If None, uses "dataset" key from results if available

    Returns
    -------
    grouped : Dict[str, List[dict]]
        Dictionary mapping dataset name to list of file metadata

    Examples
    --------
    >>> # With dataset info already in results
    >>> grouped = group_by_dataset(results)
    >>>
    >>> # Or provide explicit mapping
    >>> dataset_map = {filepath: "ttbar_semilep" for filepath in file_list}
    >>> grouped = group_by_dataset(results, dataset_map)
    """
    grouped = defaultdict(list)

    for result in results:
        # Try to get dataset name from multiple sources
        if dataset_map and result["filepath"] in dataset_map:
            dataset = dataset_map[result["filepath"]]
        elif "dataset" in result:
            dataset = result["dataset"]
        else:
            dataset = "unknown"

        grouped[dataset].append(result)

    return dict(grouped)


def compute_dataset_statistics(grouped: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Compute per-dataset statistics.

    Parameters
    ----------
    grouped : Dict[str, List[dict]]
        Grouped results from group_by_dataset()

    Returns
    -------
    dataset_stats : Dict[str, dict]
        Dictionary mapping dataset name to statistics dict

    Examples
    --------
    >>> grouped = group_by_dataset(results, dataset_map)
    >>> stats = compute_dataset_statistics(grouped)
    >>> for dataset, ds_stats in stats.items():
    ...     print(f"{dataset}: {ds_stats['total_events']:,} events")
    """
    dataset_stats = {}

    for dataset_name, files in grouped.items():
        event_counts = [f["num_events"] for f in files]
        file_sizes = [f["file_size_bytes"] for f in files if f["file_size_bytes"] > 0]

        dataset_stats[dataset_name] = {
            "num_files": len(files),
            "total_events": sum(event_counts),
            "total_size_bytes": sum(file_sizes) if file_sizes else 0,
            "avg_events_per_file": np.mean(event_counts) if event_counts else 0,
            "avg_file_size_bytes": np.mean(file_sizes) if file_sizes else 0,
            "median_file_size_bytes": np.median(file_sizes) if file_sizes else 0,
        }

    return dataset_stats


def compute_compression_stats(results: List[Dict]) -> Dict:
    """Compute compression statistics from inspection results.

    Parameters
    ----------
    results : List[dict]
        List of file metadata dicts

    Returns
    -------
    compression_stats : dict
        Dictionary containing compression statistics

    Examples
    --------
    >>> stats = compute_compression_stats(results)
    >>> print(f"Avg compression: {stats['avg_tree_compression_ratio']:.2f}x")
    """
    tree_ratios = []
    total_compressed = 0
    total_uncompressed = 0
    files_with_compression = 0

    for result in results:
        comp = result.get("tree_compressed_bytes", 0)
        uncomp = result.get("tree_uncompressed_bytes", 0)

        if comp > 0 and uncomp > 0:
            tree_ratios.append(uncomp / comp)
            total_compressed += comp
            total_uncompressed += uncomp
            files_with_compression += 1

    return {
        "avg_tree_compression_ratio": np.mean(tree_ratios) if tree_ratios else 0,
        "median_tree_compression_ratio": np.median(tree_ratios) if tree_ratios else 0,
        "files_with_compression": files_with_compression,
        "total_compressed_bytes": total_compressed,
        "total_uncompressed_bytes": total_uncompressed,
        "overall_compression_ratio": (
            total_uncompressed / total_compressed if total_compressed > 0 else 0
        ),
    }


def compute_branch_statistics(results: List[Dict]) -> Dict:
    """Compute branch-level statistics for distribution plots.

    Aggregates branch sizes across all files and collects compression ratios
    for all branches across all files.

    Parameters
    ----------
    results : List[dict]
        List of file metadata dicts from inspection

    Returns
    -------
    branch_stats : dict
        Dictionary containing:
        - branch_sizes: List of total branch sizes (summed across files)
        - compression_ratios: List of compression ratios (one per branch per file)
        - branch_names: List of unique branch names
        - num_branches: Number of unique branches

    Examples
    --------
    >>> stats = compute_branch_statistics(results)
    >>> # Use for box plot of branch sizes
    >>> plt.boxplot(stats['branch_sizes'])
    >>> # Use for box plot of compression ratios
    >>> plt.boxplot(stats['compression_ratios'])
    """
    # Aggregate branch sizes across all files
    branch_totals = defaultdict(int)
    compression_ratios = []
    all_branch_names = set()

    for result in results:
        for branch_name, info in result["branches"].items():
            all_branch_names.add(branch_name)

            # info is tuple: (num_bytes, compressed_bytes, uncompressed_bytes)
            num_bytes, compressed_bytes, uncompressed_bytes = info

            # Aggregate total size for this branch
            branch_totals[branch_name] += num_bytes

            # Collect compression ratio if available
            if compressed_bytes > 0 and uncompressed_bytes > 0:
                ratio = uncompressed_bytes / compressed_bytes
                compression_ratios.append(ratio)

    # Convert to lists for plotting
    branch_sizes = list(branch_totals.values())
    branch_names = list(all_branch_names)

    return {
        "branch_sizes": branch_sizes,
        "compression_ratios": compression_ratios,
        "branch_names": branch_names,
        "num_branches": len(branch_names),
    }
