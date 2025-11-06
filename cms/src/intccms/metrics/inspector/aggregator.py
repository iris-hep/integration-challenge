"""Aggregation and statistics for file inspection results."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from intccms.metrics.inspector.rucio import build_size_lookup  # type: ignore


def _branch_count(record: Dict) -> int:
    """Return the number of branches described in a result record."""
    branches = record.get("branches", {})
    if isinstance(branches, dict):
        return len(branches)
    if isinstance(branches, (list, tuple)):
        return len(branches)
    return 0


def _lfn_from_path(filepath: str) -> str:
    """Normalize a filepath to the underlying logical file name."""
    if filepath.startswith("root://"):
        stripped = filepath[len("root://") :]
        if "/" not in stripped:
            return filepath
        _, rest = stripped.split("/", 1)
        return "/" + rest.lstrip("/")
    return filepath


def aggregate_statistics(
    results: List[Dict],
    size_summary: Optional[Dict] = None,
) -> Dict:
    """Compute aggregate statistics from file inspection results.

    Parameters
    ----------
    results : List[dict]
        List of file metadata dicts from :func:`inspect_file`.
    size_summary : dict, optional
        Optional size information produced by :func:`inspector.rucio.fetch_file_sizes`.
        When supplied, byte-related metrics are derived from this summary. Without it,
        byte-centric numbers default to zero (useful when analysing remote files
        without local access).

    Returns
    -------
    stats : dict
        Dictionary containing aggregate statistics

    Examples
    --------
    >>> stats = aggregate_statistics(results)
    >>> print(f"Total events: {stats['total_events']:,}")
    >>>
    >>> size_summary = fetch_file_sizes(dataset_manager, processes=["signal"])
    >>> stats_with_sizes = aggregate_statistics(results, size_summary=size_summary)
    """
    event_counts = [r["num_events"] for r in results]
    total_events = sum(event_counts)

    # Unique branches across all files
    all_branches = set()
    for record in results:
        all_branches.update(record["branches"].keys())

    if event_counts:
        hist, bin_edges = np.histogram(event_counts, bins=10)
    else:
        hist, bin_edges = np.array([]), np.array([])

    size_lookup = build_size_lookup(size_summary) if size_summary else {}
    size_values = [entry["bytes"] for entry in size_summary.get("files", [])] if size_summary else []
    total_size_bytes = size_summary.get("total_bytes", 0) if size_summary else 0
    total_files_with_sizes = size_summary.get("total_files", 0) if size_summary else 0

    bytes_sum = 0
    events_with_size = 0
    event_branch_pairs = 0

    if size_lookup:
        for record in results:
            filepath = record["filepath"]
            bytes_value = size_lookup.get(filepath)
            if bytes_value is None:
                bytes_value = size_lookup.get(_lfn_from_path(filepath))

            if bytes_value:
                bytes_sum += bytes_value
                if record["num_events"] > 0:
                    events_with_size += record["num_events"]
                    branch_count = _branch_count(record)
                    if branch_count > 0:
                        event_branch_pairs += record["num_events"] * branch_count

    stats = {
        "total_files": len(results),
        "total_events": total_events,
        "total_size_bytes": total_size_bytes,
        "avg_events_per_file": np.mean(event_counts) if event_counts else 0,
        "std_events_per_file": np.std(event_counts) if event_counts else 0,
        "median_events_per_file": np.median(event_counts) if event_counts else 0,
        "p90_events_per_file": np.percentile(event_counts, 90) if event_counts else 0,
        "avg_file_size_bytes": (
            total_size_bytes / total_files_with_sizes if total_files_with_sizes else 0
        ),
        "median_file_size_bytes": np.median(size_values) if size_values else 0,
        "p90_file_size_bytes": np.percentile(size_values, 90) if size_values else 0,
        "total_branches": len(all_branches),
        "avg_bytes_per_event": (
            bytes_sum / events_with_size if events_with_size else 0
        ),
        "avg_bytes_per_event_per_branch": (
            bytes_sum / event_branch_pairs if event_branch_pairs else 0
        ),
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


def compute_dataset_statistics(
    grouped: Dict[str, List[Dict]],
    size_summary: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Compute per-dataset statistics.

    Parameters
    ----------
    grouped : Dict[str, List[dict]]
        Grouped results from group_by_dataset()
    size_summary : dict, optional
        Optional size summary produced by :func:`inspector.rucio.fetch_file_sizes`.

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
    >>>
    >>> size_summary = fetch_file_sizes(dataset_manager, processes=["signal"])
    >>> stats = compute_dataset_statistics(grouped, size_summary=size_summary)
    """
    dataset_stats: Dict[str, Dict] = {}
    size_lookup = build_size_lookup(size_summary) if size_summary else {}
    dataset_summary = size_summary.get("datasets", {}) if size_summary else {}

    for dataset_name, files in grouped.items():
        event_counts = [f["num_events"] for f in files]
        bytes_values = []
        for record in files:
            filepath = record["filepath"]
            bytes_value = size_lookup.get(filepath)
            if bytes_value is None:
                bytes_value = size_lookup.get(_lfn_from_path(filepath))
            if bytes_value:
                bytes_values.append(bytes_value)

        dataset_info = dataset_summary.get(dataset_name, {})
        total_size_bytes = dataset_info.get("total_bytes")
        if total_size_bytes is None:
            total_size_bytes = sum(bytes_values)

        num_files_with_sizes = dataset_info.get("num_files")
        if num_files_with_sizes is None:
            num_files_with_sizes = len(bytes_values)

        events_with_size = sum(
            record["num_events"]
            for record in files
            if size_lookup.get(record["filepath"])
            or size_lookup.get(_lfn_from_path(record["filepath"]))
        )
        event_branch_pairs = sum(
            record["num_events"] * _branch_count(record)
            for record in files
            if (
                size_lookup.get(record["filepath"])
                or size_lookup.get(_lfn_from_path(record["filepath"]))
            )
            and record["num_events"] > 0
            and _branch_count(record) > 0
        )

        dataset_stats[dataset_name] = {
            "num_files": len(files),
            "total_events": sum(event_counts),
            "total_size_bytes": total_size_bytes,
            "avg_events_per_file": np.mean(event_counts) if event_counts else 0,
            "avg_file_size_bytes": (
                total_size_bytes / num_files_with_sizes if num_files_with_sizes else 0
            ),
            "median_file_size_bytes": np.median(bytes_values) if bytes_values else 0,
            "bytes_per_event": (
                total_size_bytes / events_with_size if events_with_size else 0
            ),
            "bytes_per_event_per_branch": (
                total_size_bytes / event_branch_pairs if event_branch_pairs else 0
            ),
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
