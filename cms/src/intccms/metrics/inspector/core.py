"""Distributed file inspection for input data characterization.

This module provides functions to extract metadata from ROOT files in parallel
using Dask, enabling fast characterization of large datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask.bag as db
import uproot


def inspect_file(filepath: str) -> Dict:
    """Extract all metadata from a single ROOT file.

    Parameters
    ----------
    filepath : str
        Path to ROOT file (local or XRootD URL)

    Returns
    -------
    metadata : dict
        Dictionary containing:
        - filepath: Original file path
        - file_size_bytes: File size on disk
        - num_events: Number of events in tree
        - num_branches: Number of branches
        - total_branch_bytes: Sum of all branch sizes
        - tree_compressed_bytes: Tree-level compressed size
        - tree_uncompressed_bytes: Tree-level uncompressed size
        - branches: Dict mapping branch name to size info

    Examples
    --------
    >>> metadata = inspect_file("file.root")
    >>> print(f"Events: {metadata['num_events']:,}")
    >>> print(f"Size: {metadata['file_size_bytes'] / 1024**3:.2f} GB")
    """
    with uproot.open(filepath) as f:
        tree = f["Events"]

        # Get all branch info
        branches = {}
        for branch_name in tree.keys():
            branch = tree[branch_name]

            # Get sizes
            # - num_bytes: Always works (on-disk size including ROOT overhead)
            # - compressed/uncompressed: Work on production files, may be 0 on small test files
            num_bytes = branch.num_bytes if hasattr(branch, "num_bytes") else 0
            compressed = (
                branch.compressed_bytes if hasattr(branch, "compressed_bytes") else 0
            )
            uncompressed = (
                branch.uncompressed_bytes if hasattr(branch, "uncompressed_bytes") else 0
            )

            branches[branch_name] = {
                "num_bytes": num_bytes,
                "compressed_bytes": compressed,
                "uncompressed_bytes": uncompressed,
            }

        # Compute totals
        total_branch_bytes = sum(b["num_bytes"] for b in branches.values())

        # Get tree-level compression info if available
        tree_compressed = tree.compressed_bytes if hasattr(tree, "compressed_bytes") else 0
        tree_uncompressed = (
            tree.uncompressed_bytes if hasattr(tree, "uncompressed_bytes") else 0
        )

        # Get file size (handle both local and remote)
        if filepath.startswith("root://"):
            # For remote files, we can't easily get file size
            file_size_bytes = 0
        else:
            file_size_bytes = Path(filepath).stat().st_size

        return {
            "filepath": filepath,
            "file_size_bytes": file_size_bytes,
            "num_events": tree.num_entries,
            "num_branches": len(branches),
            "total_branch_bytes": total_branch_bytes,
            "tree_compressed_bytes": tree_compressed,
            "tree_uncompressed_bytes": tree_uncompressed,
            "branches": branches,
        }


def _inspect_file_with_limit(filepath: str, max_branches: Optional[int]) -> Dict:
    """Helper to inspect file with optional branch limit.

    Module-level function to avoid Dask serialization issues with closures.
    """
    if max_branches is None:
        return inspect_file(filepath)

    # Inspect with branch limit (faster for remote files)
    with uproot.open(filepath) as f:
        tree = f["Events"]

        # Only sample first N branches
        branch_names = list(tree.keys())[:max_branches]

        branches = {}
        for branch_name in branch_names:
            branch = tree[branch_name]
            num_bytes = branch.num_bytes if hasattr(branch, "num_bytes") else 0
            compressed = (
                branch.compressed_bytes
                if hasattr(branch, "compressed_bytes")
                else 0
            )
            uncompressed = (
                branch.uncompressed_bytes
                if hasattr(branch, "uncompressed_bytes")
                else 0
            )

            branches[branch_name] = {
                "num_bytes": num_bytes,
                "compressed_bytes": compressed,
                "uncompressed_bytes": uncompressed,
            }

        total_branch_bytes = sum(b["num_bytes"] for b in branches.values())
        tree_compressed = (
            tree.compressed_bytes if hasattr(tree, "compressed_bytes") else 0
        )
        tree_uncompressed = (
            tree.uncompressed_bytes if hasattr(tree, "uncompressed_bytes") else 0
        )

        if filepath.startswith("root://"):
            file_size_bytes = 0
        else:
            file_size_bytes = Path(filepath).stat().st_size

        return {
            "filepath": filepath,
            "file_size_bytes": file_size_bytes,
            "num_events": tree.num_entries,
            "num_branches": len(tree.keys()),  # Total branches
            "num_branches_sampled": len(branches),  # Sampled branches
            "total_branch_bytes": total_branch_bytes,
            "tree_compressed_bytes": tree_compressed,
            "tree_uncompressed_bytes": tree_uncompressed,
            "branches": branches,
        }


def _safe_inspect_wrapper(args: tuple) -> Union[Dict, tuple]:
    """Safe inspection wrapper that catches exceptions.

    Module-level function to avoid Dask serialization issues.
    Returns either the result dict or a tuple (filepath, Exception).
    """
    filepath, max_branches = args
    try:
        return _inspect_file_with_limit(filepath, max_branches)
    except Exception as e:
        # Return tuple with filepath for error tracking
        return (filepath, e)


def inspect_dataset_distributed(
    client,
    filepaths: List[str],
    max_branches: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """Inspect multiple files in parallel using Dask with robust error handling.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask distributed client
    filepaths : List[str]
        List of file paths to inspect
    max_branches : int, optional
        Limit branch info to first N branches (faster for large files)
        If None, inspect all branches

    Returns
    -------
    results : List[dict]
        List of metadata dicts for successfully inspected files
    error_summary : dict
        Summary of failures with keys:
        - total_files: Total files attempted
        - successful: Number of successful inspections
        - failed: Number of failed inspections
        - failures: List of failure dicts with filepath, error_type, error_message

    Examples
    --------
    >>> from dask.distributed import Client
    >>> client = Client()
    >>> results, errors = inspect_dataset_distributed(client, file_list)
    >>> print(f"Success: {errors['successful']}/{errors['total_files']}")
    >>> if errors['failures']:
    ...     for failure in errors['failures']:
    ...         print(f"Failed: {failure['filepath']} - {failure['error_type']}")
    """
    # Create tuples of (filepath, max_branches) for each file
    file_args = [(fp, max_branches) for fp in filepaths]

    # Use reasonable number of partitions (not one per file!)
    # Aim for ~100 files per partition, or at least use fewer partitions
    n_partitions = min(len(filepaths), max(1, len(filepaths) // 100))

    # Use Dask bag to distribute inspection with error handling
    bag = db.from_sequence(file_args, npartitions=n_partitions)
    raw_results = bag.map(_safe_inspect_wrapper).compute()

    # Separate successes from failures
    successes = []
    failures = []

    for result in raw_results:
        if isinstance(result, tuple):
            # This is an error: (filepath, Exception)
            filepath, exception = result
            failures.append({
                "filepath": filepath,
                "error_type": type(exception).__name__,
                "error_message": str(exception),
            })
        else:
            # This is a successful result dict
            successes.append(result)

    # Build error summary
    error_summary = {
        "total_files": len(filepaths),
        "successful": len(successes),
        "failed": len(failures),
        "failures": failures,
    }

    return successes, error_summary


def format_error_summary(error_summary: Dict) -> str:
    """Format error summary as a readable table.

    Parameters
    ----------
    error_summary : dict
        Error summary from inspect_dataset_distributed with keys:
        - total_files: Total files attempted
        - successful: Number of successful inspections
        - failed: Number of failed inspections
        - failures: List of failure dicts

    Returns
    -------
    formatted : str
        Formatted string representation with summary and failure table

    Examples
    --------
    >>> results, errors = inspect_dataset_distributed(client, file_list)
    >>> print(format_error_summary(errors))
    === Inspection Summary ===
    Total files:   100
    Successful:     98
    Failed:          2
    Success rate: 98.0%

    === Failures ===
    File                              Error Type        Error Message
    ──────────────────────────────────────────────────────────────────────────
    /path/to/file1.root              FileNotFoundError No such file or directory
    /path/to/file2.root              OSError           Connection timeout
    """
    lines = []

    # Summary section
    lines.append("=== Inspection Summary ===")
    lines.append(f"Total files:   {error_summary['total_files']:>4}")
    lines.append(f"Successful:    {error_summary['successful']:>4}")
    lines.append(f"Failed:        {error_summary['failed']:>4}")

    if error_summary['total_files'] > 0:
        success_rate = 100 * error_summary['successful'] / error_summary['total_files']
        lines.append(f"Success rate: {success_rate:>5.1f}%")

    # Failures table if any
    if error_summary['failed'] > 0:
        lines.append("")
        lines.append("=== Failures ===")

        # Calculate column widths
        max_file_len = max(len(f['filepath']) for f in error_summary['failures'])
        max_type_len = max(len(f['error_type']) for f in error_summary['failures'])
        max_msg_len = max(len(f['error_message']) for f in error_summary['failures'])

        # Ensure minimum widths
        file_width = max(max_file_len, len("File"))
        type_width = max(max_type_len, len("Error Type"))
        msg_width = max(max_msg_len, len("Error Message"))

        # Limit message width for readability
        msg_width = min(msg_width, 60)

        # Header
        header = f"{'File':<{file_width}}  {'Error Type':<{type_width}}  {'Error Message':<{msg_width}}"
        lines.append(header)
        lines.append("─" * len(header))

        # Failure rows
        for failure in error_summary['failures']:
            filepath = failure['filepath']
            error_type = failure['error_type']
            error_msg = failure['error_message']

            # Truncate long messages
            if len(error_msg) > msg_width:
                error_msg = error_msg[:msg_width-3] + "..."

            row = f"{filepath:<{file_width}}  {error_type:<{type_width}}  {error_msg:<{msg_width}}"
            lines.append(row)

    return "\n".join(lines)


def get_total_events_distributed(client, filepaths: List[str]) -> int:
    """Get total event count across files using Dask fold.

    Fast aggregation for when you only need event counts.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask distributed client
    filepaths : List[str]
        List of file paths

    Returns
    -------
    total_events : int
        Sum of events across all files

    Examples
    --------
    >>> total = get_total_events_distributed(client, file_list)
    >>> print(f"Total events: {total:,}")
    """

    def get_entries(filepath: str) -> int:
        """Get number of entries from file."""
        try:
            with uproot.open(filepath) as f:
                return f["Events"].num_entries
        except Exception:
            return 0

    # Use fold to aggregate
    bag = db.from_sequence(filepaths)
    futures = bag.map(get_entries)
    task = futures.fold(lambda x, y: x + y, split_every=4)
    total_events = task.compute()

    return total_events
