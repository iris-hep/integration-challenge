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

    def inspect_with_limit(filepath: str) -> Dict:
        """Wrapper to apply branch limit."""
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

    def safe_inspect(filepath: str) -> Union[Dict, Exception]:
        """Wrapper that catches all exceptions during inspection."""
        try:
            return inspect_with_limit(filepath)
        except Exception as e:
            # Return exception instead of raising
            return e

    # Use Dask bag to distribute inspection with error handling
    bag = db.from_sequence(filepaths, npartitions=len(filepaths))
    raw_results = bag.map(safe_inspect).compute()

    # Separate successes from failures
    successes = []
    failures = []

    for filepath, result in zip(filepaths, raw_results):
        if isinstance(result, Exception):
            failures.append({
                "filepath": filepath,
                "error_type": type(result).__name__,
                "error_message": str(result),
            })
        else:
            successes.append(result)

    # Build error summary
    error_summary = {
        "total_files": len(filepaths),
        "successful": len(successes),
        "failed": len(failures),
        "failures": failures,
    }

    return successes, error_summary


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
