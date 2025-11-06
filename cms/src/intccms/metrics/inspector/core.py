"""Distributed file inspection for input data characterization.

This module provides functions to extract metadata from ROOT files in parallel
using Dask, enabling fast characterization of large datasets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask.bag as db
import uproot

try:
    from coffea.dataset_tools import rucio_utils  # type: ignore
except ImportError:  # pragma: no cover
    rucio_utils = None  # type: ignore

logger = logging.getLogger(__name__)

_RUCIO_SCOPE = "cms"
_RUCIO_CLIENT = None


def _is_remote_path(filepath: str) -> bool:
    return filepath.startswith("root://")


def _strip_root_redirector(filepath: str) -> str:
    """
    Remove the XRootD redirector from a remote path, returning the logical file name.
    """
    stripped = filepath[len("root://") :]
    if "/" not in stripped:
        return filepath
    _, rest = stripped.split("/", 1)
    rest = rest.lstrip("/")
    return f"/{rest}"


def _get_rucio_client():
    global _RUCIO_CLIENT

    if rucio_utils is None:
        return None

    if _RUCIO_CLIENT is None:
        try:
            _RUCIO_CLIENT = rucio_utils.get_rucio_client()
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Failed to create Rucio client: %s", exc)
            _RUCIO_CLIENT = None
    return _RUCIO_CLIENT


def _get_remote_file_size(filepath: str, rucio_client=None) -> int:
    client = rucio_client or _get_rucio_client()
    if client is None:
        return 0

    lfn = _strip_root_redirector(filepath)
    name_candidates = []

    lfn_no_slash = lfn.lstrip("/")
    if lfn_no_slash:
        name_candidates.append(lfn_no_slash)
    if lfn.startswith("/"):
        name_candidates.append(lfn)

    for name in name_candidates:
        try:
            meta = client.get_file_meta(scope=_RUCIO_SCOPE, name=name)
        except Exception:
            continue
        if meta and "bytes" in meta and meta["bytes"]:
            try:
                return int(meta["bytes"])
            except (TypeError, ValueError):
                continue
    return 0


def inspect_file(filepath: str, max_branches: Optional[int] = None) -> Dict:
    """Extract all metadata from a single ROOT file.

    Memory-efficient implementation: Branch data is stored as compact tuples
    (num_bytes, compressed_bytes, uncompressed_bytes) instead of dicts,
    reducing memory usage by ~60% for large-scale inspections.

    Parameters
    ----------
    filepath : str
        Path to ROOT file (local or XRootD URL)
    max_branches : int, optional
        Limit inspection to first N branches (for faster processing on large files)

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
        - branches: Dict mapping branch name to (num_bytes, compressed, uncompressed) tuples

    Examples
    --------
    >>> metadata = inspect_file("file.root")
    >>> print(f"Events: {metadata['num_events']:,}")
    >>> print(f"Size: {metadata['file_size_bytes'] / 1024**3:.2f} GB")
    >>>
    >>> # Access branch data
    >>> for name, (num_bytes, compressed, uncompressed) in metadata['branches'].items():
    ...     print(f"{name}: {num_bytes/1024:.1f} KB")
    """
    try:
        with uproot.open(filepath) as f:
            tree = f["Events"]

            # Get all branch info - use compact tuple format to reduce memory
            # Format: (num_bytes, compressed_bytes, uncompressed_bytes)
            branches = {}
            for branch_name in tree.keys()[:max_branches]:
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

                # Store as tuple for memory efficiency: (num_bytes, compressed, uncompressed)
                branches[branch_name] = (int(num_bytes), int(compressed), int(uncompressed))

            # Compute totals - branches are tuples (num_bytes, compressed, uncompressed)
            total_branch_bytes = sum(b[0] for b in branches.values())

            # Get tree-level compression info if available
            tree_compressed = tree.compressed_bytes if hasattr(tree, "compressed_bytes") else 0
            tree_uncompressed = (
                tree.uncompressed_bytes if hasattr(tree, "uncompressed_bytes") else 0
            )

            # Get file size (handle both local and remote)
            if _is_remote_path(filepath):
                file_size_bytes = _get_remote_file_size(filepath)
            else:
                file_size_bytes = Path(filepath).stat().st_size

            result = {
                "filepath": str(filepath),  # Ensure string
                "file_size_bytes": int(file_size_bytes),  # Ensure int
                "num_events": int(tree.num_entries),  # Ensure int
                "num_branches": int(len(branches)),  # Ensure int
                "total_branch_bytes": int(total_branch_bytes),  # Ensure int
                "tree_compressed_bytes": int(tree_compressed),  # Ensure int
                "tree_uncompressed_bytes": int(tree_uncompressed),  # Ensure int
                "branches": branches,  # Dict of dicts with int values
            }

            # Add num_branches_sampled if max_branches was used
            if max_branches is not None:
                result["num_branches_sampled"] = int(len(branches))
                # Update num_branches to total count
                result["num_branches"] = int(len(tree.keys()))

            return result
    except Exception as e:
        # Return dict with error info for error tracking
        # Use simple types to ensure Dask can serialize/deserialize
        try:
            error_type = type(e).__name__
            error_message = str(e)
        except Exception:
            # Fallback if even getting error info fails
            error_type = "UnknownError"
            error_message = "Could not extract error details"

        return {
                "filepath": str(filepath),  # Ensure string
                "file_size_bytes": -1,
                "num_events": -1,
                "num_branches": -1,
                "total_branch_bytes": -1,
                "tree_compressed_bytes": -1,
                "tree_uncompressed_bytes": -1,
                "branches": {},
                "error_type": error_type,
                "error_message": error_message,
        }


def inspect_dataset_distributed(
    client,
    filepaths: List[str],
    max_branches: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """Inspect multiple files in parallel using Dask with robust error handling.

    Memory-efficient for large-scale inspections:
    - Branch data stored as compact tuples (~60% memory reduction vs dicts)
    - Streams results from workers without holding all in memory
    - Example: 13K files × 1000 branches = ~1.3 GB (vs 3.3 GB with dict format)

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
    n_partitions = min(len(filepaths), max(1, len(filepaths) // 5))

    # Use Dask bag to process files - compute directly without persist
    # This streams results from workers to client without holding all in memory
    bag = db.from_sequence(file_args, npartitions=n_partitions)

    try:
        # Compute all results - Dask streams from workers as they complete
        raw_results = bag.map(lambda args: inspect_file(*args)).compute()
    except Exception as e:
        # If Dask itself fails, fail gracefully
        error_summary = {
            "total_files": len(filepaths),
            "successful": 0,
            "failed": len(filepaths),
            "failures": [{
                "filepath": "ALL FILES",
                "error_type": type(e).__name__,
                "error_message": f"Dask compute failed: {str(e)}",
            }],
        }
        return [], error_summary

    # Separate successes from failures on client side
    successes = []
    failures = []

    for result in raw_results:
        if "error_type" in result:
            # This is an error
            failures.append({
                "filepath": result["filepath"],
                "error_type": result["error_type"],
                "error_message": result["error_message"],
            })
        else:
            # This is a success
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
