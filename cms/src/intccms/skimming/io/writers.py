"""Event writers for saving data to various file formats.

This module provides concrete implementations of the EventWriter base class for
different file formats (ROOT TTree, Parquet). Each writer handles the format-specific
details of persisting awkward arrays to disk.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import awkward as ak
import uproot

from .protocols import EventWriter

logger = logging.getLogger(__name__)


def _is_remote_path(path: str) -> bool:
    """Check if path is a remote URI (has a scheme like root://, s3://, etc.).

    Args:
        path: Path or URI string

    Returns:
        True if path has a URI scheme, False otherwise
    """
    return "://" in path


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for local paths, skip for remote URIs.

    For local paths, creates parent directories if they don't exist.
    For remote URIs (e.g., root://host/path), does nothing as the
    remote filesystem handles directory creation.

    Args:
        path: Local path or remote URI
    """
    if not _is_remote_path(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)


class ParquetWriter(EventWriter):
    """Writer for Parquet files using awkward-array.

    This writer saves events to Parquet format using awkward's native parquet
    writer. Supports compression and other parquet-specific options.

    Examples:
        >>> writer = ParquetWriter()
        >>> events = {
        ...     "Muon_pt": ak.Array([25.0, 30.0, 40.0]),
        ...     "Muon_eta": ak.Array([0.5, 1.0, -0.5])
        ... }
        >>> writer.write(events, "/path/to/output.parquet", compression="zstd")
    """

    def write(
        self,
        events: Dict[str, ak.Array],
        path: str,
        **kwargs
    ) -> None:
        """Write events to a Parquet file.

        Args:
            events: Dictionary mapping column/branch names to awkward arrays.
                Each key is a column name, each value is the data for that column.
            path: Path where the Parquet file will be written
            **kwargs: Additional keyword arguments passed to ak.to_parquet().
                Common options include 'compression' (e.g., "zstd", "gzip", None).

        Raises:
            ValueError: If events dictionary is empty
            IOError: If writing to the file fails
        """
        if not events:
            logger.warning(f"No events to write to {path}; skipping parquet write.")
            return

        # Ensure parent directory exists
        _ensure_parent_dir(path)

        # Build zipped payload with all columns at same level
        payload = ak.zip(events, depth_limit=1)

        # Write to parquet
        ak.to_parquet(payload, path, **kwargs)


class RootWriter(EventWriter):
    """Writer for ROOT TTree files using uproot.

    This writer saves events to ROOT TTree format using uproot. It creates
    a new ROOT file with a single TTree containing all the specified branches.

    Examples:
        >>> writer = RootWriter()
        >>> events = {
        ...     "Muon_pt": ak.Array([25.0, 30.0, 40.0]),
        ...     "Muon_eta": ak.Array([0.5, 1.0, -0.5])
        ... }
        >>> writer.write(
        ...     events,
        ...     "/path/to/output.root",
        ...     tree_name="Events",
        ...     compression=uproot.ZLIB(4)
        ... )
    """

    def write(
        self,
        events: Dict[str, ak.Array],
        path: str,
        tree_name: str = "Events",
        **kwargs
    ) -> None:
        """Write events to a ROOT TTree file.

        Args:
            events: Dictionary mapping branch names to awkward arrays.
                Each key is a branch name, each value is the data for that branch.
            path: Path where the ROOT file will be written
            tree_name: Name of the TTree to create (default: "Events")
            **kwargs: Additional keyword arguments. Special keys:
                - 'tree_kwargs': Dict of kwargs passed to mktree() for tree creation
                - All other kwargs passed to uproot.recreate() for file creation

        Raises:
            ValueError: If events dictionary is empty
            IOError: If writing to the file fails
        """
        if not events:
            logger.warning(f"No events to write to {path}; skipping ROOT write.")
            return

        # Ensure parent directory exists
        _ensure_parent_dir(path)

        # Separate file-level and tree-level kwargs
        file_kwargs = dict(kwargs)
        tree_kwargs = file_kwargs.pop("tree_kwargs", {})

        # Determine branch types from awkward arrays
        branch_types = {k: v.type for k, v in events.items()}

        # Write to ROOT file
        with uproot.recreate(path, **file_kwargs) as root_file:
            tree = root_file.mktree(tree_name, branch_types, **tree_kwargs)
            tree.extend(events)


def get_writer(format: str) -> Any:
    """Factory function to get the appropriate writer for a file format.

    Args:
        format: File format identifier. Supported values:
            - "root" or "root_ttree": ROOT TTree format
            - "parquet": Parquet format

    Returns:
        Writer instance for the specified format

    Raises:
        ValueError: If the format is not supported

    Examples:
        >>> writer = get_writer("parquet")
        >>> writer.write({"pt": ak.Array([1, 2, 3])}, "/path/to/output.parquet")
    """
    format_map = {
        "root": RootWriter(),
        "root_ttree": RootWriter(),
        "parquet": ParquetWriter(),
    }

    if format not in format_map:
        raise ValueError(
            f"Unsupported writer format: {format}. "
            f"Supported formats: {list(format_map.keys())}"
        )

    return format_map[format]
