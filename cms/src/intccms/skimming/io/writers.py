"""Event writers for saving data to various file formats.

This module provides concrete implementations of the EventWriter base class for
different file formats (ROOT TTree, Parquet). Each writer handles the format-specific
details of persisting awkward arrays to disk.
"""

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional

import awkward as ak
import uproot
from XRootD import client as xrd_client

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


def _is_xrd_path(path: str) -> bool:
    """Check if path uses the xrootd protocol (root://)."""
    return path.startswith("root://")


@contextmanager
def _xrd_write_workaround(path: str):
    """Write locally then copy to xrootd. Workaround for uproot remote write issues.

    uproot opens sinks in r+b mode, but fsspec-xrootd's simplecache does not
    upload on close for r+b, and fsspec-xrootd lacks _put_file. This writes to
    a local temp file, then copies to xrootd via the XRootD Python client.

    Remove when fixed upstream (fsspec-xrootd _put_file + uproot sink mode).
    To remove: delete this function and unwrap the ``with`` blocks in ROOT writers.

    Args:
        path: Target output path, may be root:// or local.

    Yields:
        str: Local temp path (if xrootd) or original path (if local).
    """
    if not _is_xrd_path(path):
        yield path
        return

    filename = path.rsplit("/", 1)[-1]
    suffix = f".{filename.rsplit('.', 1)[-1]}" if "." in filename else ".root"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        yield tmp_path

        process = xrd_client.CopyProcess()
        process.add_job(tmp_path, path, force=True)
        process.prepare()
        status, _ = process.run()
        if not status.ok:
            raise IOError(f"XRootD copy to {path} failed: {status.message}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class ParquetWriter(EventWriter):
    """Writer for Parquet files using awkward-array.

    This writer saves events to Parquet format using awkward's native parquet
    writer. Supports compression and other parquet-specific options.

    The writer automatically appends .parquet extension if not present.

    Examples:
        >>> writer = ParquetWriter()
        >>> events = {
        ...     "Muon_pt": ak.Array([25.0, 30.0, 40.0]),
        ...     "Muon_eta": ak.Array([0.5, 1.0, -0.5])
        ... }
        >>> writer.write(events, "/path/to/output", compression="zstd")
    """

    @property
    def file_extension(self) -> str:
        """File extension for Parquet format.

        Returns:
            str: ".parquet"
        """
        return ".parquet"

    def write(
        self,
        events: Dict[str, ak.Array],
        path: str,
        **kwargs
    ) -> str:
        """Write events to a Parquet file.

        Automatically appends .parquet extension if the path doesn't have it.

        Args:
            events: Dictionary mapping column/branch names to awkward arrays.
                Each key is a column name, each value is the data for that column.
            path: Path where the Parquet file will be written (extension added if missing)
            **kwargs: Additional keyword arguments passed to ak.to_parquet().
                Common options include 'compression' (e.g., "zstd", "gzip", None).

        Returns:
            str: The actual path written to (with extension)

        Raises:
            ValueError: If events dictionary is empty
            IOError: If writing to the file fails
        """
        if not events:
            logger.warning(f"No events to write to {path}; skipping parquet write.")
            return path

        # Auto-append extension if missing
        if not path.endswith(self.file_extension):
            path = f"{path}{self.file_extension}"

        # Ensure parent directory exists
        _ensure_parent_dir(path)

        # Build zipped payload with all columns at same level
        payload = ak.zip(events, depth_limit=1)

        # Write to parquet
        ak.to_parquet(payload, path, **kwargs)

        return path


class RootWriter(EventWriter):
    """Writer for ROOT TTree files using uproot.

    This writer saves events to ROOT TTree format using uproot. It creates
    a new ROOT file with a single TTree containing all the specified branches.

    The writer automatically appends .root extension if not present.

    Examples:
        >>> writer = RootWriter()
        >>> events = {
        ...     "Muon_pt": ak.Array([25.0, 30.0, 40.0]),
        ...     "Muon_eta": ak.Array([0.5, 1.0, -0.5])
        ... }
        >>> writer.write(
        ...     events,
        ...     "/path/to/output",
        ...     tree_name="Events",
        ...     compression=uproot.ZLIB(4)
        ... )
    """

    @property
    def file_extension(self) -> str:
        """File extension for ROOT format.

        Returns:
            str: ".root"
        """
        return ".root"

    def write(
        self,
        events: Dict[str, ak.Array],
        path: str,
        tree_name: str = "Events",
        **kwargs
    ) -> str:
        """Write events to a ROOT TTree file.

        Automatically appends .root extension if the path doesn't have it.

        Args:
            events: Dictionary mapping branch names to awkward arrays.
                Each key is a branch name, each value is the data for that branch.
            path: Path where the ROOT file will be written (extension added if missing)
            tree_name: Name of the TTree to create (default: "Events")
            **kwargs: Additional keyword arguments passed to uproot.recreate().

        Returns:
            str: The actual path written to (with extension)

        Raises:
            ValueError: If events dictionary is empty
            IOError: If writing to the file fails
        """
        if not events:
            logger.warning(f"No events to write to {path}; skipping ROOT write.")
            return path

        # Auto-append extension if missing
        if not path.endswith(self.file_extension):
            path = f"{path}{self.file_extension}"

        # Ensure parent directory exists
        _ensure_parent_dir(path)

        branch_types = {k: v.type for k, v in events.items()}

        with _xrd_write_workaround(path) as write_path:
            with uproot.recreate(write_path, **kwargs) as root_file:
                tree = root_file.mktree(tree_name, branch_types)
                tree.extend(events)

        return path


class RNTupleWriter(EventWriter):
    """Writer for ROOT RNTuple files using uproot.

    RNTuple is ROOT's columnar data format, succeeding TTree. Uses
    uproot.recreate + mkrntuple which accepts data directly.

    Examples:
        >>> writer = RNTupleWriter()
        >>> events = {
        ...     "Muon_pt": ak.Array([25.0, 30.0, 40.0]),
        ...     "Muon_eta": ak.Array([0.5, 1.0, -0.5])
        ... }
        >>> writer.write(events, "/path/to/output", ntuple_name="Events")
    """

    @property
    def file_extension(self) -> str:
        return ".root"

    def write(
        self,
        events: Dict[str, ak.Array],
        path: str,
        tree_name: str = "Events",
        **kwargs
    ) -> str:
        """Write events to a ROOT RNTuple file.

        Args:
            events: Dictionary mapping field names to awkward arrays.
            path: Path where the ROOT file will be written (extension added if missing)
            tree_name: Name of the RNTuple to create (default: "Events")
            **kwargs: Additional keyword arguments passed to uproot.recreate().

        Returns:
            str: The actual path written to (with extension)
        """
        if not events:
            logger.warning(f"No events to write to {path}; skipping RNTuple write.")
            return path

        if not path.endswith(self.file_extension):
            path = f"{path}{self.file_extension}"

        _ensure_parent_dir(path)

        with _xrd_write_workaround(path) as write_path:
            with uproot.recreate(write_path, **kwargs) as root_file:
                root_file.mkrntuple(tree_name, events)

        return path


def get_writer(format: str) -> Any:
    """Factory function to get the appropriate writer for a file format.

    Args:
        format: File format identifier. Supported values:
            - "ttree": ROOT TTree format
            - "rntuple": ROOT RNTuple format
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
        "ttree": RootWriter(),
        "rntuple": RNTupleWriter(),
        "parquet": ParquetWriter(),
    }

    if format not in format_map:
        raise ValueError(
            f"Unsupported writer format: {format}. "
            f"Supported formats: {list(format_map.keys())}"
        )

    return format_map[format]
