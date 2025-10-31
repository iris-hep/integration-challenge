"""Event readers for loading data from various file formats.

This module provides concrete implementations of the EventReader base class for
different file formats (ROOT, Parquet). Each reader uses the coffea NanoEventsFactory
to load events with the NanoAOD schema.
"""

from pathlib import Path
from typing import Optional, Any
import awkward as ak
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from .protocols import EventReader


class RootReader(EventReader):
    """Reader for ROOT files using NanoEventsFactory.

    This reader loads events from ROOT TTree files, applying the NanoAOD schema
    and supporting entry-range slicing for processing file chunks.

    Examples:
        >>> reader = RootReader()
        >>> events = reader.read(
        ...     path="/path/to/file.root",
        ...     tree_name="Events",
        ...     entry_start=0,
        ...     entry_stop=100000
        ... )
    """

    def read(
        self,
        path: str,
        tree_name: str,
        entry_start: Optional[int] = None,
        entry_stop: Optional[int] = None,
        **kwargs
    ) -> ak.Array:
        """Read events from a ROOT file.

        Args:
            path: Path to the ROOT file
            tree_name: Name of the TTree to read from
            entry_start: First entry to read (inclusive). If None, starts from beginning.
            entry_stop: Last entry to read (exclusive). If None, reads to end.
            **kwargs: Additional keyword arguments passed to NanoEventsFactory.from_root().
                Common options include 'schemaclass' (defaults to NanoAODSchema).

        Returns:
            Awkward array containing the events with NanoAOD schema

        Raises:
            FileNotFoundError: If the ROOT file doesn't exist
            ValueError: If tree_name is not found in the file
        """
        kwargs.setdefault("schemaclass", NanoAODSchema)

        # Build fileset dict for NanoEventsFactory
        fileset = {str(path): tree_name}

        reader = NanoEventsFactory.from_root(
            fileset,
            entry_start=entry_start,
            entry_stop=entry_stop,
            **kwargs
        )
        return reader.events()


class ParquetReader(EventReader):
    """Reader for Parquet files using NanoEventsFactory.

    This reader loads events from Parquet files that were written by awkward-array,
    applying the NanoAOD schema and supporting entry-range slicing.

    Examples:
        >>> reader = ParquetReader()
        >>> events = reader.read(
        ...     path="/path/to/file.parquet",
        ...     tree_name=None,  # Not used for parquet
        ...     entry_start=0,
        ...     entry_stop=100000
        ... )
    """

    def read(
        self,
        path: str,
        tree_name: str,
        entry_start: Optional[int] = None,
        entry_stop: Optional[int] = None,
        **kwargs
    ) -> ak.Array:
        """Read events from a Parquet file.

        Args:
            path: Path to the Parquet file
            tree_name: Not used for Parquet format (included for interface compatibility)
            entry_start: First entry to read (inclusive). If None, starts from beginning.
            entry_stop: Last entry to read (exclusive). If None, reads to end.
            **kwargs: Additional keyword arguments passed to NanoEventsFactory.from_parquet().
                Common options include 'schemaclass' (defaults to NanoAODSchema).

        Returns:
            Awkward array containing the events with NanoAOD schema

        Raises:
            FileNotFoundError: If the Parquet file doesn't exist
        """
        kwargs.setdefault("schemaclass", NanoAODSchema)

        reader = NanoEventsFactory.from_parquet(
            str(path),
            entry_start=entry_start,
            entry_stop=entry_stop,
            **kwargs
        )
        return reader.events()


def get_reader(format: str) -> Any:
    """Factory function to get the appropriate reader for a file format.

    Args:
        format: File format identifier. Supported values:
            - "root" or "root_ttree": ROOT TTree format
            - "parquet": Parquet format

    Returns:
        Reader instance for the specified format

    Raises:
        ValueError: If the format is not supported

    Examples:
        >>> reader = get_reader("parquet")
        >>> events = reader.read("/path/to/file.parquet", None)
    """
    format_map = {
        "root": RootReader(),
        "root_ttree": RootReader(),
        "parquet": ParquetReader(),
    }

    if format not in format_map:
        raise ValueError(
            f"Unsupported reader format: {format}. "
            f"Supported formats: {list(format_map.keys())}"
        )

    return format_map[format]
