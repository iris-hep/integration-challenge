"""Abstract base classes defining interfaces for event I/O operations.

This module defines the abstract interfaces for reading and writing events
from/to various file formats using ABC (Abstract Base Classes) for explicit
inheritance and interface enforcement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import awkward as ak


class EventReader(ABC):
    """Abstract base class for reading events from files.

    Implementations must provide a `read` method that loads events from a file
    with specified entry ranges. The reader should return events as awkward arrays
    compatible with the NanoAOD schema.

    Examples:
        >>> reader = RootReader()
        >>> events = reader.read(
        ...     path="/path/to/file.root",
        ...     tree_name="Events",
        ...     entry_start=0,
        ...     entry_stop=1000
        ... )
    """

    @abstractmethod
    def read(
        self,
        path: str,
        tree_name: str,
        entry_start: Optional[int] = None,
        entry_stop: Optional[int] = None,
        **kwargs
    ) -> ak.Array:
        """Read events from a file.

        Args:
            path: Path to the input file
            tree_name: Name of the tree to read (for ROOT files) or None for formats
                that don't use tree names
            entry_start: First entry to read (inclusive). If None, starts from beginning.
            entry_stop: Last entry to read (exclusive). If None, reads to end.
            **kwargs: Additional format-specific keyword arguments

        Returns:
            Awkward array containing the events with NanoAOD schema

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If entry_start/entry_stop are invalid
        """
        pass


class EventWriter(ABC):
    """Abstract base class for writing events to files.

    Implementations must provide a `write` method that saves events (represented
    as a dictionary of awkward arrays) to a file in a specific format.

    Examples:
        >>> writer = ParquetWriter()
        >>> events = {"Muon_pt": ak.Array([25.0, 30.0]), "Muon_eta": ak.Array([0.5, 1.0])}
        >>> writer.write(events, path="/path/to/output.parquet")
    """

    @abstractmethod
    def write(
        self,
        events: Dict[str, ak.Array],
        path: str,
        **kwargs
    ) -> None:
        """Write events to a file.

        Args:
            events: Dictionary mapping column names to awkward arrays.
                Keys are branch/column names, values are the data arrays.
            path: Path to the output file
            **kwargs: Additional format-specific keyword arguments

        Raises:
            IOError: If writing to the file fails
            ValueError: If events dictionary is empty or malformed
        """
        pass
