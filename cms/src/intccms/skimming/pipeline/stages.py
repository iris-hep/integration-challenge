"""Pipeline stages for event processing.

Pure functions that perform individual operations in the skimming pipeline:
loading events, applying selections, extracting columns, and saving results.
"""

import logging
from typing import Dict, Callable, Tuple, Any, Optional
import awkward as ak

from intccms.skimming.io.protocols import EventReader, EventWriter

logger = logging.getLogger(__name__)


def load_events(
    reader: EventReader,
    path: str,
    tree_name: str,
    entry_start: Optional[int] = None,
    entry_stop: Optional[int] = None,
    **reader_kwargs
) -> ak.Array:
    """Load events from a file.

    Args:
        reader: EventReader instance (RootReader, ParquetReader, etc.)
        path: Path to the input file
        tree_name: Name of the tree to read (for ROOT files)
        entry_start: First entry to read (inclusive)
        entry_stop: Last entry to read (exclusive)
        **reader_kwargs: Additional arguments passed to the reader

    Returns:
        Awkward array containing the loaded events

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If entry range is invalid

    Examples:
        >>> from intccms.skimming.io import RootReader
        >>> reader = RootReader()
        >>> events = load_events(
        ...     reader,
        ...     "/path/to/file.root",
        ...     "Events",
        ...     entry_start=0,
        ...     entry_stop=10000
        ... )
    """
    return reader.read(
        path=path,
        tree_name=tree_name,
        entry_start=entry_start,
        entry_stop=entry_stop,
        **reader_kwargs
    )


def apply_selection(
    events: ak.Array,
    selection_func: Callable,
    selection_args: Tuple,
    selection_kwargs: Optional[Dict[str, Any]] = None
) -> ak.Array:
    """Apply selection mask to filter events.

    Args:
        events: Input events as awkward array
        selection_func: Function that returns a PackedSelection
        selection_args: Positional arguments to pass to selection_func
        selection_kwargs: Keyword arguments to pass to selection_func

    Returns:
        Filtered awkward array containing only selected events

    Examples:
        >>> def my_selection(events):
        ...     from coffea.analysis_tools import PackedSelection
        ...     selection = PackedSelection()
        ...     selection.add("high_pt", events.pt > 50)
        ...     return selection
        >>>
        >>> filtered = apply_selection(
        ...     events,
        ...     my_selection,
        ...     (events,),
        ...     {}
        ... )
    """
    if selection_kwargs is None:
        selection_kwargs = {}

    # Execute selection function
    packed_selection = selection_func(*selection_args, **selection_kwargs)

    # Extract mask
    selection_names = packed_selection.names
    if selection_names:
        # Use the last selection as the final one
        final_selection = selection_names[-1]
        mask = packed_selection.all(final_selection)
    else:
        # No selection applied, keep all events
        mask = slice(None)

    # Apply mask and return filtered events
    return events[mask]


def extract_columns(
    events: ak.Array,
    columns_to_keep: list[str],
    is_data: bool = False
) -> Dict[str, ak.Array]:
    """Extract specified columns from events.

    Handles nested fields (e.g., "Muon.pt") and filters MC-only branches for data.

    Args:
        events: Input events as awkward array with NanoAOD schema
        columns_to_keep: List of column names to extract (e.g., ["Muon.pt", "run"])
        is_data: If True, skip MC-only columns (those containing "gen")

    Returns:
        Dictionary mapping column names to awkward arrays

    Examples:
        >>> columns = extract_columns(
        ...     events,
        ...     ["Muon.pt", "Muon.eta", "run", "event"],
        ...     is_data=False
        ... )
        >>> columns.keys()
        dict_keys(['Muon_pt', 'Muon_eta', 'run', 'event'])
    """
    output_columns = {}

    for col in columns_to_keep:
        # Skip MC-only columns for data
        if is_data and "gen" in col.lower():
            continue

        # Handle nested fields (e.g., "Muon.pt" -> events.Muon.pt)
        if "." in col:
            parts = col.split(".")
            value = events
            for part in parts:
                value = getattr(value, part)
            # Use underscore for output key
            output_key = "_".join(parts)
        else:
            # Top-level field
            if not hasattr(events, col):
                logger.warning(f"Column '{col}' not found in events, skipping")
                continue
            value = getattr(events, col)
            output_key = col

        output_columns[output_key] = value

    return output_columns


def save_events(
    writer: EventWriter,
    events: Dict[str, ak.Array],
    output_path: str,
    **writer_kwargs
) -> str:
    """Save events to a file.

    Args:
        writer: EventWriter instance (ParquetWriter, RootWriter, etc.)
        events: Dictionary mapping column names to awkward arrays
        output_path: Path where the file will be written
        **writer_kwargs: Additional arguments passed to the writer

    Returns:
        The output path (for tracking purposes)

    Raises:
        IOError: If writing fails
        ValueError: If events dictionary is empty

    Examples:
        >>> from intccms.skimming.io import ParquetWriter
        >>> writer = ParquetWriter()
        >>> events_dict = {"pt": ak.Array([10, 20, 30])}
        >>> path = save_events(
        ...     writer,
        ...     events_dict,
        ...     "/path/to/output.parquet",
        ...     compression="zstd"
        ... )
    """
    # Writer handles directory creation and actual writing
    writer.write(events, output_path, **writer_kwargs)
    return output_path
