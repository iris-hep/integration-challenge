"""WorkItem processing for parallel skimming operations.

This module handles processing of individual coffea WorkItems, including output path
resolution and integration with the pipeline stages.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypedDict, Union

from coffea.processor.executor import WorkItem

from intccms.skimming.io.readers import get_reader
from intccms.skimming.io.writers import get_writer
from intccms.skimming.pipeline.stages import (
    build_column_list,
    extract_columns,
    load_events,
    save_events,
)
from intccms.schema import SkimmingConfig, WorkerEval
from intccms.skimming.utils import default_histogram
from intccms.utils.functors import SelectionExecutor

logger = logging.getLogger(__name__)


# Type definitions for workitem processing structures
class ManifestEntry(TypedDict):
    """Metadata for a single workitem's output file.

    This structure tracks the relationship between input workitems and their
    output files, enabling reconstruction of the processing history.
    """
    source_file: str  # Original ROOT/parquet file path
    entrystart: int  # Starting entry index in source file
    entrystop: int  # Ending entry index in source file
    dataset: str  # Dataset name from workitem
    treename: str  # Tree name in source file
    output_file: str  # Path to generated output file
    processed_events: int  # Number of events after selection
    total_events: int  # Total events in entry range


class WorkitemResult(TypedDict):
    """Result dictionary from processing a single workitem.

    This structure is returned by process_workitem and aggregated by Dask.
    """
    hist: object  # Dummy histogram for coffea compatibility
    failed_items: Set[WorkItem]  # Empty set on success, contains workitem if failed
    processed_events: int  # Total events after selection across all files
    output_files: List[str]  # List of created output file paths
    manifest_entries: List[ManifestEntry]  # Metadata for each processed chunk


def get_deterministic_fileuuid(file_path: str) -> bytes:
    """Generate deterministic UUID from file path.

    Uses MD5 hash of absolute path to create a stable 16-byte UUID.
    Same file path will always produce the same UUID, ensuring deterministic
    behavior for coffea's WorkItem processing, caching, and worker affinity.

    Args:
        file_path: Path to the file

    Returns:
        bytes: 16-byte UUID compatible with coffea's WorkItem

    Examples:
        >>> uuid1 = get_deterministic_fileuuid("/path/to/file.parquet")
        >>> uuid2 = get_deterministic_fileuuid("/path/to/file.parquet")
        >>> uuid1 == uuid2
        True
    """
    # Normalize path and hash it
    normalized_path = str(Path(file_path).resolve())
    hash_bytes = hashlib.md5(normalized_path.encode()).digest()
    return hash_bytes  # MD5 gives us exactly 16 bytes


def resolve_lazy_values(obj: Any) -> Any:
    """Recursively resolve WorkerEval-wrapped callables in nested structures.

    Only evaluates objects explicitly marked with WorkerEval. Other callables
    are passed through unchanged.

    Args:
        obj: Configuration object potentially containing WorkerEval instances

    Returns:
        Configuration with WorkerEval instances evaluated to their values

    Examples:
        >>> from intccms.schema import WorkerEval
        >>> import os
        >>> config = {
        ...     "key": WorkerEval(lambda: os.environ['AWS_KEY']),
        ...     "compression": my_compressor_func,  # Not evaluated
        ... }
        >>> resolved = resolve_lazy_values(config)
    """
    if isinstance(obj, WorkerEval):
        return obj()
    elif isinstance(obj, dict):
        return {k: resolve_lazy_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_lazy_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(resolve_lazy_values(item) for item in obj)
    else:
        # Pass through everything else unchanged (including callables)
        return obj


def build_output_path(
    workitem: WorkItem,
    fmt: str,
) -> Tuple[str, Dict[str, Any]]:
    """Build unique output path using hash of workitem metadata.

    Creates deterministic paths based on source filename and entry range,
    with a short hash for uniqueness. Also returns metadata for manifest.

    The file extension is handled automatically by the writer, so this function
    creates paths without extensions.

    Args:
        workitem: coffea WorkItem with file metadata and entry ranges
        fmt: Output format (parquet, root_ttree) - used for validation only

    Returns:
        Tuple of (relative_path, metadata_dict)
        - relative_path: Path string like "dataset/abc12345_0_1000" (no extension)
        - metadata_dict: Manifest entry with source file info
    """
    # Create short hash from filename for stable file identifier
    file_hash = hashlib.md5(workitem.filename.encode()).hexdigest()[:8]

    # Build path with hash and entry range (no extension - writer will add it)
    filename = f"{file_hash}_{workitem.entrystart}_{workitem.entrystop}"
    path = f"{workitem.dataset}/{filename}"

    # Create metadata for manifest
    metadata = {
        "source_file": workitem.filename,
        "entrystart": workitem.entrystart,
        "entrystop": workitem.entrystop,
        "dataset": workitem.dataset,
        "treename": workitem.treename,
    }

    return path, metadata


def resolve_output_path(
    workitem: WorkItem,
    output_cfg: Any,  # SkimOutputConfig
    output_manager: Any,  # OutputDirectoryManager
) -> Tuple[str, bool, Dict[str, Any]]:
    """Resolve output path/URI for a skimmed workitem file.

    Uses hash-based naming for unique, deterministic paths without counters.

    Args:
        workitem: coffea WorkItem with file metadata and entry ranges
        output_cfg: SkimOutputConfig with format and URI settings
        output_manager: OutputDirectoryManager for directory resolution

    Returns:
        Tuple of (path_string, is_local_path, metadata_dict)
        - path_string: Full path or URI to output file
        - is_local_path: True if local filesystem, False for remote
        - metadata_dict: Manifest entry for this workitem
    """
    # Build path and get metadata
    relative_path, metadata = build_output_path(workitem, output_cfg.format)

    # Determine if local or remote
    if output_cfg.local or not output_cfg.base_uri:
        base_dir = Path(output_manager.skimmed_dir)
        return str(base_dir / relative_path), True, metadata
    else:
        base_uri = (output_cfg.base_uri or "").rstrip("/")
        if base_uri == "":
            base_dir = Path(output_manager.skimmed_dir)
            return str(base_dir / relative_path), output_cfg.local, metadata
        else:
            path = f"{base_uri}/{relative_path}"
            return path, output_cfg.local, metadata


def process_workitem(
    workitem: WorkItem,
    config: SkimmingConfig,
    configuration: Any,
    output_manager: Any,
    is_mc: bool = True,
) -> Dict[str, Any]:
    """Process a single WorkItem: load, filter, extract, and save events.

    Args:
        workitem: coffea WorkItem containing file metadata and entry ranges
        config: Skimming configuration with selection functions and output settings
        configuration: Main analysis configuration with branch selections
        output_manager: OutputDirectoryManager
        is_mc: Whether the workitem represents Monte Carlo data

    Returns:
        Dictionary containing:
            - 'hist': Dummy histogram for success tracking
            - 'failed_items': Set of failed workitems (empty on success)
            - 'processed_events': Number of events processed
            - 'output_files': List of created output files
            - 'manifest_entries': List of manifest metadata dictionaries
    """
    dummy_hist = default_histogram()

    try:
        output_files: List[str] = []

        # Extract workitem metadata
        filename = workitem.filename
        treename = workitem.treename
        entry_start = workitem.entrystart
        entry_stop = workitem.entrystop
        dataset = workitem.dataset

        # Stage 1: Load events
        reader = get_reader("root")
        events = load_events(
            reader,
            filename,
            treename,
            entry_start=entry_start,
            entry_stop=entry_stop,
        )

        total_events = len(events)

        # Stage 2: Apply selection
        executor = SelectionExecutor(config)
        selection_mask = executor.execute(events)
        filtered_events = events[selection_mask]

        processed_events = len(filtered_events)

        # Fill dummy histogram for tracking
        if processed_events > 0:
            dummy_values = [500.0] * min(processed_events, 100)
            dummy_hist.fill(dummy_values)

        # Stage 3: Extract columns
        preprocess_cfg = configuration.preprocess
        columns_to_keep, mc_only_columns = build_column_list(
            preprocess_cfg.branches,
            preprocess_cfg.get("mc_branches"),
            is_data=not is_mc,
        )

        output_columns = extract_columns(
            filtered_events,
            columns_to_keep,
            mc_only_columns=mc_only_columns,
            is_data=not is_mc,
        )

        # Resolve output path and get manifest metadata
        output_path, like_local, manifest_metadata = resolve_output_path(
            workitem,
            config.output,
            output_manager,
        )

        # Stage 4: Save events
        writer = get_writer(config.output.format)

        # Resolve lazy values in writer kwargs
        writer_kwargs = resolve_lazy_values(config.output.to_kwargs or {})

        # Add tree_name for ROOT format
        if config.output.format == "root_ttree":
            writer_kwargs["tree_name"] = config.tree_name

        # Save events and get actual path used (with extension)
        actual_output_path = save_events(writer, output_columns, output_path, **writer_kwargs)
        output_files.append(actual_output_path)

        # Add output path to manifest metadata
        manifest_metadata["output_file"] = actual_output_path
        manifest_metadata["processed_events"] = processed_events
        manifest_metadata["total_events"] = total_events

        logger.info(
            f"Processed workitem: {dataset} | {filename} | "
            f"entries [{entry_start}:{entry_stop}] | "
            f"filtered {processed_events}/{total_events} events | "
            f"output: {actual_output_path}"
        )

        return {
            "hist": dummy_hist,
            "failed_items": set(),
            "processed_events": processed_events,
            "output_files": output_files,
            "manifest_entries": [manifest_metadata],
        }

    except Exception as exc:
        logger.error(f"Failed to process workitem {workitem}: {exc}")
        return {
            "hist": dummy_hist,
            "failed_items": {workitem},
            "processed_events": 0,
            "output_files": [],
            "manifest_entries": [],
        }
