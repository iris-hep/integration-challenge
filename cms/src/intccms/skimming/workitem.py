"""WorkItem processing for parallel skimming operations.

This module handles processing of individual coffea WorkItems, including output path
resolution, counter management, and integration with the pipeline stages.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional, List, Callable
from coffea.processor.executor import WorkItem

from intccms.utils.schema import WorkerEval, SkimmingConfig
from intccms.utils.tools import get_function_arguments
from intccms.skimming.io.readers import get_reader
from intccms.skimming.io.writers import get_writer
from intccms.skimming.pipeline.stages import (
    load_events,
    apply_selection,
    extract_columns,
    save_events,
    build_column_list,
)

logger = logging.getLogger(__name__)

# Counter delimiters for hierarchical keys
COUNTER_DELIMITER = "::"
ENTRY_RANGE_DELIMITER = "_"


def resolve_lazy_values(obj: Any) -> Any:
    """Recursively resolve WorkerEval-wrapped callables in nested structures.

    Only evaluates objects explicitly marked with WorkerEval. Other callables
    are passed through unchanged.

    Args:
        obj: Configuration object potentially containing WorkerEval instances

    Returns:
        Configuration with WorkerEval instances evaluated to their values

    Examples:
        >>> from intccms.utils.schema import WorkerEval
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
    dataset: str,
    file_index: int,
    part_index: int,
    fmt: str,
) -> str:
    """Build relative output path for a skimmed file chunk.

    Constructs standardized directory structure: {dataset}/file_{N}/part_{M}.{ext}

    Args:
        dataset: Dataset name
        file_index: File number within dataset
        part_index: Part (chunk) number within file
        fmt: Output format (parquet, root_ttree)

    Returns:
        Relative path string

    Raises:
        ValueError: If format is unsupported
    """
    extension_map = {
        "parquet": ".parquet",
        "root_ttree": ".root",
        "rntuple": ".ntuple",
        "safetensors": ".safetensors",
    }
    try:
        extension = extension_map[fmt]
    except KeyError as exc:
        raise ValueError(f"Unsupported output format '{fmt}'.") from exc

    return f"{dataset}/file_{file_index}/part_{part_index}{extension}"


def resolve_output_path(
    workitem: WorkItem,
    output_cfg: Any,  # SkimOutputConfig
    output_manager: Any,  # OutputDirectoryManager
    file_counters: Dict[str, int],
    part_counters: Dict[str, int],
) -> Tuple[str, bool]:
    """Resolve output path/URI for a skimmed workitem file.

    Constructs hierarchical keys for file/part lookups in counter dictionaries.

    Args:
        workitem: coffea WorkItem with file metadata and entry ranges
        output_cfg: SkimOutputConfig with format and URI settings
        output_manager: OutputDirectoryManager for directory resolution
        file_counters: Mapping of file keys to file numbers
        part_counters: Mapping of part keys to part numbers

    Returns:
        Tuple of (path_string, is_local_path)
    """
    dataset = workitem.dataset

    # Build hierarchical keys
    file_key = f"{dataset}{COUNTER_DELIMITER}{workitem.filename}"
    file_index = file_counters[file_key]

    part_key = (
        f"{file_key}{COUNTER_DELIMITER}"
        f"{workitem.entrystart}{ENTRY_RANGE_DELIMITER}{workitem.entrystop}"
    )
    part_index = part_counters[part_key]

    suffix = build_output_path(dataset, file_index, part_index, output_cfg.format)

    # Determine if local or remote
    if output_cfg.local or not output_cfg.base_uri:
        base_dir = Path(output_manager.get_skimmed_dir())
        return str(base_dir / suffix), True
    else:
        base_uri = (output_cfg.base_uri or "").rstrip("/")
        if base_uri == "":
            base_dir = Path(output_manager.get_skimmed_dir())
            return str(base_dir / suffix), output_cfg.local
        else:
            path = f"{base_uri}/{suffix}"
            return path, output_cfg.local


def process_workitem(
    workitem: WorkItem,
    config: SkimmingConfig,
    configuration: Any,
    output_manager: Any,
    file_counters: Dict[str, int],
    part_counters: Dict[str, int],
    is_mc: bool = True,
) -> Dict[str, Any]:
    """Process a single WorkItem: load, filter, extract, and save events.

    Args:
        workitem: coffea WorkItem containing file metadata and entry ranges
        config: Skimming configuration with selection functions and output settings
        configuration: Main analysis configuration with branch selections
        output_manager: OutputDirectoryManager
        file_counters: Pre-computed mapping of file keys to file numbers
        part_counters: Pre-computed mapping of part keys to part numbers
        is_mc: Whether the workitem represents Monte Carlo data

    Returns:
        Dictionary containing:
            - 'hist': Dummy histogram for success tracking
            - 'failed_items': Set of failed workitems (empty on success)
            - 'processed_events': Number of events processed
            - 'output_files': List of created output files
    """
    from intccms.utils.schema import default_histogram

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
        selection_func = config.function
        selection_use = config.use

        # Get function arguments
        selection_args, selection_static_kwargs = get_function_arguments(
            selection_use,
            events,
            function_name=selection_func.__name__,
            static_kwargs=config.get("static_kwargs"),
        )

        filtered_events = apply_selection(
            events,
            selection_func,
            selection_args,
            selection_static_kwargs,
        )

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

        # Resolve output path
        output_path, like_local = resolve_output_path(
            workitem,
            config.output,
            output_manager,
            file_counters,
            part_counters,
        )

        # Stage 4: Save events
        writer = get_writer(config.output.format)

        # Resolve lazy values in writer kwargs
        writer_kwargs = resolve_lazy_values(config.output.to_kwargs or {})

        # Add tree_name for ROOT format
        if config.output.format == "root_ttree":
            writer_kwargs["tree_name"] = config.tree_name

        save_events(writer, output_columns, output_path, **writer_kwargs)
        output_files.append(output_path)

        logger.info(
            f"Processed workitem: {dataset} | {filename} | "
            f"entries [{entry_start}:{entry_stop}] | "
            f"filtered {processed_events}/{total_events} events | "
            f"output: {output_path}"
        )

        return {
            "hist": dummy_hist,
            "failed_items": set(),
            "processed_events": processed_events,
            "output_files": output_files,
        }

    except Exception as exc:
        logger.error(f"Failed to process workitem {workitem}: {exc}")
        return {
            "hist": dummy_hist,
            "failed_items": {workitem},
            "processed_events": 0,
            "output_files": [],
        }
