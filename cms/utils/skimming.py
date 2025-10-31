"""
Event skimming and preprocessing for NanoAOD datasets.

Workflow:
1. **Skimming**: Process WorkItems in parallel (dask.bag), apply event selections,
   save filtered output to configured format. Retry failures.
2. **Discovery**: Scan output directory for previously skimmed files.
3. **Loading & Merging**: Load output files, merge per dataset into single arrays,
   cache results for fast subsequent runs.
4. **Metadata**: Attach cross-sections, luminosity, nevts to Dataset objects.

Output Structure: {output_dir}/{dataset}/file_{N}/part_{M}.{ext}

Key Components:
    WorkitemSkimmingManager: Orchestrates parallel skimming with retries
    process_workitem: Processes individual file chunks
    process_and_load_events: Main entry point for full workflow
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import awkward as ak
import cloudpickle
import dask
import dask.bag
import hist
import uproot
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.processor.executor import WorkItem
from tabulate import tabulate

from utils.schema import SkimmingConfig, SkimOutputConfig
from utils.tools import get_function_arguments
from utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
NanoAODSchema.warn_missing_crossrefs = False

# Counter key delimiters for workitem file/part numbering
# Format: "{dataset}::{filename}" for files, "{file_key}::{start}_{stop}" for parts
COUNTER_DELIMITER = "::"
ENTRY_RANGE_DELIMITER = "_"


# =============================================================================
# Public Utilities
# =============================================================================

def default_histogram() -> hist.Hist:
    """
    Create a default histogram for tracking processing success/failure.

    This histogram serves as a dummy placeholder to track whether workitems
    were processed successfully. The actual analysis histograms are created
    separately during the analysis phase.

    Returns
    -------
    hist.Hist
        A simple histogram with regular binning for tracking purposes
    """
    return hist.Hist.new.Regular(10, 0, 1000).Weight()


def get_dataset_config_by_name(dataset_name: str, configuration: Any):
    """
    Retrieve the dataset configuration matching the provided dataset name.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier.
    configuration : Any
        Full analysis configuration containing dataset definitions.

    Returns
    -------
    DatasetConfig or None
        Matching dataset configuration if found, else None.
    """
    dataset_manager = getattr(configuration, "datasets", None)
    dataset_configs = getattr(dataset_manager, "datasets", []) if dataset_manager else []

    for dataset_config in dataset_configs:
        if dataset_config.name == dataset_name:
            return dataset_config
    return None


# =============================================================================
# Branch and Column Management Helpers
# =============================================================================

def _build_branches_to_keep(
    configuration: Any, is_mc: bool
) -> Dict[str, List[str]]:
    """
    Build dictionary of branches to keep based on configuration.

    This replicates the logic from the existing skimming code to determine
    which branches should be saved in the output files.

    Parameters
    ----------
    configuration : Any
        Main analysis configuration
    is_mc : bool
        Whether this is Monte Carlo data

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping object names to lists of branch names
    """
    branches = configuration.preprocess.branches
    mc_branches = configuration.preprocess.mc_branches

    filtered = {}
    for obj, obj_branches in branches.items():
        if not is_mc:
            # For data, exclude MC-only branches
            filtered[obj] = [
                br for br in obj_branches
                if br not in mc_branches.get(obj, [])
            ]
        else:
            # For MC, keep all branches
            filtered[obj] = obj_branches

    return filtered


def _extract_output_columns(
    events: Any,
    branches_to_keep: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Extract the branch data that should be persisted."""
    output_data: Dict[str, Any] = {}
    for obj, obj_branches in branches_to_keep.items():
        if obj == "event":
            for branch in obj_branches:
                if hasattr(events, branch):
                    output_data[branch] = getattr(events, branch)
        else:
            if hasattr(events, obj):
                obj_collection = getattr(events, obj)
                for branch in obj_branches:
                    if hasattr(obj_collection, branch):
                        output_data[f"{obj}_{branch}"] = getattr(
                            obj_collection, branch
                        )

    return output_data


def _build_parquet_payload(
    output_columns: Dict[str, Any]
) -> Optional[ak.Array]:
    """Construct the awkward payload used for parquet serialization."""
    if not output_columns:
        return None

    return ak.zip(output_columns, depth_limit=1)


# =============================================================================
# Path and Output Management Helpers
# =============================================================================

def _build_output_path(
    dataset: str,
    file_index: int,
    part_index: int,
    fmt: str,
) -> str:
    """
    Build relative output path for a skimmed file chunk.

    Constructs standardized directory structure: {dataset}/file_{N}/part_{M}.{ext}
    where N is the file index and M is the part (chunk) index within that file.
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


def _resolve_output_path(
    workitem: WorkItem,
    output_cfg: SkimOutputConfig,
    output_manager,
    file_counters: Dict[str, int],
    part_counters: Dict[str, int],
) -> Tuple[Union[Path, str], bool]:
    """
    Resolve output path/URI for a skimmed workitem file.

    Constructs hierarchical keys for file/part lookups in counter dictionaries.
    Returns either local Path or remote URI string based on protocol configuration.
    """
    dataset = workitem.dataset
    # Build hierarchical keys: file_key identifies source file, part_key identifies chunk
    file_key = f"{dataset}{COUNTER_DELIMITER}{workitem.filename}"
    file_index = file_counters[file_key]
    part_key = f"{file_key}{COUNTER_DELIMITER}{workitem.entrystart}{ENTRY_RANGE_DELIMITER}{workitem.entrystop}"
    part_index = part_counters[part_key]

    suffix = _build_output_path(dataset, file_index, part_index, output_cfg.format)

    if output_cfg.protocol == "local" or not output_cfg.base_uri:
        base_dir = Path(output_manager.get_skimmed_dir())
        return base_dir / suffix, True
    else:
        base_uri = (output_cfg.base_uri or "").rstrip("/")
        path = f"{base_uri}/{suffix}"
        return path, output_cfg.protocol == "local"


# =============================================================================
# Output Persistence
# =============================================================================

def _resolve_lazy_values(obj):
    """
    Recursively resolve WorkerEval-wrapped callables in nested dicts/lists.

    Only evaluates objects explicitly marked with WorkerEval. Other callables
    are passed through unchanged, allowing writer functions to receive callable
    arguments when needed.

    This enables configuration values to be computed on workers (e.g., reading
    environment variables that exist on workers but not client) while preserving
    legitimate callable arguments.

    Parameters
    ----------
    obj : Any
        Configuration object potentially containing WorkerEval instances

    Returns
    -------
    Any
        Configuration with WorkerEval instances evaluated to their values

    Examples
    --------
    >>> from utils.schema import WorkerEval
    >>> import os
    >>> config = {
    ...     "key": WorkerEval(lambda: os.environ['AWS_KEY']),
    ...     "compression": my_compressor_func,  # Not evaluated
    ... }
    >>> resolved = _resolve_lazy_values(config)
    # config["key"] is now the env var value
    # config["compression"] is still my_compressor_func
    """
    from utils.schema import WorkerEval

    if isinstance(obj, WorkerEval):
        return obj()
    elif isinstance(obj, dict):
        return {k: _resolve_lazy_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_lazy_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_resolve_lazy_values(item) for item in obj)
    else:
        # Pass through everything else unchanged (including callables)
        return obj


def _save_workitem_output(
    events: Any,
    output_file: Union[Path, str],
    config: SkimmingConfig,
    configuration: Any,
    is_mc: bool,
) -> None:
    """
    Persist filtered events according to the configured skim output.

    Parameters
    ----------
    events : Any
        Filtered events to save.
    output_file : Path or str
        Destination path URI.
    config : SkimmingConfig
        Skimming configuration.
    configuration : Any
        Main analysis configuration with branch selections.
    is_mc : bool
        Whether this dataset represents Monte Carlo data.
    """
    output_cfg = config.output
    path_str = str(output_file)

    # Resolve any lazy callables (WorkerEval) on worker before passing to writer
    writer_kwargs = _resolve_lazy_values(output_cfg.to_kwargs or {})

    branches_to_keep = _build_branches_to_keep(configuration, is_mc)
    output_columns = _extract_output_columns(events, branches_to_keep)

    if output_cfg.format == "parquet":
        payload = _build_parquet_payload(output_columns)
        if payload is None:
            logger.warning("No branches extracted for parquet output; skipping write.")
            return
        ak.to_parquet(payload, path_str, **writer_kwargs)
    elif output_cfg.format == "root_ttree":
        _write_root_ttree(output_columns, path_str, config.tree_name, writer_kwargs)
    elif output_cfg.format == "rntuple":
        raise NotImplementedError("RNTuple output is not yet implemented.")
    elif output_cfg.format == "safetensors":
        raise NotImplementedError("safetensors output is not yet implemented.")
    else:
        raise ValueError(f"Unsupported skim output format: {output_cfg.format}")


def _write_root_ttree(
    output_columns: Dict[str, Any],
    output_path: str,
    tree_name: str,
    writer_kwargs: Optional[Dict[str, Any]],
) -> None:
    """
    Persist skimmed events to a ROOT TTree using uproot.

    Parameters
    ----------
    output_columns : Dict[str, Any]
        Mapping of branch names to awkward arrays.
    output_path : str
        Destination ROOT file path.
    tree_name : str
        Name of the output TTree.
    writer_kwargs : Optional[Dict[str, Any]]
        Keyword arguments forwarded to uproot.recreate. Unsupported options
        specific to tree/branch creation are currently ignored.
    """
    if not output_columns:
        logger.warning("No branches extracted for ROOT output; skipping write.")
        return

    # Materialize arrays to ensure uproot receives concrete data buffers.
    file_kwargs: Dict[str, Any] = dict(writer_kwargs or {})
    tree_kwargs: Dict[str, Any] = file_kwargs.pop("tree_kwargs", {})

    branch_types = {k: v.type for k, v in output_columns.items()}

    with uproot.recreate(output_path, **file_kwargs) as root_file:
        tree = root_file.mktree(tree_name, branch_types, **tree_kwargs)
        tree.extend(output_columns)


def _load_skimmed_events(
    file_path: Union[str, Path],
    config: SkimmingConfig,
    reader_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Load events from a single skimmed file using the configured format.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the skimmed output file.
    config : SkimmingConfig
        Skimming configuration describing output format and reader kwargs.

    Returns
    -------
    Any
        Events object produced by the appropriate NanoEventsFactory reader.
    """
    output_cfg = config.output
    kwargs = dict(reader_kwargs or {})
    fmt = output_cfg.format

    if fmt == "parquet":
        kwargs.setdefault("schemaclass", NanoAODSchema)
        reader = NanoEventsFactory.from_parquet(str(file_path), **kwargs)
        return reader.events()
    elif fmt == "root_ttree":
        kwargs.setdefault("schemaclass", NanoAODSchema)
        tree_name = kwargs.pop("treename", config.tree_name)
        reader = NanoEventsFactory.from_root(
            {str(file_path): tree_name},
            **kwargs,
        )
        return reader.events()
    elif fmt == "rntuple":
        raise NotImplementedError("Reading skim output format 'rntuple' is not yet implemented.")
    elif fmt == "safetensors":
        raise NotImplementedError("Reading skim output format 'safetensors' is not yet implemented.")
    else:
        raise ValueError(f"Unsupported skim output format: {fmt}")


# =============================================================================
# Core Workitem Processing
# =============================================================================

def process_workitem(
    workitem: WorkItem,
    config: SkimmingConfig,
    configuration: Any,
    output_manager,
    file_counters: Dict[str, int],
    part_counters: Dict[str, int],
    is_mc: bool = True,
) -> Dict[str, Any]:
    """
    Load events from a WorkItem, apply event selection, and save filtered output.

    Processes a single file chunk (WorkItem) by loading events from ROOT using
    NanoEventsFactory, applying configured selection function to filter events,
    extracting specified branches, and persisting filtered output. Returns success
    or failure information for retry logic.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier.
    configuration : Any
        Full analysis configuration containing dataset definitions.

    Returns
    -------
    DatasetConfig or None
        Matching dataset configuration if found, else None.
    """
    dataset_manager = getattr(configuration, "datasets", None)
    dataset_configs = getattr(dataset_manager, "datasets", []) if dataset_manager else []

    try:
        output_files: List[str] = []
        # Extract workitem metadata
        filename = workitem.filename
        treename = workitem.treename
        entry_start = workitem.entrystart
        entry_stop = workitem.entrystop
        dataset = workitem.dataset

        # Load events using NanoEventsFactory
        events = NanoEventsFactory.from_root(
            {filename: treename},
            entry_start=entry_start,
            entry_stop=entry_stop,
            schemaclass=NanoAODSchema,
        ).events()

        total_events = len(events)

        # Apply skimming selection using the provided function
        selection_func = config.function
        selection_use = config.use

        # Get function arguments using existing utility
        selection_args, selection_static_kwargs = get_function_arguments(
            selection_use,
            events,
            function_name=selection_func.__name__,
            static_kwargs=config.get("static_kwargs"),
        )
        packed_selection = selection_func(
            *selection_args, **selection_static_kwargs
        )

        # Apply final selection mask
        selection_names = packed_selection.names
        if selection_names:
            final_selection = selection_names[-1]
            mask = packed_selection.all(final_selection)
        else:
            # No selection applied, keep all events
            mask = slice(None)

        filtered_events = events[mask]
        processed_events = len(filtered_events)

        # Fill dummy histogram with some dummy values for tracking
        if processed_events > 0:
            # Use a simple observable for the dummy histogram
            dummy_values = [500.0] * min(processed_events, 100)
            dummy_hist.fill(dummy_values)

        output_path, is_local = _resolve_output_path(
            workitem,
            config.output,
            output_manager,
            file_counters,
            part_counters,
        )
        if is_local:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_workitem_output(
            filtered_events, output_path, config, configuration, is_mc
        )
        output_files.append(str(output_path))

        return {
            "hist": dummy_hist,
            "failed_items": set(),
            "processed_events": processed_events,
            "output_files": output_files,
            "failure_info": None,
        }

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"Failed to process workitem {workitem.filename}: {error_type}: {error_msg}")
        return {
            "hist": default_histogram(),
            "failed_items": {workitem},  # Track failure
            "processed_events": 0,
            "output_files": [],
            "failure_info": {
                "workitem": workitem,
                "error_type": error_type,
                "error_msg": error_msg,
                "dataset": workitem.dataset,
                "filename": workitem.filename,
            }
        }

def merge_results(
    result_a: Dict[str, Any], result_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two result dictionaries from parallel workitem processing.

    Used by dask.bag.fold() to combine results from parallel workers. Adds
    histograms, unions failed item sets, concatenates output file lists, and
    accumulates failure information for detailed error reporting.

    Parameters
    ----------
    result_a : Dict[str, Any]
        First result dictionary
    result_b : Dict[str, Any]
        Second result dictionary

    Returns
    -------
    Dict[str, Any]
        Combined result dictionary
    """
    # Collect failure info from both results
    failure_infos = []
    if result_a.get("failure_info"):
        failure_infos.append(result_a["failure_info"])
    if result_b.get("failure_info"):
        failure_infos.append(result_b["failure_info"])

    # If we have multiple failures, accumulate them as a list
    if "failure_infos" in result_a:
        failure_infos.extend(result_a["failure_infos"])
    if "failure_infos" in result_b:
        failure_infos.extend(result_b["failure_infos"])

    return {
        "hist": result_a["hist"] + result_b["hist"],
        "failed_items": result_a["failed_items"] | result_b["failed_items"],
        "processed_events": result_a["processed_events"] + result_b["processed_events"],
        "output_files": result_a["output_files"] + result_b["output_files"],
        "failure_infos": failure_infos,
    }

# =============================================================================
# Workitem Skimming Manager
# =============================================================================

class WorkitemSkimmingManager:
    """
    Manager for workitem-based skimming using dask.bag processing.

    This class orchestrates the new preprocessing workflow that processes
    workitems directly using dask.bag, providing robust failure handling
    and retry mechanisms.

    Attributes
    ----------
    config : SkimmingConfig
        Skimming configuration with selection functions and output settings
    output_manager : OutputDirectoryManager
        Centralized output directory manager
    """

    def __init__(self, config: SkimmingConfig, output_manager):
        """
        Initialize the workitem skimming manager.

        Parameters
        ----------
        config : SkimmingConfig
            Skimming configuration with selection functions and output settings
        output_manager : OutputDirectoryManager
            Centralized output directory manager
        """
        self.config = config
        self.output_manager = output_manager
        logger.info("Initialized workitem-based skimming manager")

    def process_workitems(
        self,
        workitems: List[WorkItem],
        configuration: Any,
        datasets: List,
        split_every: int = 4,
    ) -> Dict[str, Any]:
        """
        Process a list of workitems using dask.bag with failure handling.

        This is the main entry point that implements the dask.bag workflow
        with retry logic for failed workitems.

        Parameters
        ----------
        workitems : List[WorkItem]
            List of workitems to process
        configuration : Any
            Main analysis configuration object
        datasets : List
            Dataset objects with metadata (including is_data flag)
        split_every : int, default 4
            Split parameter for dask.bag.fold operation

        Returns
        -------
        Dict[str, Any]
            Final combined results with histograms and processing statistics
        """
        max_retries = self.config.max_retries
        logger.info(f"Processing {len(workitems)} workitems with max {max_retries} retries")

        # Pre-compute file and part counters for all workitems
        file_counters, part_counters = self._compute_counters(workitems)

        dataset_lookup: Dict[str, Any] = {}
        for dataset in datasets or []:
            for fileset_key in dataset.fileset_keys:
                dataset_lookup[fileset_key] = dataset

        # Initialize accumulator for successful results
        full_result = {
            "hist": default_histogram(),
            "failed_items": set(),
            "processed_events": 0,
            "output_files": [],
            "failure_infos": [],
        }

        # Process workitems with retry logic
        remaining_workitems = workitems.copy()
        retry_count = 0

        while remaining_workitems and retry_count < max_retries:
            logger.info(
                f"Attempt {retry_count + 1}: processing "
                f"{len(remaining_workitems)} workitems"
            )

            # Create dask bag from remaining workitems
            bag = dask.bag.from_sequence(remaining_workitems)

            # Map processing function over workitems
            futures = bag.map(
                lambda wi: process_workitem(
                    wi,
                    self.config,
                    configuration,
                    self.output_manager,
                    file_counters,
                    part_counters,
                    is_mc=not (
                        (dataset := dataset_lookup.get(wi.dataset))
                        and dataset.is_data
                    ),
                )
            )

            # Reduce results using fold operation
            task = futures.fold(merge_results, split_every=split_every)

            # Compute results
            (result,) = dask.compute(task)

            # Update remaining workitems to failed ones
            remaining_workitems = list(result["failed_items"])

            # Accumulate successful results
            if result["processed_events"] > 0:
                full_result["hist"] += result["hist"]
                full_result["processed_events"] += result["processed_events"]
                full_result["output_files"].extend(result["output_files"])

            # Accumulate failure information
            if result.get("failure_infos"):
                full_result["failure_infos"].extend(result["failure_infos"])

            # Log progress
            failed_count = len(remaining_workitems)
            successful_count = len(workitems) - failed_count
            logger.info(
                f"Attempt {retry_count + 1} complete: "
                f"{successful_count} successful, {failed_count} failed"
            )

            # Show detailed failure summary after each attempt if there are failures
            if remaining_workitems and result.get("failure_infos"):
                logger.warning(f"\n=== Failures in Attempt {retry_count + 1} ===")
                self._log_failure_summary(workitems, result["failure_infos"])

            retry_count += 1

            retry_count += 1
        
        # Final logging
        if remaining_workitems:
            logger.warning(
                f"Failed to process {len(remaining_workitems)} workitems "
                f"after {max_retries} attempts"
            )
            full_result["failed_items"] = set(remaining_workitems)
            # Log final cumulative failure information
            logger.warning(f"\n=== Final Failure Summary (All Attempts) ===")
            self._log_failure_summary(workitems, full_result["failure_infos"])
        else:
            logger.info("All workitems processed successfully")

        # Create summary statistics by dataset
        self._log_processing_summary(workitems, full_result["output_files"], full_result["processed_events"])

        return full_result

    def discover_workitem_outputs(self, workitems: List[WorkItem]) -> List[str]:
        """
        Discover existing output files from previous workitem processing.

        This method scans for output files that would be created by the
        workitem processing, allowing for resumption of interrupted workflows.

        Parameters
        ----------
        workitems : List[WorkItem]
            List of workitems to check for existing outputs

        Returns
        -------
        List[str]
            List of existing output file paths
        """
        output_files = []
        dataset_counts = {}

        # Use the same counter computation as processing
        file_counters, part_counters = self._compute_counters(workitems)

        for workitem in workitems:
            resolved_path, is_local = _resolve_output_path(
                workitem,
                self.config.output,
                self.output_manager,
                file_counters,
                part_counters,
            )

            if not is_local:
                continue

            path_obj = Path(resolved_path)
            if path_obj.exists():
                output_files.append(str(path_obj))

                dataset = workitem.dataset
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        # Log with dataset breakdown
        if dataset_counts:
            dataset_info = ", ".join([
                f"{dataset}: {count}" for dataset, count in dataset_counts.items()
            ])
            logger.info(
                f"Found existing skimmed files for {dataset_info}"
            )
        else:
            logger.info("No existing output files found")

        return output_files

    def _log_processing_summary(
        self, workitems: List[WorkItem], output_files: List[str], total_events: int
    ) -> None:
        """
        Log a summary table of processing results by dataset.

        Parameters
        ----------
        workitems : List[WorkItem]
            Original list of workitems processed
        output_files : List[str]
            List of output files created
        total_events : int
            Total number of events processed
        """
        # Collect statistics by dataset
        dataset_stats = defaultdict(lambda: {"files_written": 0})

        # Count files written per dataset
        for output_file in output_files:
            try:
                # Extract dataset name from output path structure
                # IMPORTANT: Assumes path format: .../output_dir/{dataset}/file_{N}/part_{M}.ext
                # path_parts[-3] gets dataset from this fixed structure
                # If _build_output_path() changes, this logic must be updated
                path_parts = Path(output_file).parts
                if len(path_parts) >= 3:
                    dataset = path_parts[-3]
                    dataset_stats[dataset]["files_written"] += 1
            except Exception:
                pass

        # Create summary table
        if dataset_stats:
            table_data = []
            total_files = 0

            for dataset, stats in sorted(dataset_stats.items()):
                files = stats["files_written"]
                table_data.append([dataset, files])
                total_files += files

            # Add totals row
            table_data.append(["TOTAL", total_files])

            # Create and log table
            headers = ["Dataset", "Files Written"]
            table = tabulate(table_data, headers=headers, tablefmt="grid")

            logger.info(f"Processing Summary: {total_events:,} total events saved")
            logger.info(f"\n{table}")
        else:
            logger.info("No output files were created during processing")

    def _log_failure_summary(
        self, workitems: List[WorkItem], failure_infos: List[Dict[str, Any]]
    ) -> None:
        """
        Log a detailed summary of failures by dataset, including percentages and error types.

        Parameters
        ----------
        workitems : List[WorkItem]
            Original list of workitems processed
        failure_infos : List[Dict[str, Any]]
            List of failure information dictionaries
        """
        if not failure_infos:
            return

        # Count total workitems per dataset
        total_per_dataset = defaultdict(int)
        for workitem in workitems:
            total_per_dataset[workitem.dataset] += 1

        # Organize failures by dataset
        failures_by_dataset = defaultdict(list)
        for failure in failure_infos:
            dataset = failure["dataset"]
            failures_by_dataset[dataset].append(failure)

        # Count error types per dataset
        error_types_by_dataset = defaultdict(lambda: defaultdict(int))
        for failure in failure_infos:
            dataset = failure["dataset"]
            error_type = failure["error_type"]
            error_types_by_dataset[dataset][error_type] += 1

        # Count global error types for overall statistics
        global_error_counts = defaultdict(int)
        for failure in failure_infos:
            global_error_counts[failure["error_type"]] += 1

        # Create summary table
        table_data = []
        total_failed = 0
        total_workitems = sum(total_per_dataset.values())

        for dataset, failures in sorted(failures_by_dataset.items()):
            failed_count = len(failures)
            total_count = total_per_dataset[dataset]
            percentage = (failed_count / total_count * 100) if total_count > 0 else 0

            # Build error type breakdown string
            error_counts = error_types_by_dataset[dataset]
            error_breakdown = ", ".join([
                f"{error_type}: {count} ({count/failed_count*100:.0f}%)"
                for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1])
            ])

            table_data.append([
                dataset,
                failed_count,
                total_count,
                f"{percentage:.1f}%",
                error_breakdown
            ])
            total_failed += failed_count

        # Add totals row with global error breakdown
        total_percentage = (total_failed / total_workitems * 100) if total_workitems > 0 else 0
        global_error_breakdown = ", ".join([
            f"{error_type}: {count} ({count/total_failed*100:.0f}%)"
            for error_type, count in sorted(global_error_counts.items(), key=lambda x: -x[1])
        ])
        table_data.append([
            "TOTAL",
            total_failed,
            total_workitems,
            f"{total_percentage:.1f}%",
            global_error_breakdown
        ])

        # Create and log table
        headers = ["Dataset", "Failed", "Total", "Failure %", "Error Types (count, %)"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")

        logger.warning("Failure Summary:")
        logger.warning(f"\n{table}")

        # Log sample failures with file names for debugging
        logger.warning("\nSample failures (first 5):")
        for i, failure in enumerate(failure_infos[:5]):
            logger.warning(
                f"  {i+1}. [{failure['dataset']}] {failure['error_type']}: "
                f"{failure['error_msg'][:100]}... (file: {Path(failure['filename']).name})"
            )

        if len(failure_infos) > 5:
            logger.warning(f"  ... and {len(failure_infos) - 5} more failures")

    def _compute_counters(
        self, workitems: List[WorkItem]
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """
        Pre-compute file and part counters for all workitems.

        This ensures consistent numbering across all workers by computing
        the counters once before parallel processing begins.

        Parameters
        ----------
        workitems : List[WorkItem]
            List of all workitems to process

        Returns
        -------
        tuple[Dict[str, int], Dict[str, int]]
            File counters and part counters dictionaries
        """
        file_counters = {}
        part_counters = {}

        # Track unique files per dataset for sequential file numbering
        dataset_file_counts = {}

        for workitem in workitems:
            dataset = workitem.dataset
            file_key = f"{dataset}{COUNTER_DELIMITER}{workitem.filename}"
            part_key = f"{file_key}{COUNTER_DELIMITER}{workitem.entrystart}{ENTRY_RANGE_DELIMITER}{workitem.entrystop}"

            # Assign file number if not already assigned
            if file_key not in file_counters:
                if dataset not in dataset_file_counts:
                    dataset_file_counts[dataset] = 0
                file_counters[file_key] = dataset_file_counts[dataset]
                dataset_file_counts[dataset] += 1

            # Assign part number if not already assigned
            if part_key not in part_counters:
                # Count existing parts for this file
                existing_parts = [
                    k for k in part_counters.keys() if k.startswith(f"{file_key}{COUNTER_DELIMITER}")
                ]
                part_counters[part_key] = len(existing_parts)

        return file_counters, part_counters


# =============================================================================
# Main Entry Point
# =============================================================================

def process_and_load_events(
    workitems: List[WorkItem],
    config: Any,
    output_manager,
    datasets: List,
    nanoaods_summary: Optional[Dict[str, Any]] = None,
) -> List:
    """
    Run skimming workflow and load merged events into Dataset objects.

    Main entry point orchestrating: (1) skimming WorkItems in parallel if enabled,
    (2) discovering output files, (3) loading and merging events per dataset,
    (4) caching merged results, and (5) populating Dataset.events with event arrays
    and metadata. Events from multiple output files are merged into single arrays
    for performance.

    Parameters
    ----------
    workitems : List[WorkItem]
        List of workitems to process, typically from NanoAODMetadataGenerator.workitems
    config : Any
        Main analysis configuration object containing skimming and preprocessing settings
    output_manager : OutputDirectoryManager
        Centralized output directory manager
    datasets : List[Dataset]
        List of Dataset objects to populate with events
    nanoaods_summary : Optional[Dict[str, Any]], default None
        NanoAODs summary containing event counts per dataset for nevts metadata

    Returns
    -------
    List[Dataset]
        List of Dataset objects with populated events attribute.
        Each Dataset.events contains List[Tuple[events, metadata]] for each fileset_key.
    """
    logger.info(f"Starting workitem preprocessing with {len(workitems)} workitems")

    # Create workitem skimming manager
    skimming_config = config.preprocess.skimming
    skimming_manager = WorkitemSkimmingManager(skimming_config, output_manager)

    # Group workitems by dataset (fileset_key)
    workitems_by_dataset = {}
    for workitem in workitems:
        dataset = workitem.dataset
        if dataset not in workitems_by_dataset:
            workitems_by_dataset[dataset] = []
        workitems_by_dataset[dataset].append(workitem)

    # Process workitems if skimming is enabled
    if config.general.run_skimming:
        # Filter workitems by processes if configured
        workitems_to_process = workitems
        if hasattr(config.general, 'processes') and config.general.processes:
            fileset_key_to_dataset_lookup = {}
            for dataset in datasets:
                for fileset_key in dataset.fileset_keys:
                    fileset_key_to_dataset_lookup[fileset_key] = dataset

            workitems_to_process = [
                wi for wi in workitems
                if wi.dataset in fileset_key_to_dataset_lookup and
                fileset_key_to_dataset_lookup[wi.dataset].process in config.general.processes
            ]
            logger.info(f"Filtered {len(workitems)} workitems to {len(workitems_to_process)} based on processes filter")

        logger.info("Running skimming")
        results = skimming_manager.process_workitems(
            workitems_to_process,
            config,
            datasets,
        )
        logger.info(f"Skimming complete: {results['processed_events']:,} events")

    # Create a mapping from fileset_key to Dataset for quick lookup
    fileset_key_to_dataset = {}
    for dataset in datasets:
        for fileset_key in dataset.fileset_keys:
            fileset_key_to_dataset[fileset_key] = dataset

    # Always discover and read from saved files
    logger.info("Reading from saved files")

    # Initialize events list for each Dataset
    for dataset in datasets:
        dataset.events = []

    for fileset_key, dataset_workitems in workitems_by_dataset.items():
        # Get the Dataset object this fileset_key belongs to
        if fileset_key not in fileset_key_to_dataset:
            logger.warning(f"Fileset key '{fileset_key}' not found in any Dataset object, skipping")
            continue

        dataset_obj = fileset_key_to_dataset[fileset_key]

        # Skip datasets not explicitly requested in config
        if hasattr(config.general, 'processes') and config.general.processes:
            if dataset_obj.process not in config.general.processes:
                logger.info(f"Skipping {fileset_key} (process {dataset_obj.process} not requested)")
                continue

        # Discover output files for this fileset_key
        output_files = skimming_manager.discover_workitem_outputs(dataset_workitems)

        if output_files:
            # Get cross-section for this specific fileset_key
            # Find the index of this fileset_key in the Dataset's fileset_keys list
            try:
                idx = dataset_obj.fileset_keys.index(fileset_key)
                xsec = dataset_obj.cross_sections[idx]
            except (ValueError, IndexError) as e:
                logger.error(f"Failed to get cross-section for {fileset_key}: {e}")
                xsec = 1.0

            dataset_config = get_dataset_config_by_name(dataset_obj.process, config)
            lumi_mask_config = (
                dataset_config.lumi_mask
                if dataset_config and dataset_obj.is_data
                else None
            )

            # Create metadata
            metadata = {
                "dataset": fileset_key,
                "process": dataset_obj.process,
                "variation": dataset_obj.variation,
                "xsec": xsec,
                "is_data": dataset_obj.is_data,
                "lumi_mask_config": lumi_mask_config,
            }

            # Add nevts from NanoAODs summary if available
            # The analysis code expects 'nevts' field for normalization
            nevts = 0
            if nanoaods_summary:
                # Extract dataset name from fileset_key (format: "datasetname__variation")
                # This handles multi-directory datasets where dataset name includes
                # directory index (e.g., signal_0, signal_1)
                dataset_name_from_key = fileset_key.rsplit('__', 1)[0]  # Remove variation suffix
                if dataset_name_from_key in nanoaods_summary:
                    if dataset_obj.variation in nanoaods_summary[dataset_name_from_key]:
                        nevts = nanoaods_summary[dataset_name_from_key][dataset_obj.variation].get(
                            'nevts_total', 0
                        )

            metadata['nevts'] = nevts
            if nevts == 0:
                logger.warning(f"No nevts found for {fileset_key}, using 0")

            # Generate deterministic cache key from fileset and file list
            # Cache key = MD5("{fileset_key}::{file1}::{file2}::...")
            # Files are sorted to ensure same key regardless of discovery order.
            # If any output file changes or is regenerated, cache is invalidated.
            sorted_files = sorted(output_files)
            cache_input = f"{fileset_key}::{':'.join(sorted_files)}"
            cache_key = hashlib.md5(cache_input.encode()).hexdigest()
            cache_dir = output_manager.get_cache_dir()
            cache_file = cache_dir / f"{fileset_key}__{cache_key}.pkl"

            # Check if we should read from cache
            if config.general.read_from_cache and os.path.exists(cache_file):
                logger.info(f"Loading cached events for {fileset_key}")
                try:
                    with open(cache_file, "rb") as f:
                        merged_events = cloudpickle.load(f)
                    logger.info(f"Loaded {len(merged_events)} cached events")
                    dataset_obj.events.append((merged_events, metadata.copy()))
                    continue  # Skip to next fileset_key
                except Exception as e:
                    logger.error(f"Failed to load cached events for {fileset_key}: {e}")
                    # Fall back to loading from files

            # Load and merge events from all discovered files
            all_events = []
            total_events_loaded = 0

            # Resolve any lazy callables (WorkerEval) before passing to reader
            reader_kwargs = _resolve_lazy_values(skimming_config.output.from_kwargs or {})
            for file_path in output_files:
                try:
                    events = _load_skimmed_events(
                        file_path,
                        skimming_config,
                        reader_kwargs,
                    )
                    all_events.append(events)
                    total_events_loaded += len(events)
                except Exception as e:
                    logger.error(f"Failed to load events from {file_path}: {e}")
                    continue

            # Merge all events into a single array if we have any events
            if all_events:
                try:
                    if len(all_events) == 1:
                        # Single file, no need to concatenate
                        merged_events = all_events[0]
                    else:
                        # Multiple files, concatenate them
                        merged_events = ak.concatenate(all_events, axis=0)

                    logger.info(
                        f"Merged {len(output_files)} files → "
                        f"{len(merged_events)} events for {fileset_key}"
                    )

                    # Cache the merged events
                    try:
                        with open(cache_file, "wb") as f:
                            cloudpickle.dump(merged_events, f)
                        logger.info(f"Cached events for {fileset_key}")
                    except Exception as e:
                        logger.warning(f"Failed to cache events for {fileset_key}: {e}")

                    dataset_obj.events.append((merged_events, metadata.copy()))

                except Exception as e:
                    logger.error(f"Failed to merge events for {fileset_key}: {e}")
                    # Fallback to individual events if merging fails
                    for i, events in enumerate(all_events):
                        dataset_obj.events.append((events, metadata.copy()))
        else:
            logger.warning(f"No output files found for {fileset_key}")

    return datasets
