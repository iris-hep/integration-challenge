"""Core data transformation functions for metadata extraction.

This module contains pure functions for dataset key parsing, fileset construction,
and event aggregation. All functions are stateless and have no side effects,
making them easily testable.
"""

from typing import Any, Dict, List, Tuple
from collections import defaultdict
import dataclasses
import logging

from coffea.processor.executor import WorkItem

logger = logging.getLogger(__name__)

# Dataset key format constants
DATASET_DELIMITER = "__"
DEFAULT_VARIATION = "nominal"


def parse_dataset_key(dataset_key: str) -> Tuple[str, str]:
    """
    Parse a dataset key string into process and variation components.

    Dataset keys encode both the physics process and systematic variation in a
    single string using DATASET_DELIMITER ("__"). This function splits them
    for separate access. If no delimiter is found, assumes nominal variation.

    Parameters
    ----------
    dataset_key : str
        Dataset key in format "process__variation" (e.g., "ttbar__nominal")
        or just "process" for data without variations.

    Returns
    -------
    Tuple[str, str]
        (process_name, variation_name) where variation defaults to "nominal"
        if not explicitly specified in the key.

    Examples
    --------
    >>> parse_dataset_key("signal__nominal")
    ("signal", "nominal")
    >>> parse_dataset_key("data")
    ("data", "nominal")
    """
    if DATASET_DELIMITER in dataset_key:
        proc, var = dataset_key.split(DATASET_DELIMITER, 1)
    else:
        proc, var = dataset_key, DEFAULT_VARIATION
    return proc, var


def format_dataset_key(
    process_name: str,
    variation: str = DEFAULT_VARIATION,
    directory_index: int | None = None,
    is_data: bool = False,
) -> str:
    """
    Format a dataset key from components.

    Constructs a dataset key string following the convention:
    - MC: "process__variation" or "process_N__variation" for multi-directory
    - Data: "process" or "process_N" for multi-directory

    Parameters
    ----------
    process_name : str
        Name of the physics process (e.g., "signal", "ttbar")
    variation : str, optional
        Systematic variation label, defaults to "nominal"
    directory_index : int, optional
        Directory index for multi-directory datasets (e.g., different run periods)
    is_data : bool, optional
        Whether this is real data (not MC), defaults to False

    Returns
    -------
    str
        Formatted dataset key

    Examples
    --------
    >>> format_dataset_key("signal", "nominal")
    "signal__nominal"
    >>> format_dataset_key("signal", "nominal", directory_index=0)
    "signal_0__nominal"
    >>> format_dataset_key("data", is_data=True)
    "data"
    >>> format_dataset_key("data", directory_index=1, is_data=True)
    "data_1"
    """
    # Add directory index to process name if provided
    base_name = f"{process_name}_{directory_index}" if directory_index is not None else process_name

    # Data datasets don't include variation in key
    if is_data:
        return base_name

    # MC datasets include variation
    return f"{base_name}{DATASET_DELIMITER}{variation}"


def build_fileset_entry(
    file_paths: List[str],
    tree_name: str,
    process_name: str,
    variation: str,
    xsec: float,
    is_data: bool,
) -> Dict[str, Any]:
    """
    Build a single fileset entry for coffea.

    Creates the dictionary structure that coffea expects for a single dataset,
    including file paths mapped to tree names and metadata.

    Parameters
    ----------
    file_paths : List[str]
        ROOT file paths for this dataset
    tree_name : str
        Name of TTree in ROOT files (e.g., "Events")
    process_name : str
        Physics process name
    variation : str
        Systematic variation label
    xsec : float
        Cross-section in picobarns
    is_data : bool
        Whether this is real data

    Returns
    -------
    Dict[str, Any]
        Fileset entry with structure:
        {
            "files": {path: tree_name, ...},
            "metadata": {process, variation, xsec, is_data}
        }

    Examples
    --------
    >>> entry = build_fileset_entry(
    ...     ["file1.root", "file2.root"],
    ...     "Events",
    ...     "signal",
    ...     "nominal",
    ...     100.5,
    ...     False
    ... )
    >>> entry["files"]
    {"file1.root": "Events", "file2.root": "Events"}
    """
    return {
        "files": {file_path: tree_name for file_path in file_paths},
        "metadata": {
            "process": process_name,
            "variation": variation,
            "xsec": xsec,
            "is_data": is_data,
        }
    }


def aggregate_workitem_events(workitems: List[WorkItem]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Aggregate event counts from WorkItems by process, variation, and file.

    Processes WorkItems to count total events per file. Since coffea splits large
    files into multiple WorkItems (chunks), this function sums events across all
    chunks of the same file.

    Parameters
    ----------
    workitems : List[WorkItem]
        WorkItems from coffea preprocessing, each containing dataset, filename,
        and entry range information

    Returns
    -------
    Dict[str, Dict[str, Dict[str, int]]]
        Nested dictionary: process -> variation -> filename -> event_count

    Examples
    --------
    >>> workitems = [...]  # From coffea preprocessing
    >>> counts = aggregate_workitem_events(workitems)
    >>> counts["signal"]["nominal"]["file1.root"]
    10000
    """
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for wi in workitems:
        wi_dict = dataclasses.asdict(wi)

        dataset = wi_dict["dataset"]
        filename = wi_dict["filename"]
        start = int(wi_dict.get("entrystart", 0))
        stop = int(wi_dict.get("entrystop", 0))

        # Calculate number of events in this chunk
        nevts = max(0, stop - start)

        # Parse dataset key to get process and variation
        proc, var = parse_dataset_key(dataset)

        # Aggregate event counts
        counts[proc][var][filename] += nevts

    return dict(counts)


def format_event_summary(
    event_counts: Dict[str, Dict[str, Dict[str, int]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Format aggregated event counts into final summary structure.

    Converts the raw event counts into the schema used for nanoaods.json,
    including per-file breakdowns and total event counts.

    Parameters
    ----------
    event_counts : Dict[str, Dict[str, Dict[str, int]]]
        Aggregated counts from aggregate_workitem_events()

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Formatted summary with structure:
        {
            "process": {
                "variation": {
                    "files": [{"path": str, "nevts": int}, ...],
                    "nevts_total": int
                }
            }
        }

    Examples
    --------
    >>> counts = {"signal": {"nominal": {"file1.root": 1000, "file2.root": 2000}}}
    >>> summary = format_event_summary(counts)
    >>> summary["signal"]["nominal"]["nevts_total"]
    3000
    """
    summary: Dict[str, Dict[str, Any]] = {}

    for proc, per_var in event_counts.items():
        summary[proc] = {}
        for var, per_file in per_var.items():
            # Create sorted list of files with event counts
            files_list = [
                {"path": str(path), "nevts": nevts}
                for path, nevts in sorted(per_file.items())
            ]

            # Calculate total events
            nevts_total = sum(f["nevts"] for f in files_list)

            summary[proc][var] = {
                "files": files_list,
                "nevts_total": int(nevts_total),
            }

    return summary


def extract_nevts_from_summary(
    fileset_key: str,
    variation: str,
    nanoaods_summary: Dict[str, Dict[str, Any]] | None,
) -> int:
    """
    Extract nevts from nanoaods_summary for a given fileset_key.

    The nanoaods_summary has structure:
    {
        "dataset_name": {
            "variation": {
                "nevts_total": 12345,
                ...
            }
        }
    }

    Parameters
    ----------
    fileset_key : str
        Fileset key in format "datasetname__variation" or "datasetname"
    variation : str
        Variation name (e.g., "nominal", "JESUp")
    nanoaods_summary : dict, optional
        Summary dictionary from metadata generation

    Returns
    -------
    int
        Number of events, or 0 if not found
    """
    nevts = 0

    if nanoaods_summary:
        # Extract dataset name from fileset_key (format: "datasetname__variation")
        # This handles multi-directory datasets where dataset name includes
        # directory index (e.g., signal_0, signal_1)
        dataset_name_from_key = fileset_key.rsplit(DATASET_DELIMITER, 1)[0]

        if dataset_name_from_key in nanoaods_summary:
            if variation in nanoaods_summary[dataset_name_from_key]:
                nevts = nanoaods_summary[dataset_name_from_key][variation].get(
                    "nevts_total", 0
                )

    if nevts == 0:
        logger.warning(f"No nevts found for {fileset_key}, using 0")

    return nevts
