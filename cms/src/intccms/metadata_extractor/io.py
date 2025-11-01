"""I/O operations for metadata extraction.

This module handles all file system operations including reading file listings,
saving/loading JSON metadata, and WorkItem serialization. All functions in this
module interact with the file system.
"""

import base64
import dataclasses
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Union

from coffea.processor.executor import WorkItem

logger = logging.getLogger(__name__)


def collect_file_paths(
    directory: Union[str, Path],
    identifiers: int | List[int] | None = None,
    redirector: str | None = None,
) -> List[str]:
    """
    Read ROOT file paths from .txt listing files.

    Reads .txt files where each line contains one ROOT file path. This approach
    separates file lists from code, enabling version control and easy updates.
    The `identifiers` parameter allows processing subsets for testing.
    The `redirector` parameter prepends protocol prefixes for remote access.

    Parameters
    ----------
    directory : str or Path
        Directory containing .txt listing files
    identifiers : int or list of ints, optional
        Process only specific listing files by ID (e.g., [0, 1] reads 0.txt, 1.txt).
        If None, reads all .txt files in directory.
    redirector : str, optional
        URL prefix to prepend to paths (e.g., "root://xrootd.server.com//").
        If None, paths used as-is.

    Returns
    -------
    List[str]
        ROOT file paths with optional redirector prefix applied.

    Raises
    ------
    FileNotFoundError
        If directory contains no .txt files or specified identifier file missing.
    """
    dir_path = Path(directory)

    # Determine which text files to parse
    if identifiers is None:
        listing_files = list(dir_path.glob("*.txt"))
    else:
        ids = [identifiers] if isinstance(identifiers, int) else identifiers
        listing_files = [dir_path / f"{i}.txt" for i in ids]

    # Raise error if no listing files are found
    if not listing_files:
        raise FileNotFoundError(f"No listing files found in {dir_path}")

    root_paths: List[str] = []

    # Iterate through each listing file
    for txt_file in listing_files:
        if not txt_file.is_file():
            raise FileNotFoundError(f"Missing listing file: {txt_file}")

        # Read each non-empty line as a file path
        for line in txt_file.read_text().splitlines():
            path_str = line.strip()
            if path_str:
                if redirector:
                    path_str = f"{redirector}{path_str}"
                root_paths.append(path_str)

    return root_paths


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    """
    Save dictionary to JSON file with pretty formatting.

    Parameters
    ----------
    data : dict
        Data to save
    output_path : Path
        Output file path

    Raises
    ------
    OSError
        If file cannot be written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Saved JSON to {output_path}")


def load_json(input_path: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Parameters
    ----------
    input_path : Path
        Input file path

    Returns
    -------
    dict
        Loaded data

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    json.JSONDecodeError
        If file contains invalid JSON
    """
    with input_path.open("r") as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from {input_path}")
    return data


def serialize_workitems(workitems: List[WorkItem]) -> List[Dict[str, Any]]:
    """
    Serialize WorkItems to JSON-compatible format.

    WorkItems contain non-serializable objects (UUIDs), so this function
    converts them to a serializable format using base64 encoding.

    Parameters
    ----------
    workitems : List[WorkItem]
        WorkItems from coffea preprocessing

    Returns
    -------
    List[Dict[str, Any]]
        Serializable dictionaries

    Examples
    --------
    >>> workitems = [...]  # From coffea
    >>> serialized = serialize_workitems(workitems)
    >>> save_json(serialized, Path("workitems.json"))
    """
    serializable = []

    for wi in workitems:
        wi_dict = dataclasses.asdict(wi)

        # Encode file UUID as base64 string for JSON compatibility
        # dataclasses.asdict() converts UUID objects to bytes
        if "fileuuid" in wi_dict and wi_dict["fileuuid"] is not None:
            wi_dict["fileuuid"] = base64.b64encode(wi_dict["fileuuid"]).decode("ascii")

        serializable.append(wi_dict)

    return serializable


def deserialize_workitems(serialized_data: List[Dict[str, Any]]) -> List[WorkItem]:
    """
    Deserialize WorkItems from JSON-compatible format.

    Converts base64-encoded UUIDs back to proper WorkItem objects.

    Parameters
    ----------
    serialized_data : List[Dict[str, Any]]
        Serialized workitems from serialize_workitems()

    Returns
    -------
    List[WorkItem]
        Reconstructed WorkItem objects

    Examples
    --------
    >>> data = load_json(Path("workitems.json"))
    >>> workitems = deserialize_workitems(data)
    """
    workitems = []

    for wi_dict in serialized_data:
        # Decode base64-encoded file UUID back to binary format
        if "fileuuid" in wi_dict and wi_dict["fileuuid"] is not None:
            wi_dict["fileuuid"] = base64.b64decode(wi_dict["fileuuid"])

        workitems.append(WorkItem(**wi_dict))

    return workitems
