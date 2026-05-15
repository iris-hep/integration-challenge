"""I/O operations for metadata extraction.

This module handles file system operations for JSON metadata save/load and
WorkItem serialization. All functions in this module interact with the file
system.
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
