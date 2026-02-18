import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import awkward as ak

logger = logging.getLogger(__name__)


def load_dotenv(path: str = ".env", overwrite: bool = True) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Ignores comments, blank lines, and keys already set in the environment.
    Does nothing if the file does not exist.

    Args:
        path: Path to the .env file (default: ".env" in current directory).
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            if key not in os.environ or overwrite: 
                os.environ[key] = value

def nested_defaultdict_to_dict(nested_structure: Any) -> dict:
    """
    Recursively convert any nested defaultdicts into standard Python dictionaries.

    Parameters
    ----------
    nested_structure : Any
        A nested structure possibly containing defaultdicts.

    Returns
    -------
    dict
        Fully converted structure using built-in dict.
    """
    if isinstance(nested_structure, defaultdict):
        return {
            key: nested_defaultdict_to_dict(value)
            for key, value in nested_structure.items()
        }
    elif isinstance(nested_structure, dict):
        return {
            key: nested_defaultdict_to_dict(value)
            for key, value in nested_structure.items()
        }
    return nested_structure


def recursive_to_backend(data_structure: Any, backend: str = "jax") -> Any:
    """
    Recursively convert all Awkward Arrays in a data structure to the specified backend.

    Parameters
    ----------
    data_structure : Any
        Input data structure possibly containing Awkward Arrays.
    backend : str
        Target backend to convert arrays to (e.g. 'jax', 'cpu').

    Returns
    -------
    Any
        Data structure with Awkward Arrays converted to the desired backend.
    """
    if isinstance(data_structure, ak.Array):
        # Convert only if not already on the target backend
        return (
            ak.to_backend(data_structure, backend)
            if ak.backend(data_structure) != backend
            else data_structure
        )
    elif isinstance(data_structure, Mapping):
        # Recurse into dictionary values
        return {
            key: recursive_to_backend(value, backend)
            for key, value in data_structure.items()
        }
    elif isinstance(data_structure, Sequence) and not isinstance(
        data_structure, (str, bytes)
    ):
        # Recurse into list or tuple elements
        return [
            recursive_to_backend(value, backend) for value in data_structure
        ]
    else:
        # Leave unchanged if not an Awkward structure
        return data_structure
