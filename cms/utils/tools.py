import logging
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import awkward as ak

logger = logging.getLogger(__name__)

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


def get_function_arguments(
    arg_spec: Optional[List[Tuple[str, Optional[str]]]],
    objects: Dict[str, ak.Array],
    function_name: Optional[str] = "generic_function",
    static_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ak.Array], Dict[str, Any]]:
    """
    Prepare function arguments from object dictionary with optional static kwargs.

    Parameters
    ----------
    arg_spec : Optional[List[Tuple[str, Optional[str]]]]
        Optional list of (object, field) specifications for dynamic arguments
    objects : Dict[str, ak.Array]
        Object dictionary
    function_name : Optional[str]
        Name of function for error reporting
    static_kwargs : Optional[Dict[str, Any]]
        Optional static keyword arguments to append when invoking the function.

    Returns
    -------
    Tuple[List[ak.Array], Dict]
        Tuple of (prepared positional arguments, static keyword arguments)
    """
    def raise_error(field_name: str) -> None:
        """
        Raise KeyError if object is missing in objects dictionary.

        Parameters
        ----------
        field_name : str
            Missing field name
        """
        logger.error(
            f"Field '{field_name}' needed for {function_name} "
            f"is not found in objects dictionary"
        )
        raise KeyError(f"Missing field: {field_name}, function: {function_name}")

    args: List[ak.Array] = []
    for obj_name, field_name in arg_spec or []:
        if field_name:
            try:
                args.append(objects[obj_name][field_name])
            except KeyError:
                raise_error(f"{obj_name}.{field_name}")
        else:
            try:
                args.append(objects[obj_name])
            except KeyError:
                raise_error(obj_name)

    if static_kwargs is not None and not isinstance(static_kwargs, dict):
        raise TypeError(
            f"static_kwargs for {function_name} must be a dictionary of keyword "
            f"arguments, got {type(static_kwargs)}"
        )

    final_static_kwargs = dict(static_kwargs) if static_kwargs is not None else {}
    return args, final_static_kwargs
