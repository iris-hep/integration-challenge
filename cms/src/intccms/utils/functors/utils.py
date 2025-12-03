"""Utility functions for functor argument preparation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import awkward as ak

logger = logging.getLogger(__name__)


def get_function_arguments(
    arg_spec: Optional[List[Tuple[str, Optional[str]]]],
    objects: Dict[str, ak.Array],
    function_name: Optional[str] = "generic_function",
    static_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ak.Array], Dict[str, Any]]:
    """Prepare function arguments from object dictionary with optional static kwargs.

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
        """Raise KeyError if object is missing in objects dictionary.

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
        if field_name and obj_name != "event":
            try:
                args.append(objects[obj_name][field_name])
            except KeyError:
                raise_error(f"{obj_name}.{field_name}")
        elif obj_name == "event" and field_name:
            try:
                args.append(objects[field_name])
            except KeyError:
                raise_error(f"event level field {field_name}")
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
