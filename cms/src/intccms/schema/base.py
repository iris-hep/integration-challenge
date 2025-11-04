"""Base classes for configuration models.

This module provides the foundational classes used throughout the schema package.
"""

from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# Type alias for (object, variable) pairs
ObjVar = Tuple[str, Optional[str]]


class WorkerEval:
    """
    Marker for lazy evaluation on distributed workers.

    Wraps a callable to indicate it should be evaluated on the worker
    rather than passed through. This distinguishes between:
    - Values computed from worker environment: WorkerEval(lambda: os.environ['KEY'])
    - Actual callable arguments: my_compression_func (passed through as-is)

    Useful when configuration values need to access environment variables
    that exist on workers but not on the client side (e.g., in dask distributed).

    Parameters
    ----------
    func : Callable
        Function to evaluate on the worker. Should take no arguments.

    Examples
    --------
    >>> import os
    >>> # Resolved on worker:
    >>> key = WorkerEval(lambda: os.environ['AWS_ACCESS_KEY_ID'])
    >>>
    >>> # Passed through as callable:
    >>> compression = my_custom_compressor
    """
    __slots__ = ('func',)

    def __init__(self, func):
        if not callable(func):
            raise TypeError(f"WorkerEval requires a callable, got {type(func)}")
        self.func = func

    def __call__(self):
        """Evaluate the wrapped function."""
        return self.func()

    def __repr__(self):
        return f"WorkerEval({self.func!r})"


_UNSET = object()


class SubscriptableModel(BaseModel):
    """A Pydantic BaseModel that supports dictionary-style item access."""

    def __getitem__(self, key):
        """Allows dictionary-style `model[key]` access."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Allows dictionary-style `model[key] = value` assignment."""
        return setattr(self, key, value)

    def __contains__(self, key):
        """Allows `key in model` checks."""
        return hasattr(self, key)

    def get(self, key, default=None):
        """Allows `.get(key, default)` method."""
        return getattr(self, key, default)

    def pop(self, key, default=_UNSET):
        """
        Remove a field from the model and return its value.

        Mirrors dict.pop semantics: raises KeyError if missing and no default
        is provided, otherwise returns the default.
        """
        if hasattr(self, key):
            value = getattr(self, key)
            self.__dict__.pop(key, None)
            fields_set = getattr(self, "model_fields_set", None)
            if fields_set is not None:
                fields_set.discard(key)
            return value
        if default is _UNSET:
            raise KeyError(key)
        return default


class FunctorConfig(SubscriptableModel):
    """Base configuration for functor-based operations.

    Functors are callable objects that process physics data with specified
    inputs and parameters. Used across skimming, analysis, and MVA.
    """

    function: Annotated[
        Callable,
        Field(description="A Python callable to be executed."),
    ]
    use: Annotated[
        Optional[List[ObjVar]],
        Field(
            default=None,
            description="A list of (object variable) tuples specifying "
            "the inputs for the function.",
        ),
    ]
    static_kwargs: Annotated[
        Optional[Dict[str, Any]],
        Field(
            default=None,
            description=(
                "Optional static keyword arguments appended when invoking the "
                "function. These values do not depend on event data."
            ),
        ),
    ]
