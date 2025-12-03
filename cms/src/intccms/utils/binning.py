"""Utilities for parsing and working with histogram binning configurations.

This module centralizes the logic for handling binning specifications in both
string format ("low,high,nbins") and explicit edge lists.
"""

from typing import List, Optional, Union

import hist
import numpy as np


def validate_binning_spec(v: Union[str, List[float], None]) -> Union[str, List[float], None]:
    """Validate binning specification for Pydantic field validators.

    For string format: must be 'low,high,nbins' with low < high and nbins > 0
    For list format: must have at least 2 edges in ascending order

    Parameters
    ----------
    v : Union[str, List[float], None]
        Binning specification to validate

    Returns
    -------
    Union[str, List[float], None]
        The validated binning specification (unchanged)

    Raises
    ------
    ValueError
        If binning specification is invalid

    Examples
    --------
    >>> validate_binning_spec("0,100,20")
    '0,100,20'
    >>> validate_binning_spec([0, 20, 50, 100])
    [0, 20, 50, 100]
    >>> validate_binning_spec(None)
    None
    """
    if v is None:
        return v

    if isinstance(v, str):
        # Validate string format "low,high,nbins"
        parts = v.strip().split(",")
        if len(parts) != 3:
            raise ValueError(
                f"Binning string must have format 'low,high,nbins', got '{v}'"
            )
        try:
            low, high, nbins = map(float, parts)
            nbins = int(nbins)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid binning string '{v}': cannot parse as numbers"
            ) from e

        if low >= high:
            raise ValueError(
                f"Invalid binning: low ({low}) must be less than high ({high})"
            )
        if nbins <= 0:
            raise ValueError(f"Invalid binning: nbins ({nbins}) must be positive")

    elif isinstance(v, list):
        # Validate list of bin edges
        if len(v) < 2:
            raise ValueError(
                f"Binning edges must have at least 2 values, got {len(v)}"
            )
        # Check ascending order
        for i in range(len(v) - 1):
            if v[i] >= v[i + 1]:
                raise ValueError(
                    f"Binning edges must be in ascending order, "
                    f"but edge[{i}]={v[i]} >= edge[{i+1}]={v[i+1]}"
                )
    return v


def parse_binning_string(binning_str: str) -> tuple[float, float, int]:
    """Parse a binning string in format 'low,high,nbins'.

    Parameters
    ----------
    binning_str : str
        Binning specification as "low,high,nbins" (e.g., "0,100,20")

    Returns
    -------
    tuple[float, float, int]
        (low, high, nbins) where low and high are bin edges and nbins is the
        number of bins

    Raises
    ------
    ValueError
        If string format is invalid or cannot be parsed

    Examples
    --------
    >>> low, high, nbins = parse_binning_string("0,100,20")
    >>> (low, high, nbins)
    (0.0, 100.0, 20)
    """
    try:
        parts = binning_str.strip().split(",")
        if len(parts) != 3:
            raise ValueError(
                f"Binning string must have exactly 3 comma-separated values, got {len(parts)}"
            )
        low, high, nbins = map(float, parts)
        return low, high, int(nbins)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid binning string '{binning_str}': expected format 'low,high,nbins'"
        ) from e


def create_hist_axis(
    binning: Union[str, List[float]],
    name: str = "observable",
    label: str = "observable",
) -> Union[hist.axis.Regular, hist.axis.Variable]:
    """Create a hist.axis from binning specification.

    Parameters
    ----------
    binning : Union[str, List[float]]
        Either a string "low,high,nbins" or a list of explicit bin edges
    name : str, default="observable"
        Axis name for histogramming
    label : str, default="observable"
        Axis label for plots

    Returns
    -------
    hist.axis.Axis
        Either hist.axis.Regular (for string binning) or hist.axis.Variable
        (for explicit edges)

    Examples
    --------
    >>> axis = create_hist_axis("0,100,20", name="mass", label=r"$m$ [GeV]")
    >>> axis.edges[0], axis.edges[-1], len(axis.edges) - 1
    (0.0, 100.0, 20)

    >>> axis = create_hist_axis([0, 20, 50, 100, 200], name="pt")
    >>> list(axis.edges)
    [0.0, 20.0, 50.0, 100.0, 200.0]
    """
    if isinstance(binning, str):
        low, high, nbins = parse_binning_string(binning)
        return hist.axis.Regular(int(nbins), low, high, name=name, label=label)
    else:
        return hist.axis.Variable(binning, name=name, label=label)


def binning_to_edges(binning: Union[str, List[float]]) -> np.ndarray:
    """Convert binning specification to explicit bin edges.

    Parameters
    ----------
    binning : Union[str, List[float]]
        Either a string "low,high,nbins" or a list of explicit bin edges

    Returns
    -------
    np.ndarray
        Array of bin edges (length = nbins + 1)

    Examples
    --------
    >>> edges = binning_to_edges("0,100,5")
    >>> list(edges)
    [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]

    >>> edges = binning_to_edges([0, 10, 50, 100])
    >>> list(edges)
    [0.0, 10.0, 50.0, 100.0]
    """
    if isinstance(binning, str):
        low, high, nbins = parse_binning_string(binning)
        return np.linspace(low, high, int(nbins) + 1)
    else:
        return np.asarray(binning)
