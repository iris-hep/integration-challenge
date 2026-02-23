"""
Histogram I/O and filtering utilities.

This module provides pure I/O functions for saving/loading histograms
and explicit filtering functions for preparing histograms for export.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import uproot

logger = logging.getLogger(__name__)


# ============================================================================
# Pure I/O Functions (no filtering logic)
# ============================================================================

def _write_histograms_to_pickle(
    histograms: Dict[str, Dict[str, Any]], output_file: Union[str, Path]
) -> None:
    """
    Write histograms to a pickle file (no filtering applied).

    Parameters
    ----------
    histograms : dict
        Mapping from channel names to observables to histogram objects
    output_file : str or Path
        Path to the output pickle file. Directory created if needed.

    Raises
    ------
    IOError
        If writing to the pickle file fails
    """
    path = Path(output_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(histograms, file)
        logger.info(f"Histograms written to pickle: {path}")
    except Exception as exc:
        logger.error(f"Failed to write pickle {path}: {exc}")
        raise


def load_histograms_from_pickle(
    output_file: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """
    Load histograms from a pickle file.

    Parameters
    ----------
    output_file : str or Path
        Path to the input pickle file

    Returns
    -------
    dict
        Nested mapping from channel names to observables to histogram objects

    Raises
    ------
    FileNotFoundError
        If the specified pickle file does not exist
    IOError
        If reading from the pickle file fails
    """
    path = Path(output_file)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    try:
        with path.open("rb") as file:
            histograms = pickle.load(file)
        logger.info(f"Histograms loaded from pickle: {path}")
        return histograms
    except Exception as exc:
        logger.error(f"Failed to load pickle {path}: {exc}")
        raise


def _write_histograms_to_root(
    histograms: Dict[str, Dict[str, Any]],
    output_file: Union[str, Path],
    add_offset: bool = False,
) -> None:
    """
    Write histograms to a ROOT file using uproot (no filtering applied).

    Parameters
    ----------
    histograms : dict
        Nested mapping of channel names to observables to histogram objects
    output_file : str or Path
        Path to the output ROOT file. Directory created if needed.
    add_offset : bool, default=False
        If True, add a small offset (1e-6) to each bin to avoid empty bins

    Notes
    -----
    - Filenames in ROOT file: "<channel>__<observable>__<sample>[__<variation>]"
    - This is pure I/O - no filtering for empty histograms or invalid systematics
    """
    path = Path(output_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(str(path)) as root_file:
            for channel, obs_dict in histograms.items():
                for observable, hist in obs_dict.items():
                    # Optionally add a minimal floating-point offset
                    if add_offset:
                        hist = hist + 1e-6

                    # Iterate samples and systematic variations
                    for sample in hist.axes[1]:
                        sample_hist = hist[:, sample, :]
                        for variation in sample_hist.axes[1]:
                            # Construct key
                            suffix = (
                                ""
                                if variation == "nominal"
                                else f"__{variation}"
                            )
                            hist_slice = hist[:, sample, variation]
                            key = f"{channel}__{observable}__{sample}{suffix}"

                            # Write to ROOT file
                            root_file[key] = hist_slice
                            logger.debug(f"Wrote ROOT histogram: {key}")

        logger.info(f"Histograms written to ROOT file: {path}")
    except Exception as exc:
        logger.error(f"Failed to write ROOT file {path}: {exc}")
        raise


# ============================================================================
# Filtering Functions
# ============================================================================

def filter_invalid_systematics(
    histograms: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Remove systematic variations from data samples.

    Data samples don't have systematic variations - only the nominal histogram
    is valid. This function filters out non-nominal variations for samples
    named 'data'.

    Parameters
    ----------
    histograms : dict
        Nested mapping of channel -> observable -> histogram

    Returns
    -------
    dict
        Filtered histograms with invalid systematics removed
    """
    filtered = {}
    skipped_count = 0

    for channel, obs_dict in histograms.items():
        filtered[channel] = {}
        for observable, hist in obs_dict.items():
            # Check each sample and systematic
            for sample in hist.axes[1]:
                for variation in hist[:, sample, :].axes[1]:
                    # Skip non-nominal variations for data
                    if sample == "data" and variation != "nominal":
                        logger.debug(
                            f"Skipping invalid systematic: {channel}__{observable}__data__{variation}"
                        )
                        skipped_count += 1
                        continue

            # Keep the histogram (filtering happens during ROOT export)
            filtered[channel][observable] = hist

    return filtered


def filter_empty_histograms(
    histograms: Dict[str, Dict[str, Any]],
    add_offset: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Remove histograms with zero entries.

    Empty histograms can cause issues in downstream analysis tools (ROOT, cabinetry).
    This function identifies and removes them.

    Parameters
    ----------
    histograms : dict
        Nested mapping of channel -> observable -> histogram
    add_offset : bool, default=False
        If True, consider offset threshold when determining emptiness

    Returns
    -------
    dict
        Filtered histograms with empty entries removed
    """
    filtered = {}
    skipped_count = 0

    for channel, obs_dict in histograms.items():
        filtered[channel] = {}
        for observable, hist in obs_dict.items():
            # Calculate empty threshold
            if add_offset:
                num_bins = hist.axes[0].size
                empty_threshold = num_bins * 1e-6 * 1.01
            else:
                empty_threshold = 0.0

            # Check each sample/variation
            has_entries = False
            for sample in hist.axes[1]:
                for variation in hist[:, sample, :].axes[1]:
                    hist_slice = hist[:, sample, variation]
                    total_entries = sum(hist_slice.values())

                    if total_entries > empty_threshold:
                        has_entries = True
                        break
                if has_entries:
                    break

            if has_entries:
                filtered[channel][observable] = hist
            else:
                logger.warning(
                    f"Skipping empty histogram: {channel}__{observable}"
                )
                skipped_count += 1

    if skipped_count > 0:
        logger.info(f"Filtered {skipped_count} empty histograms")

    return filtered


# ============================================================================
# Convenience Facades (apply filtering then write)
# ============================================================================

def save_histograms_to_pickle(
    histograms: Dict[str, Dict[str, Any]], output_file: Union[str, Path]
) -> None:
    """
    Save histograms to pickle format (convenience wrapper).

    This is a simple wrapper around _write_histograms_to_pickle for API consistency.
    No filtering is applied since pickle preserves everything.

    Parameters
    ----------
    histograms : dict
        Mapping from channel names to observables to histogram objects
    output_file : str or Path
        Path to the output pickle file
    """
    _write_histograms_to_pickle(histograms, output_file)


def save_histograms_to_root(
    histograms: Dict[str, Dict[str, Any]],
    output_file: Union[str, Path],
    add_offset: bool = False,
    skip_empty: bool = True,
    skip_invalid_systematics: bool = True,
) -> None:
    """
    Save histograms to ROOT format with optional filtering.

    This facade applies filters before writing to ensure clean ROOT output.

    Parameters
    ----------
    histograms : dict
        Nested mapping of channel names to observables to histogram objects
    output_file : str or Path
        Path to the output ROOT file
    add_offset : bool, default=False
        If True, add small offset (1e-6) to bins to avoid zeros
    skip_empty : bool, default=True
        If True, filter out histograms with zero entries
    skip_invalid_systematics : bool, default=True
        If True, remove systematic variations from data samples

    Notes
    -----
    Filtering options:
    - skip_empty: Removes histograms that would cause issues in ROOT/cabinetry
    - skip_invalid_systematics: Data samples have no systematics (only nominal)
    """
    filtered = histograms

    # Apply filters if requested
    if skip_invalid_systematics:
        filtered = filter_invalid_systematics(filtered)

    if skip_empty:
        filtered = filter_empty_histograms(filtered, add_offset=add_offset)

    # Write to ROOT file
    _write_histograms_to_root(filtered, output_file, add_offset=add_offset)
