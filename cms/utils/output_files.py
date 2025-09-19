import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import uproot

# Configure module-level logger
typing_logger = logging.getLogger(__name__)


def save_histograms_to_pickle(
    histograms: Dict[str, Dict[str, Any]], pickle_path: Union[str, Path]
) -> None:
    """
    Save a nested dictionary of histograms to a pickle file.

    Parameters
    ----------
    histograms : dict
        Mapping from channel names to observables to histogram objects.
    pickle_path : str or Path
        Path to the output pickle file. The directory will be
        created if it does not exist.

    Raises
    ------
    IOError
        If writing to the pickle file fails.
    """
    path = Path(pickle_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(histograms, file)
        typing_logger.info(f"Histograms successfully pickled to {path}")
    except Exception as exc:
        typing_logger.error(f"Failed to pickle histograms to {path}: {exc}")
        raise


def load_histograms_from_pickle(
    pickle_path: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """
    Load a nested dictionary of histograms from a pickle file.

    Parameters
    ----------
    pickle_path : str or Path
        Path to the input pickle file.

    Returns
    -------
    dict
        Nested mapping from channel names to observables to histogram objects.

    Raises
    ------
    FileNotFoundError
        If the specified pickle file does not exist.
    IOError
        If reading from the pickle file fails.
    """
    path = Path(pickle_path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    try:
        with path.open("rb") as file:
            histograms = pickle.load(file)
        typing_logger.info(f"Histograms successfully loaded from {path}")
        return histograms
    except Exception as exc:
        typing_logger.error(f"Failed to load histograms from {path}: {exc}")
        raise


def save_histograms_to_root(
    histograms: Dict[str, Dict[str, Any]],
    root_path: Union[str, Path],
    add_offset: bool = False,
) -> None:
    """
    Save histograms to a ROOT file using uproot.

    Parameters
    ----------
    histograms : dict
        Nested mapping of channel names to observables to histogram objects.
    root_path : str or Path
        Path to the output ROOT (.root) file. The directory will be
        created if it does not exist.
    add_offset : bool, optional
        If True, add a small offset to each bin to avoid empty bins
        (default is False).

    Notes
    -----
    - Histograms with no entries (after optional offset) are skipped.
    - Filenames in the ROOT file follow the pattern:
      "<channel>__<observable>__<sample>[__<variation>]".
    """
    path = Path(root_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(str(path)) as root_file:
            for channel, obs_dict in histograms.items():
                for observable, hist in obs_dict.items():
                    # Optionally add a minimal floating-point offset
                    if add_offset:
                        hist = hist + 1e-6
                        num_bins = hist.axes[0].size
                        empty_threshold = num_bins * 1e-6 * 1.01
                    else:
                        empty_threshold = 0.0

                    # Iterate samples and systematic variations
                    for sample in hist.axes[1]:
                        sample_hist = hist[:, sample, :]
                        for variation in sample_hist.axes[1]:
                            # Skip non-nominal variations for data
                            if sample == "data" and variation != "nominal":
                                continue

                            # Construct key and histogram slice
                            suffix = (
                                ""
                                if variation == "nominal"
                                else f"__{variation}"
                            )
                            hist_slice = hist[:, sample, variation]

                            # Check for non-empty histogram
                            total_entries = sum(hist_slice.values())
                            if total_entries > empty_threshold:
                                key = f"{channel}__{observable}__{sample}{suffix}"
                                root_file[key] = hist_slice
                                typing_logger.debug(
                                    f"Saved ROOT histogram: {key}"
                                )
                            else:
                                typing_logger.warning(
                                    f"Skipping empty histogram: "
                                    f"{channel}__{observable}__{sample}{suffix}"
                                )
        typing_logger.info(
            f"Histograms successfully written to ROOT file {path}"
        )
    except Exception as exc:
        typing_logger.error(f"Failed to write ROOT file {path}: {exc}")
        raise
