"""Utility functions for skimming operations.

This module provides helper functions used across the skimming subsystem.
"""

import hist


def default_histogram() -> hist.Hist:
    """Create a default histogram for tracking processing success/failure.

    This histogram serves as a dummy placeholder to track whether workitems
    were processed successfully. The actual analysis histograms are created
    separately during the analysis phase.

    Returns
    -------
    hist.Hist
        A simple histogram with regular binning for tracking purposes
    """
    return hist.Hist.new.Regular(10, 0, 1000).Weight()
