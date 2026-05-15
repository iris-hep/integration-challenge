"""Metrics and inspection tools.

Runtime metrics collection (processor timing, throughput, worker tracking)
is handled by roastcoffea. This package provides input data inspection
utilities for characterizing ROOT files before processing.
"""

from . import inspector

__all__ = ["inspector"]
