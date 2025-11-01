import logging

from . import base as base
from . import nondiff as nondiff
from .base import Analysis
from .nondiff import NonDiffAnalysis
from .processor import UnifiedProcessor


__all__ = [
    "base",
    "nondiff",
    "Analysis",
    "NonDiffAnalysis",
    "UnifiedProcessor",
]


def __dir__():
    return __all__


def set_logging() -> None:
    """Sets up customized and verbose logging output.

    Logging can be alternatively customized with the Python ``logging`` module directly.
    """
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s: %(name)s] %(message)s"
    )
