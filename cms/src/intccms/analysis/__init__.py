import logging

from . import base as base
from . import cms as cms
from .base import Analysis
from .cms import CMSAnalysis
from .processors import SkimAndAnalyseProcessor, TwoHundredGbpsProcessor, HistServProcessor
from .runner import run_processor_workflow


__all__ = [
    "base",
    "cms",
    "Analysis",
    "CMSAnalysis",
    "SkimAndAnalyseProcessor",
    "TwoHundredGbpsProcessor",
    "HistServProcessor",
    "run_processor_workflow",
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
