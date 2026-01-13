"""Functor execution system for OO-based analysis workflows.

This module provides an object-oriented approach to executing functors,
replacing repetitive boilerplate with reusable executor classes.
"""

from intccms.utils.functors.base import FunctorExecutor
from intccms.utils.functors.executors import (
    CorrectionExecutor,
    FeatureExecutor,
    GhostObservableExecutor,
    MaskExecutor,
    ObservableExecutor,
    SelectionExecutor,
)
from intccms.utils.functors.utils import get_function_arguments

__all__ = [
    "FunctorExecutor",
    "MaskExecutor",
    "SelectionExecutor",
    "ObservableExecutor",
    "FeatureExecutor",
    "GhostObservableExecutor",
    "CorrectionExecutor",
    "get_function_arguments",
]
