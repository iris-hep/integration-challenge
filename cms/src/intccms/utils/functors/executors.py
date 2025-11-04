"""Concrete functor executor implementations for different use cases."""

import logging
from typing import Any, Dict

import awkward as ak
import numpy as np

from intccms.utils.functors.base import FunctorExecutor

logger = logging.getLogger(__name__)


class MaskExecutor(FunctorExecutor):
    """Executes object mask functors and applies to filter objects.

    Required kwargs:
        object_name (str): Name of the object collection to filter
    """

    def apply_result(
        self, result: Any, objects: Dict[str, ak.Array], **kwargs
    ) -> ak.Array:
        """Apply mask to filter object collection."""
        # Validate result type
        if not isinstance(result, ak.Array):
            raise TypeError(f"Mask must be awkward array, got {type(result)}")

        # Get required parameter
        object_name = kwargs.get("object_name")
        if not object_name:
            raise ValueError("MaskExecutor requires 'object_name' parameter")

        # Apply mask and return filtered array
        return objects[object_name][result]


class SelectionExecutor(FunctorExecutor):
    """Executes selection functors returning event masks.

    Returns the selection mask as an awkward array. Handles both
    PackedSelection objects and direct awkward arrays.
    """

    def apply_result(
        self, result: Any, objects: Dict[str, ak.Array], **kwargs
    ) -> ak.Array:
        """Return selection mask, handling PackedSelection if needed."""
        # If result is PackedSelection, extract the combined mask
        if hasattr(result, "all") and hasattr(result, "names"):
            # PackedSelection object
            return ak.Array(result.all(result.names[-1]))

        # Otherwise assume it's already an awkward array mask
        return result


class ObservableExecutor(FunctorExecutor):
    """Executes observable functors for histogram filling.

    Returns the computed observable value as-is. Histogram filling
    happens elsewhere in the analysis pipeline.
    """

    def apply_result(
        self, result: Any, objects: Dict[str, ak.Array], **kwargs
    ) -> Any:
        """Return observable value as-is."""
        return result


class FeatureExecutor(FunctorExecutor):
    """Executes feature functors with optional scaling for MVA.

    Applies scaling if configured and converts to numpy array.
    """

    def apply_result(
        self, result: Any, objects: Dict[str, ak.Array], **kwargs
    ) -> np.ndarray:
        """Apply scaling if configured and return as numpy array."""
        # Apply scaling transform if present
        if hasattr(self.config, "scale") and self.config.scale:
            result = self.config.scale(result)

        # Convert to numpy array
        return np.asarray(result)


class GhostObservableExecutor(FunctorExecutor):
    """Executes ghost observable functors and attaches results to objects.

    Ghost observables compute derived quantities once and attach them
    to the objects dictionary for reuse.

    Required kwargs:
        field_names (List[str]): Names for the computed fields
    """

    def apply_result(
        self, result: Any, objects: Dict[str, ak.Array], **kwargs
    ) -> Dict[str, ak.Array]:
        """Attach computed fields to objects dictionary."""
        field_names = kwargs.get("field_names")
        if not field_names:
            raise ValueError("GhostObservableExecutor requires 'field_names' parameter")

        # Handle tuple results (multiple outputs)
        if isinstance(result, tuple):
            if len(result) != len(field_names):
                raise ValueError(
                    f"Ghost observable returned {len(result)} values but "
                    f"{len(field_names)} field names provided"
                )
            for name, value in zip(field_names, result):
                objects[name] = value
        else:
            # Single output
            if len(field_names) != 1:
                raise ValueError(
                    f"Ghost observable returned single value but "
                    f"{len(field_names)} field names provided"
                )
            objects[field_names[0]] = result

        return objects


class CorrectionExecutor(FunctorExecutor):
    """Executes correction functors and applies to object fields.

    Required kwargs:
        object_name (str): Name of the object collection
        field_name (str): Name of the field to correct
    """

    def apply_result(
        self, result: Any, objects: Dict[str, ak.Array], **kwargs
    ) -> Dict[str, ak.Array]:
        """Apply correction to object field."""
        object_name = kwargs.get("object_name")
        field_name = kwargs.get("field_name")

        if not object_name:
            raise ValueError("CorrectionExecutor requires 'object_name' parameter")
        if not field_name:
            raise ValueError("CorrectionExecutor requires 'field_name' parameter")

        # Apply correction to the field
        objects[object_name][field_name] = result

        return objects
