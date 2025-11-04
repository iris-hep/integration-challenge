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
        # Import here to avoid circular dependency
        from coffea.analysis_tools import PackedSelection

        # Validate and extract from PackedSelection
        if hasattr(result, "all") and hasattr(result, "names"):
            if not isinstance(result, PackedSelection):
                raise TypeError(
                    f"Expected PackedSelection, got {type(result).__name__}"
                )
            # PackedSelection object - extract the combined mask
            return ak.Array(result.all(result.names[-1]))

        # Direct awkward array mask
        if isinstance(result, ak.Array):
            return result

        # Invalid result type
        raise TypeError(
            f"Selection function must return PackedSelection or ak.Array, "
            f"got {type(result).__name__}"
        )


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

    Optional kwargs:
        apply_scaling (bool): Whether to apply scaling transform (default: True)
    """

    def apply_result(
        self, result: Any, objects: Dict[str, ak.Array], **kwargs
    ) -> np.ndarray:
        """Apply scaling if configured and return as numpy array."""
        apply_scaling = kwargs.get("apply_scaling", True)

        # Apply scaling transform if present and requested
        if apply_scaling and hasattr(self.config, "scale") and self.config.scale:
            result = self.config.scale(result)

        # Convert to numpy array
        return np.asarray(result)


class GhostObservableExecutor(FunctorExecutor):
    """Executes ghost observable functors and attaches results to objects.

    Ghost observables compute derived quantities once and attach them
    to the objects dictionary for reuse. Handles collection assignment,
    single-field record unwrapping, and creation of new collections.

    Required kwargs:
        field_names (List[str]): Names for the computed fields
        collections (List[str]): Collection names to attach fields to
    """

    def apply_result(
        self, result: Any, objects: Dict[str, ak.Array], **kwargs
    ) -> Dict[str, ak.Array]:
        """Attach computed fields to collections in objects dictionary."""
        field_names = kwargs.get("field_names")
        collections = kwargs.get("collections")

        if not field_names:
            raise ValueError("GhostObservableExecutor requires 'field_names' parameter")
        if not collections:
            raise ValueError("GhostObservableExecutor requires 'collections' parameter")

        # Normalize result to list
        if not isinstance(result, (list, tuple)):
            result = [result]

        # Validate lengths match
        if len(result) != len(field_names):
            raise ValueError(
                f"Ghost observable returned {len(result)} values but "
                f"{len(field_names)} field names provided"
            )
        if len(field_names) != len(collections):
            raise ValueError(
                f"Got {len(field_names)} field names but {len(collections)} collections"
            )

        # Attach each output to its collection
        for value, name, collection in zip(result, field_names, collections):
            # Handle single-field records
            if (
                isinstance(value, ak.Array)
                and len(ak.fields(value)) == 1
                and name in ak.fields(value)
            ):
                value = value[name]

            # Update existing collection
            if collection in objects:
                try:
                    objects[collection][name] = value
                except ValueError as error:
                    logger.exception(
                        f"Failed to add field '{name}' to collection '{collection}'"
                    )
                    raise error
            # Create new collection
            else:
                objects[collection] = ak.Array({name: value})

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
