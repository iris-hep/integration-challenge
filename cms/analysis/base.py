import gzip
import logging
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import awkward as ak
import vector
from coffea.nanoevents import NanoAODSchema
from correctionlib import Correction, CorrectionSet

from utils.schema import GoodObjectMasksConfig
from utils.tools import get_function_arguments
from utils.output_manager import OutputDirectoryManager

# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()

# -----------------------------
# Logging Configuration
# -----------------------------
logger = logging.getLogger("BaseAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")


def is_jagged(array_like: ak.Array) -> bool:
    """
    Determine if an array is jagged (has variable-length subarrays).

    Parameters
    ----------
    array_like : ak.Array
        Input array to check

    Returns
    -------
    bool
        True if the array is jagged, False otherwise
    """
    try:
        return ak.num(array_like, axis=1) is not None
    except Exception:
        return False


class Analysis:
    """Base class for physics analysis implementations."""

    def __init__(
        self,
        config: Dict[str, Any],
        processed_datasets: Dict[str, List[Tuple[Any, Dict[str, Any]]]],
        output_manager: OutputDirectoryManager,
    ) -> None:
        """
        Initialize analysis with configuration for systematics, corrections,
        and channels.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with keys:
            - 'systematics': Systematic variations configuration
            - 'corrections': Correction configurations
            - 'channels': Analysis channel definitions
            - 'general': General settings including output directory
        processed_datasets : Dict[str, List[Tuple[Any, Dict[str, Any]]]]
            Pre-processed datasets from skimming (required)
        output_manager : OutputDirectoryManager
            Centralized output directory manager (required)
        """
        self.config = config
        self.channels = config.channels
        self.systematics = config.systematics
        self.corrections = config.corrections
        self.processed_datasets = processed_datasets
        self.output_manager = output_manager
        self.corrlib_evaluators = self._load_correctionlib()

    def _load_correctionlib(self) -> Dict[str, CorrectionSet]:
        """
        Load correctionlib JSON files into evaluators.

        Returns
        -------
        Dict[str, CorrectionSet]
            Mapping of correction name to CorrectionSet evaluator
        """
        evaluators = {}
        for correction in self.corrections:
            if not correction.use_correctionlib:
                continue

            corr_name = correction.name
            file_path = correction.file

            if file_path.endswith(".json.gz"):
                with gzip.open(file_path, "rt") as file_handle:
                    evaluators[corr_name] = CorrectionSet.from_string(
                        file_handle.read().strip()
                    )
            elif file_path.endswith(".json"):
                evaluators[corr_name] = CorrectionSet.from_file(file_path)
            else:
                raise ValueError(
                    f"Unsupported correctionlib format: {file_path}. "
                    "Expected .json or .json.gz"
                )

        return evaluators

    def get_object_copies(self, events: ak.Array) -> Dict[str, ak.Array]:
        """
        Extract a dictionary of objects from the NanoEvents array.

        Parameters
        ----------
        events : ak.Array
            Input event array

        Returns
        -------
        Dict[str, ak.Array]
            Dictionary of field names to awkward arrays
        """
        return {field: events[field] for field in events.fields}

    def get_good_objects(
        self,
        object_copies: Dict[str, ak.Array],
        masks: Iterable[GoodObjectMasksConfig] = [],
    ) -> Dict[str, ak.Array]:
        """
        Apply selection masks to objects.

        Parameters
        ----------
        object_copies : Dict[str, ak.Array]
            Original objects dictionary
        masks : Iterable[GoodObjectMasksConfig], optional
            List of mask configurations

        Returns
        -------
        Dict[str, ak.Array]
            Dictionary of filtered objects
        """
        good_objects = {}
        for mask_config in masks:
            mask_args = get_function_arguments(
                mask_config.use,
                object_copies,
                function_name=mask_config.function.__name__,
            )

            selection_mask = mask_config.function(*mask_args)
            if not isinstance(selection_mask, ak.Array):
                raise TypeError(
                    f"Mask must be an awkward array. Got {type(selection_mask)}"
                )

            obj_name = mask_config.object
            good_objects[obj_name] = object_copies[obj_name][selection_mask]

        return good_objects

    def apply_object_masks(
        self, object_copies: Dict[str, ak.Array], mask_set: str = "analysis"
    ) -> Dict[str, ak.Array]:
        """
        Apply predefined object masks to object copies.

        Parameters
        ----------
        object_copies : Dict[str, ak.Array]
            Objects to filter
        mask_set : str, optional
            Key for mask configuration (default: "analysis")

        Returns
        -------
        Dict[str, ak.Array]
            Updated objects with masks applied
        """
        mask_configs = self.config.good_object_masks.get(mask_set, [])
        if not mask_configs:
            return object_copies

        filtered_objects = self.get_good_objects(object_copies, mask_configs)
        for obj_name in filtered_objects:
            if obj_name not in object_copies:
                logger.error(f"Object {obj_name} not found in object copies")
                raise KeyError(f"Missing object: {obj_name}")
            object_copies[obj_name] = filtered_objects[obj_name]

        return object_copies

    def apply_correctionlib(
        self,
        correction_name: str,
        correction_key: str,
        direction: Literal["up", "down", "nominal"],
        correction_args: List[ak.Array],
        target: Optional[Union[ak.Array, List[ak.Array]]] = None,
        operation: Optional[str] = None,
        transform: Optional[Callable[..., Any]] = None,
    ) -> Union[ak.Array, List[ak.Array]]:
        """
        Apply correction using correctionlib.

        Parameters
        ----------
        correction_name : str
            Name of the correction in evaluators
        correction_key : str
            Specific correction key
        direction : Literal["up", "down", "nominal"]
            Systematic direction
        correction_args : List[ak.Array]
            Input arguments for correction
        target : Optional[Union[ak.Array, List[ak.Array]]], optional
            Target array(s) to modify
        operation : Optional[str], optional
            Operation to apply ('add' or 'mult')
        transform : Optional[Callable[..., Any]], optional
            Transformation function for arguments

        Returns
        -------
        Union[ak.Array, List[ak.Array]]
            Corrected value(s)
        """
        logger.info(
            "Applying correction: %s/%s (%s)",
            correction_name,
            correction_key,
            direction,
        )

        # Apply argument transformation if provided
        correction_args = transform(*correction_args)

        # Flatten jagged arrays
        flat_args, counts = [], []
        for arg in correction_args:
            if is_jagged(arg):
                flat_args.append(ak.flatten(arg))
                counts.append(ak.num(arg))
            else:
                flat_args.append(arg)

        # Evaluate correction
        correction_evaluator: Correction = self.corrlib_evaluators[
            correction_name
        ][correction_key]
        correction_values = correction_evaluator.evaluate(
            *flat_args, direction
        )

        # Restore jagged structure if needed
        if counts:
            correction_values = ak.unflatten(correction_values, counts[0])

            # Apply to target
            if isinstance(target, list):
                backend = ak.backend(target[0])
                correction_values = ak.to_backend(correction_values, backend)
                return [
                    self._apply_operation(
                        operation, target_array, correction_values
                    )
                    for target_array in target
                ]
            else:
                backend = ak.backend(target)
                correction_values = ak.to_backend(correction_values, backend)
                return self._apply_operation(
                    operation, target, correction_values
                )

        return correction_values

    def apply_syst_function(
        self,
        syst_name: str,
        syst_function: Callable[..., ak.Array],
        function_args: List[ak.Array],
        affected_arrays: Union[ak.Array, List[ak.Array]],
        operation: str,
    ) -> Union[ak.Array, List[ak.Array]]:
        """
        Apply function-based systematic variation.

        Parameters
        ----------
        syst_name : str
            Systematic name
        syst_function : Callable[..., ak.Array]
            Variation function
        function_args : List[ak.Array]
            Function arguments
        affected_arrays : Union[ak.Array, List[ak.Array]]
            Array(s) to modify
        operation : str
            Operation to apply ('add' or 'mult')

        Returns
        -------
        Union[ak.Array, List[ak.Array]]
            Modified array(s)
        """
        logger.debug("Applying function-based systematic: %s", syst_name)
        variation = syst_function(*function_args)

        if isinstance(affected_arrays, list):
            return [
                self._apply_operation(operation, arr, variation)
                for arr in affected_arrays
            ]
        return self._apply_operation(operation, affected_arrays, variation)

    def _apply_operation(
        self,
        operation: str,
        left_operand: ak.Array,
        right_operand: ak.Array,
    ) -> ak.Array:
        """
        Apply binary operation between two arrays.

        Parameters
        ----------
        operation : str
            Operation type ('add' or 'mult')
        left_operand : ak.Array
            Left operand array
        right_operand : ak.Array
            Right operand array

        Returns
        -------
        ak.Array
            Result of operation

        Raises
        -------
        ValueError
            For unsupported operations
        """
        if operation == "add":
            return left_operand + right_operand
        elif operation == "mult":
            return left_operand * right_operand
        else:
            raise ValueError(f"Unsupported operation: '{operation}'")

    def _get_target_arrays(
        self,
        target_spec: Union[Tuple[str, str], List[Tuple[str, str]]],
        objects: Dict[str, ak.Array],
        function_name: Optional[str] = "generic_target_function",
    ) -> List[ak.Array]:
        """
        Extract target arrays from object dictionary.

        Parameters
        ----------
        target_spec : Union[Tuple[str, str], List[Tuple[str, str]]]
            Single or multiple (object, field) specifications
        objects : Dict[str, ak.Array]
            Object dictionary

        Returns
        -------
        List[ak.Array]
            Target arrays
        """
        specs = target_spec if isinstance(target_spec, list) else [target_spec]

        targets = []
        for object_name, field_name in specs:
            try:
                targets.append(objects[object_name][field_name])
            except KeyError:
                logger.error(
                    f"Field {object_name}.{field_name} needed for {function_name} "
                    "is not found in objects dictionary"
                )
                raise KeyError(
                    f"Missing target field: {object_name}.{field_name}, "
                    f"function: {function_name}"
                )

        return targets

    def _set_target_arrays(
        self,
        target_spec: Union[Tuple[str, str], List[Tuple[str, str]]],
        objects: Dict[str, ak.Array],
        new_values: Union[ak.Array, List[ak.Array]],
    ) -> None:
        """
        Update target arrays in object dictionary.

        Parameters
        ----------
        target_spec : Union[Tuple[str, str], List[Tuple[str, str]]]
            Single or multiple (object, field) specifications
        objects : Dict[str, ak.Array]
            Object dictionary to update
        new_values : Union[ak.Array, List[ak.Array]]
            New values to assign
        """
        specs = target_spec if isinstance(target_spec, list) else [target_spec]
        values = new_values if isinstance(new_values, list) else [new_values]

        for (obj_name, field_name), value in zip(specs, values):
            objects[obj_name][field_name] = value

    def apply_object_corrections(
        self,
        object_copies: Dict[str, ak.Array],
        corrections: List[Dict[str, Any]],
        direction: Literal["up", "down", "nominal"] = "nominal",
    ) -> Dict[str, ak.Array]:
        """
        Apply object-level corrections.

        Parameters
        ----------
        object_copies : Dict[str, ak.Array]
            Objects to correct
        corrections : List[Dict[str, Any]]
            Correction configurations
        direction : Literal["up", "down", "nominal"], optional
            Systematic direction (default: "nominal")

        Returns
        -------
        Dict[str, ak.Array]
            Corrected objects
        """
        for correction in corrections:
            if correction.type != "object":
                continue

            # Prepare arguments and targets
            args = get_function_arguments(
                correction.use,
                object_copies,
                function_name=f"correction::{correction.name}",
            )
            targets = self._get_target_arrays(
                correction.target,
                object_copies,
                function_name=f"correction::{correction.name}",
            )
            operation = correction.op
            transform = correction.transform
            key = correction.key

            # Determine direction mapping
            dir_map = correction.up_and_down_idx
            corr_direction = (
                dir_map[0]
                if direction == "up"
                else dir_map[1] if direction in ["up", "down"] else "nominal"
            )

            # Apply correction
            if correction.get("use_correctionlib", False):
                corrected_values = self.apply_correctionlib(
                    correction_name=correction.name,
                    correction_key=key,
                    direction=corr_direction,
                    correction_args=args,
                    target=targets,
                    operation=operation,
                    transform=transform,
                )
            else:
                syst_func = correction.get(f"{direction}_function")
                if syst_func:
                    corrected_values = self.apply_syst_function(
                        syst_name=correction.name,
                        syst_function=syst_func,
                        function_args=args,
                        affected_arrays=targets,
                        operation=operation,
                    )
                else:
                    corrected_values = targets

            # Update objects
            self._set_target_arrays(
                correction.target, object_copies, corrected_values
            )

        return object_copies

    def apply_event_weight_correction(
        self,
        weights: ak.Array,
        systematic: Dict[str, Any],
        direction: Literal["up", "down"],
        object_copies: Dict[str, ak.Array],
    ) -> ak.Array:
        """
        Apply event-level weight correction.

        Parameters
        ----------
        weights : ak.Array
            Original event weights
        systematic : Dict[str, Any]
            Systematic configuration
        direction : Literal["up", "down"]
            Systematic direction
        object_copies : Dict[str, ak.Array]
            Current object copies

        Returns
        -------
        ak.Array
            Corrected weights
        """
        if systematic.type != "event":
            return weights

        # Prepare arguments
        args = get_function_arguments(
            systematic.use,
            object_copies,
            function_name=f"systematic::{systematic.name}",
        )
        operation = systematic.op
        key = systematic.key
        transform = systematic.transform
        dir_map = systematic.up_and_down_idx
        corr_direction = dir_map[0] if direction == "up" else dir_map[1]

        # Apply correction
        if systematic.get("use_correctionlib", False):
            return self.apply_correctionlib(
                correction_name=systematic.name,
                correction_key=key,
                direction=corr_direction,
                correction_args=args,
                target=weights,
                operation=operation,
                transform=transform,
            )
        else:
            syst_func = systematic.get(f"{direction}_function")
            if syst_func:
                return self.apply_syst_function(
                    syst_name=systematic.name,
                    syst_function=syst_func,
                    function_args=args,
                    affected_arrays=weights,
                    operation=operation,
                )
            return weights

    def compute_ghost_observables(
        self,
        object_copies: Dict[str, ak.Array],
        use_jax: bool = False,
    ) -> Dict[str, ak.Array]:
        """
        Compute derived observables not present in the original dataset.

        Parameters
        ----------
        object_copies : Dict[str, ak.Array]
            Current object copies
        use_jax : bool, optional
            Whether JAX is being used (default: False)

        Returns
        -------
        Dict[str, ak.Array]
            Updated object copies with new observables
        """
        for ghost in self.config.ghost_observables:
            # Skip JAX-incompatible ghosts
            if not ghost.works_with_jax and use_jax:
                logger.warning(
                    "Skipping JAX-incompatible ghost observable: %s",
                    ghost.names,
                )
                continue

            logger.debug("Computing ghost observables: %s", ghost.names)
            args = get_function_arguments(
                ghost.use, object_copies, function_name=ghost.function.__name__
            )
            outputs = ghost.function(*args)

            # Normalize outputs to list
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            # Normalize names and collections
            names = (
                [ghost.names] if isinstance(ghost.names, str) else ghost.names
            )
            collections = (
                [ghost.collections] * len(names)
                if isinstance(ghost.collections, str)
                else ghost.collections
            )

            # Update object copies
            for value, name, collection in zip(outputs, names, collections):
                # Handle single-field records
                if (
                    isinstance(value, ak.Array)
                    and len(ak.fields(value)) == 1
                    and name in ak.fields(value)
                ):
                    value = value[name]

                # Update existing collection
                if collection in object_copies:
                    try:
                        object_copies[collection][name] = value
                    except ValueError as error:
                        logger.exception(
                            f"Failed to add field '{name}' to collection '{collection}'"
                        )
                        raise error
                # Create new collection
                else:
                    object_copies[collection] = ak.Array({name: value})

        return object_copies
