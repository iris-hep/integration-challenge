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

from intccms.schema import GoodObjectMasksConfig, ObjVar, Sys
from intccms.utils.functors import (
    GhostObservableExecutor,
    MaskExecutor,
    get_function_arguments,
)
from intccms.utils.output import OutputDirectoryManager

# -----------------------------
# Register backends
# -----------------------------
vector.register_awkward()

# -----------------------------
# Logging Configuration
# -----------------------------
logger = logging.getLogger("BaseAnalysis")

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

import sys


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
        output_manager: OutputDirectoryManager,
    ) -> None:
        """
        Initialize analysis with configuration for systematics, corrections,
        and channels.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with keys:
            - 'systematics': Systematic variations configuration (list or year-keyed dict)
            - 'corrections': Correction configurations (list or year-keyed dict)
            - 'channels': Analysis channel definitions
            - 'general': General settings including output directory
        output_manager : OutputDirectoryManager
            Centralized output directory manager (required)
        """
        self.config = config
        self.channels = config.channels
        self.output_manager = output_manager

        # Store corrections and systematics in original format
        # They can be either List[CorrectionConfig] or Dict[str, List[CorrectionConfig]]
        self._corrections_config = config.corrections
        self._systematics_config = config.systematics

        # Determine if year-keyed
        self._year_keyed_corrections = isinstance(self._corrections_config, dict)
        self._year_keyed_systematics = isinstance(self._systematics_config, dict)

        self.corrlib_evaluators = self._load_correctionlib()

    def _load_correctionlib(self) -> Dict[str, CorrectionSet]:
        """
        Load correctionlib JSON files into evaluators.

        For year-keyed corrections, loads each year's corrections with keys
        formatted as "{year}_{name}" to allow year-specific lookups.
        For flat list corrections, uses just the correction name as key.

        Returns
        -------
        Dict[str, CorrectionSet]
            Mapping of correction name (or year_name) to CorrectionSet evaluator
        """
        evaluators = {}

        def load_correction(correction, key_prefix: str = "") -> None:
            """Load a single correction into evaluators."""
            if not correction.use_correctionlib:
                return

            corr_name = correction.name
            file_path = correction.file
            eval_key = f"{key_prefix}{corr_name}" if key_prefix else corr_name

            if file_path.endswith(".json.gz"):
                with gzip.open(file_path, "rt") as file_handle:
                    evaluators[eval_key] = CorrectionSet.from_string(
                        file_handle.read().strip()
                    )
            elif file_path.endswith(".json"):
                evaluators[eval_key] = CorrectionSet.from_file(file_path)
            else:
                raise ValueError(
                    f"Unsupported correctionlib format: {file_path}. "
                    "Expected .json or .json.gz"
                )

        if self._year_keyed_corrections:
            # Year-keyed: load with year prefix
            for year, corrections in self._corrections_config.items():
                for correction in corrections:
                    load_correction(correction, key_prefix=f"{year}_")
        else:
            # Flat list: load with just name
            for correction in self._corrections_config:
                load_correction(correction)

        # Also load systematics that use correctionlib
        if self._year_keyed_systematics:
            for year, systematics in self._systematics_config.items():
                for syst in systematics:
                    load_correction(syst, key_prefix=f"{year}_")
        else:
            for syst in self._systematics_config:
                load_correction(syst)

        return evaluators

    def get_corrections_for_year(self, year: Optional[str]) -> List[Any]:
        """
        Get corrections for a specific year.

        Parameters
        ----------
        year : str or None
            Correction year (e.g., "2016preVFP", "2017", "2018").
            If None and corrections are year-keyed, returns empty list.

        Returns
        -------
        List[CorrectionConfig]
            Corrections for the specified year
        """
        if not self._year_keyed_corrections:
            return self._corrections_config

        if year is None:
            logger.warning("Year is None but corrections are year-keyed. Returning empty list.")
            return []

        if year not in self._corrections_config:
            logger.warning(f"Year '{year}' not found in corrections config. Available: {list(self._corrections_config.keys())}")
            return []

        return self._corrections_config[year]

    def get_systematics_for_year(self, year: Optional[str]) -> List[Any]:
        """
        Get systematics for a specific year.

        Parameters
        ----------
        year : str or None
            Correction year (e.g., "2016preVFP", "2017", "2018").
            If None and systematics are year-keyed, returns empty list.

        Returns
        -------
        List[SystematicConfig]
            Systematics for the specified year
        """
        if not self._year_keyed_systematics:
            return self._systematics_config

        if year is None:
            logger.warning("Year is None but systematics are year-keyed. Returning empty list.")
            return []

        if year not in self._systematics_config:
            logger.warning(f"Year '{year}' not found in systematics config. Available: {list(self._systematics_config.keys())}")
            return []

        return self._systematics_config[year]

    @staticmethod
    def get_weight_overrides(
        corrections: List[Any],
        obj_corr_name: str,
        direction: str,
    ) -> Dict[str, str]:
        """
        Build sys_value overrides for corrections sensitive to an object correction.

        Scans corrections for those declaring reruns_with templates matching
        the given object correction name. Convention: each template has the form
        "{direction}_<name>" where <name> is the object correction it applies to.

        Parameters
        ----------
        corrections : List[CorrectionConfig]
            All corrections to scan
        obj_corr_name : str
            Name of the object correction being varied (e.g. "jesAbsolute")
        direction : str
            Variation direction ("up" or "down")

        Returns
        -------
        Dict[str, str]
            Mapping of correction name to formatted sys_value override.
            E.g. {"btag_hf": "up_jesAbsolute"} when obj_corr_name="jesAbsolute",
            direction="up"
        """
        overrides = {}
        for corr in corrections:
            if not corr.reruns_with:
                continue
            for template in corr.reruns_with:
                target_name = template.replace("{direction}_", "")
                if target_name == obj_corr_name:
                    overrides[corr.name] = template.format(direction=direction)
                    break
        return overrides

    def get_corrlib_evaluator(self, name: str, year: Optional[str]) -> CorrectionSet:
        """
        Get correctionlib evaluator for a correction, handling year-keyed lookups.

        Parameters
        ----------
        name : str
            Correction name
        year : str or None
            Correction year for year-keyed configs

        Returns
        -------
        CorrectionSet
            The correction evaluator

        Raises
        ------
        KeyError
            If correction not found
        """
        if self._year_keyed_corrections or self._year_keyed_systematics:
            # Try year-prefixed key first
            if year:
                year_key = f"{year}_{name}"
                if year_key in self.corrlib_evaluators:
                    return self.corrlib_evaluators[year_key]

        # Fall back to name only (for flat configs or if year key not found)
        if name in self.corrlib_evaluators:
            return self.corrlib_evaluators[name]

        raise KeyError(f"Correction '{name}' not found. Available: {list(self.corrlib_evaluators.keys())}")

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
            executor = MaskExecutor(mask_config)
            obj_name = mask_config.object
            good_objects[obj_name] = executor.execute(
                object_copies, object_name=obj_name
            )

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

    def resolve_correction_args(
        self,
        args: List[Union[ObjVar, Sys, str, int, float]],
        events: Dict[str, ak.Array],
        sys_value: str,
    ) -> List[Any]:
        """
        Resolve correction args list to actual values.

        Parameters
        ----------
        args : List[Union[ObjVar, Sys, str, int, float]]
            Argument specification from correction config
        events : Dict[str, ak.Array]
            Event data (object collections)
        sys_value : str
            Systematic variation string to substitute for Sys marker

        Returns
        -------
        List[Any]
            Resolved arguments ready for correctionlib
        """
        resolved = []
        for arg in args:
            if isinstance(arg, Sys):
                resolved.append(sys_value)
            elif isinstance(arg, ObjVar):
                resolved.append(events[arg.obj][arg.field])
            else:
                resolved.append(arg)  # fixed value (str, int, float)
        return resolved

    def apply_correctionlib(
        self,
        correction: Dict[str, Any],
        events: Dict[str, ak.Array],
        direction: Literal["up", "down", "nominal"],
        target: Optional[Union[ak.Array, List[ak.Array]]] = None,
        year: Optional[str] = None,
        sys_value_override: Optional[str] = None,
    ) -> Union[ak.Array, List[ak.Array]]:
        """
        Apply correction using correctionlib.

        Parameters
        ----------
        correction : Dict[str, Any]
            Full correction configuration dict with 'args', 'key', 'transform', etc.
        events : Dict[str, ak.Array]
            Event data (object collections)
        direction : Literal["up", "down", "nominal"]
            Systematic direction
        target : Optional[Union[ak.Array, List[ak.Array]]], optional
            Target array(s) to modify
        year : Optional[str], optional
            Correction year for year-keyed configs
        sys_value_override : Optional[str], optional
            When set, bypasses direction-based sys_value resolution and uses
            this string directly for the Sys() marker. Used for JEC-context
            overrides (e.g. "up_jes" for btag inside JEC variation).

        Returns
        -------
        Union[ak.Array, List[ak.Array]]
            Corrected value(s)
        """
        correction_name = correction["name"]
        correction_key = correction["key"]
        operation = correction.get("op", "mult")
        transform_in = correction.get("transform_in")
        transform_out = correction.get("transform_out")
        reduce_op = correction.get("reduce")

        # Resolve systematic string: override takes precedence over direction
        if sys_value_override is not None:
            sys_value = sys_value_override
        elif direction == "up":
            sys_value = correction["up_and_down_idx"][0]
        elif direction == "down":
            sys_value = correction["up_and_down_idx"][1]
        else:  # nominal
            sys_value = correction.get("nominal_idx", "nominal")

        logger.info(
            "Applying correction: %s/%s (sys=%s) [year=%s]",
            correction_name,
            correction_key,
            sys_value,
            year,
        )

        # Resolve args: ObjVar -> event data, Sys -> sys_value, else pass through
        resolved_args = self.resolve_correction_args(
            correction["args"], events, sys_value
        )

        # Find indices of ObjVar args and store original arrays
        objvar_indices = [
            i for i, arg in enumerate(correction["args"])
            if isinstance(arg, ObjVar)
        ]
        original_data_arrays = [resolved_args[i] for i in objvar_indices]

        # Apply transform_in to modify inputs before evaluation
        if transform_in is not None:
            transformed = transform_in(*original_data_arrays)
            if not isinstance(transformed, tuple):
                transformed = (transformed,)
            for idx, val in zip(objvar_indices, transformed):
                resolved_args[idx] = val

        # Flatten jagged arrays (keep track of structure for unflattening)
        flat_args = []
        counts = None
        for arg in resolved_args:
            if isinstance(arg, ak.Array) and is_jagged(arg):
                flat_args.append(ak.flatten(arg))
                if counts is None:
                    counts = ak.num(arg)
            else:
                flat_args.append(arg)

        # Evaluate correction using year-aware lookup
        correction_set = self.get_corrlib_evaluator(correction_name, year)
        correction_evaluator = correction_set[correction_key]
        correction_values = correction_evaluator.evaluate(*flat_args)

        # Restore jagged structure if needed
        if counts is not None:
            correction_values = ak.unflatten(correction_values, counts)

        # Apply transform_out to process output (receives result + original arrays)
        if transform_out is not None:
            correction_values = transform_out(correction_values, *original_data_arrays)

        # Apply reduce if specified (jagged -> event-level)
        if reduce_op is not None and is_jagged(correction_values):
            if reduce_op == "prod":
                correction_values = ak.prod(correction_values, axis=1)
            elif reduce_op == "sum":
                correction_values = ak.sum(correction_values, axis=1)

        # Apply to target if provided
        if target is not None:
            if isinstance(target, list):
                return [
                    self._apply_operation(operation, t, correction_values)
                    for t in target
                ]
            return self._apply_operation(operation, target, correction_values)

        return correction_values

    def apply_syst_function(
        self,
        syst_name: str,
        syst_function: Callable[..., ak.Array],
        function_args: List[ak.Array],
        affected_arrays: Union[ak.Array, List[ak.Array]],
        operation: str,
        static_kwargs: Optional[Dict[str, Any]] = None,
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
            Positional arguments for the variation function
        static_kwargs : Optional[Dict[str, Any]]
            Static keyword arguments for the variation function
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
        kwargs = static_kwargs or {}
        variation = syst_function(*function_args, **kwargs)

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
        target_spec: Union[ObjVar, List[ObjVar]],
        objects: Dict[str, ak.Array],
        function_name: Optional[str] = "generic_target_function",
    ) -> List[ak.Array]:
        """
        Extract target arrays from object dictionary.

        Parameters
        ----------
        target_spec : Union[ObjVar, List[ObjVar]]
            Single or multiple ObjVar specifications
        objects : Dict[str, ak.Array]
            Object dictionary

        Returns
        -------
        List[ak.Array]
            Target arrays
        """
        specs = target_spec if isinstance(target_spec, list) else [target_spec]

        targets = []
        for spec in specs:
            try:
                targets.append(objects[spec.obj][spec.field])
            except KeyError:
                logger.error(
                    f"Field {spec.obj}.{spec.field} needed for {function_name} "
                    "is not found in objects dictionary"
                )
                raise KeyError(
                    f"Missing target field: {spec.obj}.{spec.field}, "
                    f"function: {function_name}"
                )

        return targets

    def _set_target_arrays(
        self,
        target_spec: Union[ObjVar, List[ObjVar]],
        objects: Dict[str, ak.Array],
        new_values: Union[ak.Array, List[ak.Array]],
    ) -> None:
        """
        Update target arrays in object dictionary.

        Parameters
        ----------
        target_spec : Union[ObjVar, List[ObjVar]]
            Single or multiple ObjVar specifications
        objects : Dict[str, ak.Array]
            Object dictionary to update
        new_values : Union[ak.Array, List[ak.Array]]
            New values to assign
        """
        specs = target_spec if isinstance(target_spec, list) else [target_spec]
        values = new_values if isinstance(new_values, list) else [new_values]

        for spec, value in zip(specs, values):
            objects[spec.obj][spec.field] = value

    def apply_object_corrections(
        self,
        object_copies: Dict[str, ak.Array],
        corrections: List[Dict[str, Any]],
        direction: Literal["up", "down", "nominal"] = "nominal",
        year: Optional[str] = None,
        sys_value_override: Optional[str] = None,
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
        year : Optional[str], optional
            Correction year for year-keyed configs
        sys_value_override : Optional[str], optional
            Override for Sys() marker, bypassing direction-based resolution.

        Returns
        -------
        Dict[str, ak.Array]
            Corrected objects
        """
        for correction in corrections:
            if correction.type != "object":
                continue

            # Get target arrays
            targets = self._get_target_arrays(
                correction.target,
                object_copies,
                function_name=f"correction::{correction.name}",
            )

            # Apply corrections
            if correction.get("use_correctionlib", False):
                corrected_values = self.apply_correctionlib(
                    correction=correction,
                    events=object_copies,
                    direction=direction,
                    target=targets,
                    year=year,
                    sys_value_override=sys_value_override,
                )
            else:
                # Non-correctionlib path (custom function)
                syst_func = correction.get(f"{direction}_function")
                if syst_func:
                    # Extract ObjVar arrays for function args
                    func_args = [
                        object_copies[arg.obj][arg.field]
                        for arg in correction.get("args", [])
                        if isinstance(arg, ObjVar)
                    ]
                    corrected_values = self.apply_syst_function(
                        syst_name=correction.name,
                        syst_function=syst_func,
                        function_args=func_args,
                        affected_arrays=targets,
                        operation=correction.get("op", "mult"),
                        static_kwargs=correction.get("static_kwargs"),
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
        correction: Dict[str, Any],
        direction: Literal["up", "down", "nominal"],
        events: Dict[str, ak.Array],
        year: Optional[str] = None,
        sys_value_override: Optional[str] = None,
    ) -> ak.Array:
        """
        Apply event-level weight correction.

        Parameters
        ----------
        weights : ak.Array
            Original event weights
        correction : Dict[str, Any]
            Correction configuration with 'args', 'key', etc.
        direction : Literal["up", "down", "nominal"]
            Systematic direction
        events : Dict[str, ak.Array]
            Event data (object collections)
        year : Optional[str], optional
            Correction year for year-keyed configs
        sys_value_override : Optional[str], optional
            Override for Sys() marker, bypassing direction-based resolution.

        Returns
        -------
        ak.Array
            Corrected weights
        """
        if correction.type != "event":
            return weights

        # Apply correction using correctionlib
        if correction.get("use_correctionlib", False):
            return self.apply_correctionlib(
                correction=correction,
                events=events,
                direction=direction,
                target=weights,
                year=year,
                sys_value_override=sys_value_override,
            )
        else:
            # Non-correctionlib path (custom function)
            syst_func = correction.get(f"{direction}_function")
            if syst_func:
                # Extract ObjVar arrays for function args
                func_args = [
                    events[arg.obj][arg.field]
                    for arg in correction.get("args", [])
                    if isinstance(arg, ObjVar)
                ]
                return self.apply_syst_function(
                    syst_name=correction.name,
                    syst_function=syst_func,
                    function_args=func_args,
                    affected_arrays=weights,
                    operation=correction.get("op", "mult"),
                    static_kwargs=correction.get("static_kwargs"),
                )
            return weights

    def compute_ghost_observables(
        self,
        object_copies: Dict[str, ak.Array],
    ) -> Dict[str, ak.Array]:
        """
        Compute derived observables not present in the original dataset.

        Parameters
        ----------
        object_copies : Dict[str, ak.Array]
            Current object copies

        Returns
        -------
        Dict[str, ak.Array]
            Updated object copies with new observables
        """
        for ghost in self.config.ghost_observables:
            logger.debug("Computing ghost observables: %s", ghost.names)

            # Normalize names and collections for executor
            names = [ghost.names] if isinstance(ghost.names, str) else ghost.names
            collections = (
                [ghost.collections] * len(names)
                if isinstance(ghost.collections, str)
                else ghost.collections
            )

            executor = GhostObservableExecutor(ghost)
            object_copies = executor.execute(
                object_copies, field_names=names, collections=collections
            )

        return object_copies
