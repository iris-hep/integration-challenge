"""
Pydantic schemas for validating the analysis configuration.
"""

from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import Field, model_validator, field_validator

from intccms.schema.base import FunctorConfig, ObjVar, SubscriptableModel, Sys
from intccms.utils.binning import validate_binning_spec, binning_to_edges


class MetricsConfig(SubscriptableModel):
    """Performance benchmarking configuration.

    Controls performance metrics collection, worker tracking, and benchmark
    measurement persistence. All features are opt-in and have minimal overhead
    when enabled (~0.2%), zero overhead when disabled.

    Attributes
    ----------
    enable : bool
        Master switch for all metrics collection. When False, no overhead.
    track_workers : bool
        Enable background thread tracking worker count every `worker_tracking_interval` seconds.
        Required for core efficiency calculations.
    worker_tracking_interval : float
        Seconds between worker count samples (default: 1.0 second).
    save_measurements : bool
        Save complete benchmark measurement to disk for later reanalysis.
        Creates timestamped directory with task results, timing, worker timeline, etc.
    generate_plots : bool
        Auto-generate performance visualization plots (worker timeline, runtime distributions, etc.).
    generate_reports : bool
        Auto-generate human-readable summary reports matching idap-200gbps format.
    measurement_name : Optional[str]
        Custom name for measurement directory. If None, uses timestamp.
    target_throughput_gbps : float
        Target throughput for comparison (default: 200.0 Gbps for idap-200gbps target).
    """

    enable: Annotated[
        bool,
        Field(
            default=True,
            description="Enable performance metrics collection. When False, zero overhead.",
        ),
    ]
    track_workers: Annotated[
        bool,
        Field(
            default=True,
            description="Track worker count over time in background thread. "
            "Required for core efficiency calculations.",
        ),
    ]
    worker_tracking_interval: Annotated[
        float,
        Field(
            default=1.0,
            description="Seconds between worker count samples (default: 1.0).",
            gt=0.0,
        ),
    ]
    save_measurements: Annotated[
        bool,
        Field(
            default=True,
            description="Save complete measurement to disk for later reanalysis.",
        ),
    ]
    generate_plots: Annotated[
        bool,
        Field(
            default=True,
            description="Auto-generate performance visualization plots.",
        ),
    ]
    generate_reports: Annotated[
        bool,
        Field(
            default=True,
            description="Auto-generate human-readable summary reports.",
        ),
    ]
    measurement_name: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Custom measurement name. If None, uses timestamp YYYY-MM-DD_HH-MM-SS.",
        ),
    ]
    target_throughput_gbps: Annotated[
        float,
        Field(
            default=200.0,
            description="Target throughput for comparison (e.g., 200 Gbps for idap-200gbps).",
            gt=0.0,
        ),
    ]


class GeneralConfig(SubscriptableModel):
    lumi: Annotated[float, Field(description="Integrated luminosity in /pb")]
    weight_branch: Annotated[
        str, Field(description="Branch name for event weight")
    ]
    analysis: Annotated[
        Optional[str],
        Field(default="nondiff",
              description="The analysis mode to run: 'nondiff' or 'skip' (skim-only mode)."),
    ]
    use_skimmed_input: Annotated[
        bool,
        Field(
            default=False,
            description="If True, read from pre-skimmed files instead of original NanoAOD. "
            "Automatically builds fileset from skimmed_dir.",
        ),
    ]
    save_skimmed_output: Annotated[
        bool,
        Field(
            default=True,
            description="If True, save filtered events to disk in skimmed_dir. "
            "Controls output behavior independent of input source.",
        ),
    ]
    run_processor: Annotated[
        bool,
        Field(
            default=True,
            description="If True, run the coffea processor over data. If False, load previously saved histograms from disk.",
        ),
    ]
    run_analysis: Annotated[
        bool,
        Field(
            default=True,
            description="If True, run the analysis step (object selection, corrections, observables).",
        ),
    ]
    run_histogramming: Annotated[
        bool,
        Field(
            default=True,
            description="If True run the histogramming step for the "
            "non-differentiable analysis.",
        ),
    ]
    run_statistics: Annotated[
        bool,
        Field(
            default=True,
            description="If True run the statistical analysis "
            "(e.g. cabinetry fit) in non-differentiable analysis.",
        ),
    ]
    run_systematics: Annotated[
        bool,
        Field(
            default=True,
            description="If True, process systematic variations.",
        ),
    ]
    run_plots_only: Annotated[
        bool,
        Field(
            default=False,
            description="TODO: Not yet implemented. "
            "If True, load cached results and generate plots without re-running the analysis.",
        ),
    ]
    run_mva_training: Annotated[
        bool,
        Field(
            default=False,
            description="TODO: Not yet implemented. "
            "If True, run the MVA model pre-training step.",
        ),
    ]
    run_metadata_generation: Annotated[
        bool,
        Field(
            default=True,
            description="If True, run the JSON metadata generation step before constructing fileset.",
        ),
    ]
    read_from_cache: Annotated[
        bool,
        Field(
            default=True,
            description="TODO: Not yet implemented. "
            "If True, read preprocessed data from the cache directory if available.",
        ),
    ]
    output_dir: Annotated[
        Optional[str],
        Field(
            default="output/",
            description="Root directory for all analysis outputs "
            "(plots, models, histograms, statistics, etc.).",
        ),
    ]
    cache_dir: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Cache directory for intermediate products of the analysis. "
            "If None, uses system temp directory with 'graep' subdirectory.",
        ),
    ]
    metadata_dir: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Directory containing existing metadata JSON files. "
            "If None, uses output_dir/metadata/ and creates if needed.",
        ),
    ]
    skimmed_dir: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Directory containing existing skimmed ROOT files. "
            "If None, uses output_dir/skimmed/ and creates if needed.",
        ),
    ]
    processes: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description="If specified, limit the analysis to this list "
            "of process names.",
        ),
    ]
    channels: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description="If specified, limit the analysis to this list "
            "of channel names.",
        ),
    ]
    metrics: Annotated[
        MetricsConfig,
        Field(
            default_factory=MetricsConfig,
            description="Performance benchmarking configuration. "
            "Controls metrics collection, worker tracking, and measurement persistence.",
        ),
    ]

    @model_validator(mode="after")
    def validate_general(self) -> "GeneralConfig":
        """Validate the general configuration settings."""
        if self.analysis not in ["nondiff", "skip"]:
            raise ValueError(
                f"Invalid analysis mode '{self.analysis}'. Must be 'nondiff' or 'skip'."
            )

        return self


class GoodObjectMasksConfig(FunctorConfig):
    object: Annotated[
        str,
        Field(
            description="The object collection to which this mask applies "
            "(e.g. 'Jet')."
        ),
    ]

    @model_validator(mode="after")
    def validate_fields(self) -> "GoodObjectMasksConfig":
        """Validate that the object is a recognised type."""
        if self.object not in ["Muon", "Jet", "FatJet"]:
            raise ValueError(
                f"Invalid object '{self.object}'. Must be one of "
                f"'Muon' 'Jet' or 'FatJet'."
            )

        return self


class GoodObjectMasksBlockConfig(SubscriptableModel):
    """Configuration block for defining 'good' object masks."""

    analysis: Annotated[
        List[GoodObjectMasksConfig],
        Field(description="Masks for the main physics analysis branch."),
    ]
    mva: Annotated[
        List[GoodObjectMasksConfig],
        Field(description="Masks for the MVA training data branch."),
    ]


class ObservableConfig(FunctorConfig):
    name: Annotated[str, Field(description="Name of the observable")]
    binning: Annotated[
        Union[str, List[float]],
        Field(
            description="Histogram binning, specified as a 'low,high,nbins' string "
            + "or a list of explicit bin edges. Parsed to array during validation."
        ),
    ]
    label: Annotated[
        Optional[str],
        Field(
            default="observable",
            description="A LaTeX-formatted string for plot axis labels.",
        ),
    ]

    @field_validator("binning", mode="before")
    @classmethod
    def validate_and_parse_binning(cls, v: Union[str, List[float]]) -> np.ndarray:
        """Validate and parse binning specification to array of edges."""
        validate_binning_spec(v)  # Validate first
        return binning_to_edges(v)  # Then convert to edges


class GhostObservable(FunctorConfig):
    """Represents a derived quantity computed once and attached to the event record."""

    names: Annotated[
        Union[str, List[str]],
        Field(description="Name(s) of the computed observable(s)."),
    ]
    collections: Annotated[
        Union[str, List[str]],
        Field(
            description="The collection(s) to which the "
            + "new observable(s) should be attached."
        ),
    ]


class ChannelConfig(SubscriptableModel):
    name: Annotated[str, Field(description="Name of the analysis channel")]
    observables: Annotated[
        List[ObservableConfig],
        Field(
            description="A list of observable configurations for this channel."
        ),
    ]
    fit_observable: Annotated[str, Field]
    selection: Annotated[
        Optional[FunctorConfig],
        Field(
            default=None,
            description="Event selection function for this channel. "
            + "If None, all events are selected. Function must "
            + "return a PackedSelection object.",
        ),
    ]
    use_in_diff: Annotated[
        Optional[bool],
        Field(
            default=False,
            description="Whether to use this channel in differentiable analysis. "
            + "If None, defaults to True.",
        ),
    ]

    @model_validator(mode="after")
    def validate_fields(self) -> "ChannelConfig":
        """Validate channel configuration fields."""
        if self.selection and self.selection.function and not self.selection.use:
            raise ValueError(
                "If 'selection.function' is provided, 'selection.use' must also "
                + "be specified."
            )
        if not self.observables:
            raise ValueError("Each channel must have at least one observable.")

        obs_names = [obs.name for obs in self.observables]
        if self.fit_observable not in obs_names:
            raise ValueError(
                f"'fit_observable'='{self.fit_observable}' is not in the list of "
                + f"observables: {sorted(obs_names)}"
            )

        if len(set(obs_names)) != len(obs_names):
            raise ValueError(
                "Duplicate observable names found in the channel configuration."
            )

        return self


class UncertaintySourceConfig(SubscriptableModel):
    """Uncertainty source on a correction.

    Supports both correctionlib (sys strings) and custom functions (up/down callables).
    Correctionlib sources override the parent correction's sys string.
    Custom-function sources provide up/down callables for object-level variations (e.g. JEC).
    """

    name: Annotated[str, Field(description="Name of the systematic variation")]
    up_and_down_idx: Annotated[
        List[str],
        Field(
            default=["up", "down"],
            description="Sys strings for [up, down] directions. For correctionlib "
            "sources, overrides the parent correction's sys string during evaluation. "
            "For custom-function sources, defaults to ['up', 'down'].",
        ),
    ]
    args: Annotated[
        Optional[List[Union[ObjVar, Sys, str, int, float]]],
        Field(
            default=None,
            description="Source-specific args. Falls back to parent correction's args "
            "when None. Needed when source functions take different inputs than the "
            "correction's nominal function (e.g. JEC uncertainty vs JEC nominal).",
        ),
    ]
    is_delta: Annotated[
        bool,
        Field(
            default=False,
            description="If True, the variation function is applied on top of the "
            "already-applied nominal (e.g. JEC uncertainty as multiplicative delta). "
            "If False (default), the variation replaces the nominal entirely.",
        ),
    ]
    up_function: Annotated[
        Optional[Callable],
        Field(
            default=None,
            description="Custom function for up variation (non-correctionlib path).",
        ),
    ]
    down_function: Annotated[
        Optional[Callable],
        Field(
            default=None,
            description="Custom function for down variation (non-correctionlib path).",
        ),
    ]


class CorrectionConfig(SubscriptableModel):
    """Configuration for a single correction (e.g., scale factor, pileup weight).

    The `args` field specifies the exact argument order for correctionlib calls:
    - ObjVar(obj, field): resolved from event data
    - Sys(): marker for systematic variation string insertion
    - str/int/float: passed through as fixed values (year, working point, etc.)
    """

    name: Annotated[str, Field(description="Name of the correction")]
    type: Annotated[
        Literal["event", "object"],
        Field(description="Whether correction is event/object-level"),
    ]
    args: Annotated[
        List[Union[ObjVar, Sys, str, int, float]],
        Field(
            description="Ordered arguments for correctionlib. Use ObjVar for event data, "
            "Sys() for systematic position, strings/numbers for fixed values."
        ),
    ]
    op: Annotated[
        Optional[Literal["mult", "add", "subtract"]],
        Field(
            default="mult",
            description="How to apply correction to targets",
        ),
    ]
    key: Annotated[
        Optional[str],
        Field(default=None, description="Correctionlib key"),
    ]
    use_correctionlib: Annotated[
        bool,
        Field(
            default=True,
            description="True if using correctionlib to apply correction",
        ),
    ]
    file: Annotated[str, Field(description="Path to correction file")]
    transform_in: Annotated[
        Optional[Callable],
        Field(
            default=None,
            description="Transform inputs before evaluation. "
            "Receives ObjVar arrays in order, returns transformed arrays.",
        ),
    ]
    transform_out: Annotated[
        Optional[Callable],
        Field(
            default=None,
            description="Transform output after evaluation. "
            "Receives (result, *original_arrays), returns processed result.",
        ),
    ]
    reduce: Annotated[
        Optional[Literal["prod", "sum"]],
        Field(
            default=None,
            description="Reduction operation for per-object corrections. "
            "Reduces jagged array to event-level before applying to target.",
        ),
    ]
    nominal_idx: Annotated[
        str,
        Field(
            default="nominal",
            description="Systematic variation key for nominal direction",
        ),
    ]
    target: Annotated[
        Optional[Union[ObjVar, List[ObjVar]]],
        Field(default=None, description="Target (object, variable) to modify"),
    ]

    reruns_with: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description="List of sys-string templates this correction is sensitive to "
            "when object corrections vary. Each template has the form "
            "'{direction}_<object_correction_name>' where {direction} is replaced "
            "with 'up'/'down'. The part after '{direction}_' identifies which "
            "object correction triggers this override.",
        ),
    ]

    applies_to: Annotated[
        Literal["mc", "data", "both"],
        Field(
            default="mc",
            description="Whether this correction applies to MC, data, or both.",
        ),
    ]

    # Custom function fields (used when use_correctionlib=False)
    nominal_function: Annotated[
        Optional[Callable],
        Field(
            default=None,
            description="Custom function for nominal correction (non-correctionlib path).",
        ),
    ]
    uncertainty_sources: Annotated[
        Optional[List[UncertaintySourceConfig]],
        Field(
            default=None,
            description="Uncertainty sources for this correction. Each source produces "
            "alternative histograms. Correctionlib sources use up_and_down_idx to override "
            "the parent's sys string. Custom-function sources use up/down_function.",
        ),
    ]

    @model_validator(mode="after")
    def validate_corrections_fields(self) -> "CorrectionConfig":
        """Validate correction configuration fields."""
        if self.use_correctionlib:
            if not self.file:
                raise ValueError(
                    "If 'use_correctionlib' is True, 'file' must also be specified."
                )
            if not self.key:
                raise ValueError(
                    "If 'use_correctionlib' is True, 'key' must also be specified."
                )
            # Validate exactly one Sys marker in args
            sys_count = sum(1 for a in self.args if isinstance(a, Sys))
            if sys_count != 1:
                raise ValueError(
                    f"'args' must contain exactly one Sys() marker, found {sys_count}."
                )
        if self.type == "object":
            if not self.target:
                raise ValueError(
                    "If correction 'type' is 'object', 'target' must be specified."
                )
            if isinstance(self.target, ObjVar) and self.target.field is None:
                raise ValueError(
                    "If correction 'type' is 'object', "
                    "target field must not be None."
                )
        return self


class SystematicConfig(SubscriptableModel):
    name: Annotated[str, Field(description="Name of the systematic variation")]
    type: Annotated[
        Literal["event", "object"],
        Field(description="Whether variation is event/object-level"),
    ]
    up_function: Annotated[
        Optional[Callable],
        Field(default=None, description="Callable for 'up' variation"),
    ]
    down_function: Annotated[
        Optional[Callable],
        Field(default=None, description="Callable for 'down' variation"),
    ]
    transform: Annotated[
        Optional[Callable],
        Field(
            default=lambda *x: x,
            description="Optional function to apply transformation to inputs "
            + "before applying systematic variation",
        ),
    ]
    target: Annotated[
        Optional[Union[ObjVar, List[ObjVar]]],
        Field(default=None, description="Target (object, variable) to modify"),
    ]
    use: Annotated[
        Optional[Union[ObjVar, List[ObjVar]]],
        Field(
            default=[],
            description="(object, variable) inputs to variation functions. If variable "
            + "is None, object is passed.",
        ),
    ]
    symmetrise: Annotated[
        bool,
        Field(default=False, description="Whether to symmetrise variation"),
    ]
    op: Annotated[
        Literal["mult", "add", "subtract"],
        Field(
            description="How (operation) to apply systematic variation function "
            + "to targets"
        ),
    ]

    @model_validator(mode="after")
    def validate_functions_and_consistency(self) -> "SystematicConfig":
        """Validate systematic configuration fields."""
        if not self.up_function and not self.down_function:
            raise ValueError(
                f"Systematic '{self.name}' must define at least one of 'up_function' "
                + "or 'down_function'."
            )

        if self.type == "object":
            if not self.target:
                raise ValueError(
                    "If correction 'type' is 'object', 'target' must be specified."
                )

        return self


class StatisticalConfig(SubscriptableModel):
    cabinetry_config: Annotated[
        str,
        Field(
            description="Path to the YAML configuration file for the `cabinetry` \
                statistical tool (non-differentiable analysis).",
        ),
    ]


class PlottingConfig(SubscriptableModel):
    process_colors: Annotated[
        Optional[dict[str, str]],
        Field(default=None, description="Hex colors for each process key"),
    ]
    process_labels: Annotated[
        Optional[dict[str, str]],
        Field(
            default=None,
            description="LaTeX‐style legend labels for each process",
        ),
    ]
    process_order: Annotated[
        Optional[List[str]],
        Field(default=None, description="Draw/order sequence for processes"),
    ]
