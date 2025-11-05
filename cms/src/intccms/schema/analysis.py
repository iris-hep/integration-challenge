"""
Pydantic schemas for validating the analysis configuration.
"""

from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import Field, model_validator, field_validator

from intccms.schema.base import FunctorConfig, ObjVar, SubscriptableModel
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
    run_skimming: Annotated[
        bool,
        Field(
            default=True,
            description="DEPRECATED: Use save_skimmed_output instead. "
            "If True, run the initial NanoAOD skimming and filtering step.",
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
            description="If True load cached results and generate plots "
            "without re-running the analysis.",
        ),
    ]
    run_mva_training: Annotated[
        bool,
        Field(
            default=False,
            description="If True, run the MVA model pre-training step.",
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
            description="If True, read preprocessed data from the cache directory "
            "if available.",
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


class GoodObjectMasksBlockConfig(SubscriptableModel):
    """Configuration block for defining 'good' object masks."""


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


class CorrectionConfig(SubscriptableModel):
    name: Annotated[str, Field(description="Name of the correction")]
    type: Annotated[
        Literal["event", "object"],
        Field(description="Whether correction is event/object-level"),
    ]
    use: Annotated[
        Optional[Union[ObjVar, List[ObjVar]]],
        Field(default=[], description="(object, variable) inputs"),
    ]
    op: Annotated[
        Optional[Literal["mult", "add", "subtract"]],
        Field(
            default="mult",
            description="How (operationa) to apply correction to targets",
        ),
    ]
    key: Annotated[
        Optional[str],
        Field(default=None, description="Correctionlib key (optional)"),
    ]
    use_correctionlib: Annotated[
        bool,
        Field(
            default=True,
            description="True if using correctionlib to apply correction",
        ),
    ]
    file: Annotated[str, Field(description="Path to correction file")]
    transform: Annotated[
        Optional[Callable],
        Field(
            default=lambda *x: x,
            description="Optional function to apply transformation to inputs "
            + "before applying correction",
        ),
    ]
    up_and_down_idx: Annotated[
        Optional[List[str]],
        Field(
            default=["up", "down"],
            description="Systematic variation keys (optional)",
        ),
    ]
    target: Annotated[
        Optional[Union[ObjVar, List[ObjVar]]],
        Field(default=None, description="Target (object, variable) to modify"),
    ]


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
