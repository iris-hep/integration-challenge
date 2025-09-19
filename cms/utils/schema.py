"""
Pydantic schemas for validating the analysis configuration.

This module defines a set of Pydantic models that correspond to the structure
of the main analysis configuration dictionary. It ensures that the configuration
is well-formed, all required fields are present, and values have the correct types
before the analysis runs. This provides type safety and clear error messages for
invalid configurations.
"""

import copy
from enum import Enum
from typing import Annotated, Callable, List, Literal, Optional, Tuple, Union

from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, Field, model_validator


# Type alias for (object, variable) pairs
ObjVar = Tuple[str, Optional[str]]


class SubscriptableModel(BaseModel):
    """A Pydantic BaseModel that supports dictionary-style item access."""

    def __getitem__(self, key):
        """Allows dictionary-style `model[key]` access."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Allows dictionary-style `model[key] = value` assignment."""
        return setattr(self, key, value)

    def __contains__(self, key):
        """Allows `key in model` checks."""
        return hasattr(self, key)

    def get(self, key, default=None):
        """Allows `.get(key, default)` method."""
        return getattr(self, key, default)


class FunctorConfig(SubscriptableModel):
    function: Annotated[
        Callable,
        Field(description="A Python callable to be executed."),
    ]
    use: Annotated[
        Optional[List[ObjVar]],
        Field(
            default=None,
            description="A list of (object variable) tuples specifying "
            "the inputs for the function.",
        ),
    ]


class GoodObjectMasksConfig(SubscriptableModel):
    object: Annotated[
        str,
        Field(
            description="The object collection to which this mask applies "
            "(e.g. 'Jet')."
        ),
    ]
    function: Annotated[
        Callable,
        Field(
            description="A callable that takes object collections and "
            "returns a boolean mask."
        ),
    ]
    use: Annotated[
        List[ObjVar],
        Field(
            description="A list of (object variable) tuples specifying "
            "the inputs for the mask function."
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


# ------------------------
# General configuration
# ------------------------
class GeneralConfig(SubscriptableModel):
    lumi: Annotated[float, Field(description="Integrated luminosity in /pb")]
    weight_branch: Annotated[
        str, Field(description="Branch name for event weight")
    ]
    lumifile: Annotated[
        str,
        Field(
            description="Path to JSON file with good luminosity sections",
        ),
    ]
    analysis: Annotated[
        Optional[str],
        Field(default="nondiff",
              description="The analysis mode to run: 'diff' (differentiable), 'nondiff', 'both', or 'skip' (skim-only mode)."),
    ]
    run_skimming: Annotated[
        bool,
        Field(
            default=True,
            description="If True, run the initial NanoAOD skimming and filtering step.",
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

    @model_validator(mode="after")
    def validate_general(self) -> "GeneralConfig":
        """Validate the general configuration settings."""
        if self.analysis not in ["diff", "nondiff", "both", "skip"]:
            raise ValueError(
                f"Invalid analysis mode '{self.analysis}'. Must be 'diff', 'nondiff', 'both', or 'skip'."
            )

        return self


# ------------------------
# JAX configuration
# ------------------------
class JaxConfig(SubscriptableModel):
    soft_selection: Annotated[
        FunctorConfig,
        Field(
            description="The differentiable selection function. It "
            "should return a per-event weight."
        ),
    ]
    params: Annotated[
        dict[str, float],
        Field(
            description="A dictionary of all optimizable "
            + "parameters and their initial values."
        ),
    ]
    optimise: Annotated[
        bool,
        Field(
            default=True,
            description="If True, run the gradient-based optimisation of parameters.",
        ),
    ]
    learning_rate: Annotated[
        float,
        Field(
            default=0.01,
            description="The default learning rate for the optimiser.",
        ),
    ]
    max_iterations: Annotated[
        Optional[int],
        Field(
            default=25,
            description="The number of optimisation steps to perform.",
        ),
    ]
    param_updates: Annotated[
        dict[str, Callable[[float, float], float]],
        Field(
            default_factory=dict,
            description="Optional per-parameter update/clamping rules. "
            + "Maps a parameter name to a callable that accepts "
            + "(current_value, update_delta) and returns "
            + "the new value. Example: "
            + "{'met_threshold': lambda x, d: jnp.clip(x + d, 20, 150)}",
        ),
    ]

    learning_rates: Annotated[
        Optional[dict[str, float]],
        Field(
            default=None,
            description="A dictionary of parameter-specific learning rates, \
                        overriding the default.",
        ),
    ]

    explicit_optimisation: Annotated[
        bool,
        Field(
            default=False,
            description="If True, use a manual optimisation loop instead of the \
                `jaxopt` solver.",
        ),
    ]


# ------------------------
# Dataset configuration
# ------------------------
class DatasetConfig(SubscriptableModel):
    """Configuration for individual dataset paths, cross-sections, and metadata"""
    name: Annotated[str, Field(description="Dataset name/identifier")]
    directory: Annotated[str, Field(description="Directory containing dataset files")]
    cross_section: Annotated[float, Field(description="Cross-section in picobarns")]
    file_pattern: Annotated[str, Field(default="*.root", description="File pattern for dataset files")]
    tree_name: Annotated[str, Field(default="Events", description="ROOT tree name")]
    weight_branch: Annotated[str, Field(default="genWeight", description="Branch name for event weights")]
    remote_access: Annotated[
        Optional[dict[str, str]],
        Field(default=None, description="Configuration for remote access (EOS, XRootD, etc.)")
    ]

class DatasetManagerConfig(SubscriptableModel):
    """Top-level dataset management configuration"""
    datasets: Annotated[List[DatasetConfig], Field(description="List of dataset configurations")]
    max_files: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Maximum number of files to process per dataset."
        ),
    ]

# ------------------------
# Skimming configuration
# ------------------------
class SkimmingConfig(SubscriptableModel):
    """Configuration for workitem-based skimming selections and output"""

    # Selection function - required for workitem-based skimming
    selection_function: Annotated[
        Callable,
        Field(description="Selection function that returns a PackedSelection object")
    ]

    # Selection inputs - required to specify what the function needs
    selection_use: Annotated[
        List[ObjVar],
        Field(description="List of (object, variable) tuples specifying inputs for the selection function")
    ]


    # File handling configuration
    chunk_size: Annotated[
        int,
        Field(default=100_000, description="Number of events to process per chunk (used for configuration compatibility)")
    ]
    tree_name: Annotated[
        str,
        Field(default="Events", description="ROOT tree name for input and output files")
    ]

# ------------------------
# Preprocessing configuration
# ------------------------
class PreprocessConfig(SubscriptableModel):
    branches: Annotated[
        dict[str, List[str]],
        Field(
            description="A mapping of collection names to a list of branches to keep."
        ),
    ]
    ignore_missing: Annotated[
        bool,
        Field(
            default=False, description="If True, missing branches are ignored."
        ),
    ]
    mc_branches: Annotated[
        dict[str, List[str]],
        Field(
            description="Additional branches to keep only for Monte Carlo samples."
        ),
    ]

    # Enhanced skimming configuration
    skimming: Annotated[
        Optional[SkimmingConfig],
        Field(default=None, description="Configuration for skimming selections and output")
    ]

    @model_validator(mode="after")
    def validate_branches(self) -> "PreprocessConfig":
        """Validate the branch configuration for duplicates and consistency."""
        # check for duplicate objects in branches
        if len(list(self.branches.keys())) != len(set(self.branches.keys())):
            raise ValueError("Duplicate objects found in branch list.")
        # check for duplicate objects in mc_branches
        if len(list(self.mc_branches.keys())) != len(
            set(self.mc_branches.keys())
        ):
            raise ValueError("Duplicate objects found in mc_branch list.")

        # Check for duplicate branches in the same object in branches
        for obj, obj_branches in self.branches.items():
            if len(obj_branches) != len(set(obj_branches)):
                raise ValueError(f"Duplicate branches found in '{obj}'.")

        # Check for duplicate branches in the same object in mc_branches
        for obj, obj_branches in self.mc_branches.items():
            if len(obj_branches) != len(set(obj_branches)):
                raise ValueError(f"Duplicate branches found in '{obj}'.")

        # check that MC branches are in branches
        for obj, obj_branches in self.mc_branches.items():
            if obj not in self.branches:
                raise ValueError(f"'{obj}' is not present in branches.")
            for br in obj_branches:
                if br not in self.branches[obj]:
                    raise ValueError(
                        f"'{br}' is not present in branches for '{obj}'."
                    )

        return self


# ------------------------
# Statistical analysis configuration
# ------------------------]
class StatisticalConfig(SubscriptableModel):
    cabinetry_config: Annotated[
        str,
        Field(
            description="Path to the YAML configuration file for the `cabinetry` \
                statistical tool (non-differentiable analysis).",
        ),
    ]


# ------------------------
# Observable configuration
# ------------------------
class ObservableConfig(SubscriptableModel):
    name: Annotated[str, Field(description="Name of the observable")]
    binning: Annotated[
        Union[str, List[float]],
        Field(
            description="Histogram binning, specified as a 'low,high,nbins' string "
            + "or a list of explicit bin edges."
        ),
    ]
    function: Annotated[
        Callable,
        Field(
            description="A callable that computes the "
            + "observable values from event data."
        ),
    ]
    use: Annotated[
        List[ObjVar],
        Field(
            description="A list of (object, variable) tuples specifying the inputs \
                for the function.",
        ),
    ]
    label: Annotated[
        Optional[str],
        Field(
            default="observable",
            description="A LaTeX-formatted string for plot axis labels.",
        ),
    ]
    works_with_jax: Annotated[
        bool,
        Field(
            default=True,
            description="If True, the function is compatible with the JAX backend "
            "for differentiable analysis.",
        ),
    ]

    @model_validator(mode="after")
    def validate_binning(self) -> "ObservableConfig":
        """Validate the binning format and values."""
        if isinstance(self.binning, str):
            self.binning = (
                self.binning.strip("[").strip("]").strip("(").strip(")")
            )
            binning = self.binning.split(",")
            if len(binning) != 3:
                raise ValueError(
                    f"Invalid binning string: {self.binning}. Need 3 values."
                    + "Expected format: 'low,high,nbins'"
                )
            else:
                try:
                    low, high, nbins = map(float, binning)
                    nbins = int(nbins)
                except ValueError:
                    raise ValueError(
                        f"Invalid binning string: {self.binning}. Need 3 floats."
                        + "Expected format: 'low,high,nbins'"
                    )
                if low >= high:
                    raise ValueError(
                        f"Invalid binning string: {self.binning}. Low must be < high."
                        + "Expected format: 'low,high,nbins'"
                    )
                if nbins <= 0 or not isinstance(nbins, int):
                    raise ValueError(
                        f"Invalid binning string: {self.binning}. nbins must be != 0."
                        + "Expected format: 'low,high,nbins'"
                    )

        elif isinstance(self.binning, list):
            if len(self.binning) < 2:
                raise ValueError("At least two bin edges required.")
            if any(b <= a for a, b in zip(self.binning, self.binning[1:])):
                raise ValueError("Binning edges must be strictly increasing.")
        return self


# ------------------------
# Ghost observable configuration
# ------------------------
class GhostObservable(SubscriptableModel):
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
    function: Annotated[
        Callable,
        Field(description="A callable that computes the ghost observables."),
    ]
    use: Annotated[
        List[ObjVar],
        Field(
            description="A list of (object, variable) tuples "
            + "specifying the inputs for the function."
        ),
    ]
    works_with_jax: Annotated[
        bool,
        Field(
            default=True,
            description="If True, the function is compatible with the JAX backend.",
        ),
    ]


# ------------------------
# Channel configuration
# ------------------------
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
        if self.selection.function and not self.selection.use:
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


# ------------------------
# Corrections configuration
# ------------------------
# should this just be restricted to correctionlib
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

    @model_validator(mode="after")
    def validate_corrections_fields(self) -> "CorrectionConfig":
        if self.use_correctionlib:
            if not self.file:
                raise ValueError(
                    "If 'use_correctionlib' is True, 'file' must also be specified."
                )
            if not self.key:
                raise ValueError(
                    "If 'use_correctionlib' is True, 'key' must also be specified."
                )
        if self.type == "object":
            if not self.target:
                raise ValueError(
                    "If correction 'type' is 'object', 'target' must be specified."
                )
            if self.target[2] is None:
                raise ValueError(
                    "If correction 'type' is 'object', \
                        target variable must not be None."
                )
        return self


# ------------------------
# Systematics configuration
# ------------------------
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


class PlottingJaxConfig(SubscriptableModel):
    aux_param_labels: Annotated[
        Optional[dict[str, str]],
        Field(
            default=None,
            description="LaTeX labels for auxiliary parameters in JAX‐scan plots",
        ),
    ]
    fit_param_labels: Annotated[
        Optional[dict[str, str]],
        Field(
            default=None,
            description="LaTeX labels for fit parameters in JAX‐scan plots",
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
    jax: Annotated[
        Optional[PlottingJaxConfig],
        Field(default=None, description="JAX‐specific label overrides"),
    ]


# =================================
# MVA configuration
# =================================


# ========
# Activation functions
# ========
class ActivationKey(str, Enum):
    relu = "relu"
    tanh = "tanh"
    sigmoid = "sigmoid"


# ========
# Gradient optimisation configuration
# ========
class GradOptimConfig(SubscriptableModel):
    optimise: Annotated[
        bool,
        Field(
            default=True,
            description="Include this MVA’s weights in the global optimisation",
        ),
    ]
    learning_rate: Annotated[
        float,
        Field(
            default=1e-3,
            description="Learning rate for this MVA when optimise=True",
        ),
    ]
    log_param_changes: Annotated[
        bool,
        Field(
            default=False,
            description="If True, log the mean of weight/bias changes \
                during optimisation.",
        ),
    ]


# ========
# Network layers configuration
# ========
class LayerConfig(SubscriptableModel):
    ndim: Annotated[
        int, Field(..., description="Output dimension of this layer")
    ]
    activation: Annotated[
        Union[Callable, ActivationKey],
        Field(
            ...,
            description=(
                "For framework='jax', a Python callable; "
                "for TF/Keras, one of the ActivationKey enums"
            ),
        ),
    ]
    weights: Annotated[
        str,
        Field(
            ...,
            description="Parameter name for the weight tensor in this layer",
        ),
    ]
    bias: Annotated[
        str,
        Field(
            ..., description="Parameter name for the bias vector in this layer"
        ),
    ]


# ========
# Features to train the MVA on
# ========
class FeatureConfig(SubscriptableModel):
    name: Annotated[str, Field(..., description="Feature name")]
    label: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Optional label for plots (e.g. LaTeX string)",
        ),
    ]
    function: Annotated[
        Callable,
        Field(
            ...,
            description="Callable extracting the raw feature \
                (e.g. lambda mva: mva.n_jet)",
        ),
    ]
    use: Annotated[
        List[ObjVar],
        Field(
            ...,
            description="(object, variable) pairs to pass into function, \
                e.g. [('mva', None)]",
        ),
    ]
    scale: Annotated[
        Optional[Callable],
        Field(
            default=None,
            description="Optional callable to scale the extracted feature",
        ),
    ]
    binning: Annotated[
        Optional[Union[str, List[float]]],
        Field(
            default=None,
            description=(
                "Optional histogramming binning for diagnostics: "
                "either 'low,high,nbins' or explicit edge list"
            ),
        ),
    ]

    @model_validator(mode="after")
    def validate_binning(self) -> "FeatureConfig":
        """Validate the binning format and values."""
        if isinstance(self.binning, str):
            self.binning = (
                self.binning.strip("[").strip("]").strip("(").strip(")")
            )
            binning = self.binning.split(",")
            if len(binning) != 3:
                raise ValueError(
                    f"Invalid binning string: {self.binning}. Need 3 values."
                    + "Expected format: 'low,high,nbins'"
                )
            else:
                try:
                    low, high, nbins = map(float, binning)
                    nbins = int(nbins)
                except ValueError:
                    raise ValueError(
                        f"Invalid binning string: {self.binning}. Need 3 floats."
                        + "Expected format: 'low,high,nbins'"
                    )
                if low >= high:
                    raise ValueError(
                        f"Invalid binning string: {self.binning}. Low must be < high."
                        + "Expected format: 'low,high,nbins'"
                    )
                if nbins <= 0 or not isinstance(nbins, int):
                    raise ValueError(
                        f"Invalid binning string: {self.binning}. nbins must be != 0."
                        + "Expected format: 'low,high,nbins'"
                    )

        elif isinstance(self.binning, list):
            if len(self.binning) < 2:
                raise ValueError("At least two bin edges required.")
            if any(b <= a for a, b in zip(self.binning, self.binning[1:])):
                raise ValueError("Binning edges must be strictly increasing.")
        return self


# ========
# Full MVA configuration
# ========
class MVAConfig(SubscriptableModel):
    name: Annotated[
        str, Field(..., description="Unique name for this neural network")
    ]
    use_in_diff: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to include this MVA in the global JAX gradient",
        ),
    ]  # is this useful?
    framework: Annotated[
        Literal["jax", "keras", "tf"],
        Field(..., description="Framework to use for building/training"),
    ]
    # Global pre-training learning rate:
    learning_rate: Annotated[
        float,
        Field(
            default=0.01, description="Step size for pre-training the network"
        ),
    ]
    grad_optimisation: Annotated[
        GradOptimConfig,
        Field(
            default_factory=GradOptimConfig,
            description="Per-MVA optimisation settings",
        ),
    ]
    layers: Annotated[
        List[LayerConfig],
        Field(..., description="Sequential layer definitions"),
    ]
    loss: Annotated[
        Union[Callable, str],
        Field(
            ...,
            description=(
                "For 'jax': a Python callable (preds, features, targets)->scalar; "
                "for TF/Keras: string loss key (e.g. 'binary_crossentropy')"
            ),
        ),
    ]
    features: Annotated[
        List[FeatureConfig],
        Field(..., description="List of input features for this network"),
    ]

    classes: Annotated[
        List[Union[str, dict[str, Union[Tuple[str, ...], List[str]]]]],
        Field(
            ...,
            description=(
                "List of class definitions for training the classifier. Each entry"
                "can be:\n"
                "- a single process name (e.g. 'wjets'), or\n"
                "- a dictionary mapping a class label to multiple process names "
                "(e.g. {'ttbar': ['ttbar_semilep', 'ttbar_had', 'ttbar_lep']}).\n"
                "The index of each entry determines the class label."
            ),
        ),
    ]

    plot_classes: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description=(
                "Optional list of class names to use in MVA-related plotting."
                "For example, features/scores will be plotted for this set of classes."
                "If None, defaults to all classes in `classes`."
            ),
        ),
    ]
    balance_strategy: Annotated[
        Literal["none", "undersample", "oversample", "class_weight"],
        Field(
            default="undersample",
            description=(
                "How to rebalance classes before training:\n"
                "- `none`: leave as is\n"
                "- `undersample`: down-sample every class to the smallest class size\n"
                "- `oversample`: up-sample every class to the largest class size\n"
                "- `class_weight`: compute sample-weights inversely proportional "
                "to class frequency"
            ),
        ),
    ]
    random_state: Annotated[
        Optional[int],
        Field(
            default=42,
            description="Seed for any sampling shuffle (undersample/oversample)",
        ),
    ]

    # -------------------------------------------------------------------------
    # Training  parameters
    # -------------------------------------------------------------------------
    epochs: Annotated[
        int,
        Field(
            default=1000, description="Number of training epochs for this MVA"
        ),
    ]
    batch_size: Annotated[
        Optional[int],
        Field(
            default=32,
            description="Batch size for training; None for full-batch GD",
        ),
    ]
    validation_split: Annotated[
        float,
        Field(
            default=0.2, description="Fraction of data reserved for validation"
        ),
    ]
    log_interval: Annotated[
        int,
        Field(
            default=100,
            description="Epoch interval for logging training progress",
        ),
    ]

    @model_validator(mode="after")
    def check_framework_consistency(self) -> "MVAConfig":
        if self.framework == "jax":
            if not callable(self.loss):
                raise ValueError("JAX 'loss' must be a callable.")
            for L in self.layers:
                if not callable(L.activation):
                    raise ValueError(
                        "JAX 'activation' must be a Python callable."
                    )
        else:
            if not isinstance(self.loss, str):
                raise ValueError("TF/Keras 'loss' must be a string key.")
            for L in self.layers:
                if not isinstance(L.activation, ActivationKey):
                    raise ValueError(
                        f"TF/Keras 'activation' must be one of {list(ActivationKey)}."
                    )

        if self.plot_classes is None:
            # Default to all classes if not specified
            self.plot_classes = self.classes.keys()
        return self

    @model_validator(mode="after")
    def check_duplicate_features(self) -> "MVAConfig":
        names = [feat.name for feat in self.features]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(
                f"Duplicate feature names in MVA '{self.name}': {sorted(dup)}"
            )
        return self


# ------------------------
# Top-level configuration
# ------------------------
class Config(SubscriptableModel):
    general: Annotated[
        GeneralConfig, Field(description="Global settings for the analysis")
    ]
    jax: Annotated[
        Optional[JaxConfig],
        Field(
            default=None,
            description="JAX configuration block for differentiable analysis",
        ),
    ]
    ghost_observables: Annotated[
        Optional[List[GhostObservable]],
        Field(
            default=[],
            description="Variables to compute and store ahead of channel selection."
            "This variables will not be histogrammed unless specified as observable in "
            "a channel.",
        ),
    ]
    baseline_selection: Annotated[
        Optional[FunctorConfig],
        Field(
            default=None,
            description="Baseline event selection applied before "
            "channel-specific logic",
        ),
    ]
    good_object_masks: Annotated[
        Optional[GoodObjectMasksBlockConfig],
        Field(
            default={},
            description="Good object masks to apply before channel "
            + "selection in analysis or pre-training of MVAs."
            "The mask functions are applied to the object in the 'object' field",
        ),
    ]
    channels: Annotated[
        List[ChannelConfig], Field(description="List of analysis channels")
    ]
    corrections: Annotated[
        List[CorrectionConfig],
        Field(description="Corrections to apply to data"),
    ]
    systematics: Annotated[
        List[SystematicConfig],
        Field(description="Systematic variations to apply"),
    ]
    preprocess: Annotated[
        Optional[PreprocessConfig],
        Field(default=None, description="Preprocessing settings"),
    ]
    statistics: Annotated[
        Optional[StatisticalConfig],
        Field(default=None, description="Statistical analysis settings"),
    ]
    mva: Annotated[
        Optional[List[MVAConfig]],
        Field(
            default=None,
            description="List of MVA configurations for pre-training and inference",
        ),
    ]
    plotting: Annotated[
        Optional[PlottingConfig],
        Field(
            default=None,
            description="Global plotting configuration (all keys are optional)",
        ),
    ]

    # Enhanced dataset management
    datasets: Annotated[
        DatasetManagerConfig,
        Field(description="Dataset management configuration (required)")
    ]

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        # Check for duplicate channel names
        channel_names = [channel.name for channel in self.channels]
        if len(channel_names) != len(set(channel_names)):
            raise ValueError("Duplicate channel names found in configuration.")

        # Check for duplicate correction names
        correction_names = [correction.name for correction in self.corrections]
        if len(correction_names) != len(set(correction_names)):
            raise ValueError(
                "Duplicate correction names found in configuration."
            )

        # Check for duplicate systematic names
        systematic_names = [systematic.name for systematic in self.systematics]
        if len(systematic_names) != len(set(systematic_names)):
            raise ValueError(
                "Duplicate systematic names found in configuration."
            )

        if self.general.run_skimming and not self.preprocess:
            raise ValueError(
                "Skimming is enabled but no preprocess configuration provided."
            )

        if self.general.run_skimming and (not self.preprocess.skimming):
            raise ValueError(
                "Skimming is enabled but no skimming configuration provided. "
                "Please provide a SkimmingConfig with selection_function and selection_use."
            )

        if self.statistics is not None:
            if (
                self.general.run_statistics
                and not self.statistics.cabinetry_config
            ):
                raise ValueError(
                    "Statistical analysis run enabled but no cabinetry configuration "
                    + "provided."
                )

        seen_ghost_obs = set()
        for obs in self.ghost_observables:
            names = obs.names if isinstance(obs.names, list) else [obs.names]
            colls = (
                obs.collections
                if isinstance(obs.collections, list)
                else [obs.collections] * len(names)
            )

            if len(names) != len(colls):
                raise ValueError(
                    f"In GhostObservable with function `{obs.function}`, "
                    f"number of names and collections must match if both are lists."
                )

            for name, coll in zip(names, colls):
                pair = (coll, name)
                if pair in seen_ghost_obs:
                    raise ValueError(
                        f"Duplicate (collection, name) pair: {pair}"
                    )
                seen_ghost_obs.add(pair)

        # check for duplicate object names in object masks

        for object_mask in self.good_object_masks.analysis:
            seen_objects = set()
            if object_mask.object in seen_objects:
                raise ValueError(
                    f"Duplicate object '{object_mask.object}' found in good object"
                    "masks collection 'analysis'."
                )
            seen_objects.add(object_mask.object)

        for object_mask in self.good_object_masks.mva:
            seen_objects = set()
            if object_mask.object in seen_objects:
                raise ValueError(
                    f"Duplicate object '{object_mask.object}' found in good object"
                    "masks collection 'mva'."
                )
            seen_objects.add(object_mask.object)

        # check for duplicate mva parameter names
        if self.mva is not None:
            all_mva_params: List[str] = []
            for net in self.mva:
                for layer in net.layers:
                    all_mva_params += [layer.weights, layer.bias]
            duplicates = {p for p in all_mva_params if all_mva_params.count(p) > 1}
            if duplicates:
                raise ValueError(
                    f"Duplicate NN parameter names across MVAs: {sorted(duplicates)}"
                )

        return self


def load_config_with_restricted_cli(
    base_cfg: dict, cli_args: list[str]
) -> dict:
    """
    Load base config and override only `general`, `preprocess`, or `statistics`
    keys via CLI arguments in dotlist form. Raises error for non-existent keys.

    Parameters
    ----------
    base_cfg : dict
        The full Python config with logic, lambdas, etc.
    cli_args : list of str
        CLI args in OmegaConf dotlist format (e.g. general.lumi=25000)

    Returns
    -------
    dict
        Full merged config (with overrides applied to whitelisted sections only).

    Raises
    ------
    ValueError
        If attempting to override a non-existent key or disallowed section
    KeyError
        If attempting to override a non-existent setting in allowed sections
    """
    # {"general", "preprocess", "statistics", "channels"}
    ALLOWED_CLI_TOPLEVEL_KEYS = {}

    # Deep copy so we don't modify the original
    base_copy = copy.deepcopy(base_cfg)

    # Create safe base config with only allowed top-level keys
    safe_base = {
        k: v for k, v in base_copy.items() if k in ALLOWED_CLI_TOPLEVEL_KEYS
    }

    if safe_base == {}:
        safe_base = base_copy

    safe_base_oc = OmegaConf.create(safe_base, flags={"allow_objects": True})

    # Create a set of all valid keys in the safe base config
    valid_keys = set()
    for key_path, _ in OmegaConf.to_container(safe_base_oc).items():
        # Flatten nested dictionary keys
        if isinstance(safe_base_oc[key_path], DictConfig):
            for subkey in safe_base_oc[key_path].keys():
                valid_keys.add(f"{key_path}.{subkey}")
        else:
            valid_keys.add(key_path)

    # Filter CLI args to allowed keys only and check existence
    filtered_cli = []
    for arg in cli_args:
        try:
            key, value = arg.split("=", 1)
        except ValueError:
            raise ValueError(
                f"Invalid CLI argument format: {arg}. Expected 'key=value'"
            )

        top_key = key.split(".", 1)[0]

        # Check if top-level key is allowed
        if ALLOWED_CLI_TOPLEVEL_KEYS != {}:
            if top_key not in ALLOWED_CLI_TOPLEVEL_KEYS:
                raise ValueError(
                    f"Override of top-level key `{top_key}` is not allowed. "
                    f"Allowed keys: {', '.join(ALLOWED_CLI_TOPLEVEL_KEYS)}"
                )

        # Check if full key exists in base config
        if key not in valid_keys:
            raise KeyError(
                f"Cannot override non-existent setting: {key}. "
                f"Valid settings in section '{top_key}':\
                {', '.join(sorted(k for k in valid_keys
                                  if k.startswith(top_key)))}"
            )

        filtered_cli.append(arg)

    # Merge CLI with OmegaConf
    cli_cfg = OmegaConf.from_dotlist(filtered_cli)
    merged_cfg = OmegaConf.merge(safe_base_oc, cli_cfg)
    updated_subsections = OmegaConf.to_container(merged_cfg, resolve=True)

    # Patch back into full config
    if ALLOWED_CLI_TOPLEVEL_KEYS != {}:
        for k in ALLOWED_CLI_TOPLEVEL_KEYS:
            if k in updated_subsections:
                base_copy[k] = updated_subsections[k]
    else:
        for k in updated_subsections.keys():
            base_copy[k] = updated_subsections[k]

    return base_copy
