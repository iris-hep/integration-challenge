"""Configuration schema for the analysis framework.

This package provides Pydantic models for validating analysis configurations.
All public classes are re-exported from this module for convenient imports.
"""

from intccms.schema.base import FunctorConfig, ObjVar, SubscriptableModel, WorkerEval
from intccms.schema.datasets import DatasetConfig, DatasetManagerConfig
from intccms.schema.skimming import PreprocessConfig, SkimOutputConfig, SkimmingConfig
from intccms.schema.analysis import (
    ChannelConfig,
    CorrectionConfig,
    GeneralConfig,
    GhostObservable,
    GoodObjectMasksBlockConfig,
    GoodObjectMasksConfig,
    ObservableConfig,
    PlottingConfig,
    StatisticalConfig,
    SystematicConfig,
)
from intccms.schema.mva import ActivationKey, FeatureConfig, LayerConfig, MVAConfig
from intccms.schema.config import Config, load_config_with_restricted_cli

__all__ = [
    # Base
    "FunctorConfig",
    "ObjVar",
    "SubscriptableModel",
    "WorkerEval",
    # Datasets
    "DatasetConfig",
    "DatasetManagerConfig",
    # Skimming
    "PreprocessConfig",
    "SkimOutputConfig",
    "SkimmingConfig",
    # Analysis
    "ChannelConfig",
    "CorrectionConfig",
    "GeneralConfig",
    "GhostObservable",
    "GoodObjectMasksBlockConfig",
    "GoodObjectMasksConfig",
    "ObservableConfig",
    "PlottingConfig",
    "StatisticalConfig",
    "SystematicConfig",
    # MVA
    "ActivationKey",
    "FeatureConfig",
    "LayerConfig",
    "MVAConfig",
    # Config
    "Config",
    "load_config_with_restricted_cli",
]
