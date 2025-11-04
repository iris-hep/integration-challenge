"""MVA (Machine Learning) configuration models.

This module defines configuration classes for ML/MVA training and inference.
"""

from enum import Enum
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import Field, field_validator, model_validator

from intccms.schema.base import FunctorConfig, SubscriptableModel
from intccms.utils.binning import validate_binning_spec, binning_to_edges


class ActivationKey(str, Enum):
    relu = "relu"
    tanh = "tanh"
    sigmoid = "sigmoid"


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


class FeatureConfig(FunctorConfig):
    name: Annotated[str, Field(..., description="Feature name")]
    label: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Optional label for plots (e.g. LaTeX string)",
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
                "either 'low,high,nbins' or explicit edge list. Parsed to array during validation."
            ),
        ),
    ]

    @field_validator("binning", mode="before")
    @classmethod
    def validate_and_parse_binning(cls, v: Optional[Union[str, List[float]]]) -> Optional[np.ndarray]:
        """Validate and parse binning specification to array of edges."""
        if v is None:
            return None
        validate_binning_spec(v)  # Validate first
        return binning_to_edges(v)  # Then convert to edges


class MVAConfig(SubscriptableModel):
    name: Annotated[
        str, Field(..., description="Unique name for this neural network")
    ]
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
