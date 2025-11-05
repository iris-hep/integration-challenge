"""Main configuration model and utility functions.

This module defines the top-level Config class that composes all subsystem
configurations, plus utility functions for loading and managing configs.
"""

import copy
from typing import Annotated, List, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from intccms.schema.analysis import (
    ChannelConfig,
    CorrectionConfig,
    GeneralConfig,
    GhostObservable,
    GoodObjectMasksBlockConfig,
    MetricsConfig,
    PlottingConfig,
    StatisticalConfig,
    SystematicConfig,
)
from intccms.schema.base import FunctorConfig, SubscriptableModel
from intccms.schema.mva import MVAConfig
from intccms.schema.skimming import PreprocessConfig


class Config(SubscriptableModel):
    """Top-level configuration model for the analysis framework."""

    general: Annotated[
        GeneralConfig, Field(description="Global settings for the analysis")
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
    datasets: Annotated[
        DatasetManagerConfig,
        Field(description="Dataset management configuration (required)")
    ]

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
