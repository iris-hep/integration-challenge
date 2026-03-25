"""Skimming configuration models.

This module defines configuration classes for data skimming operations.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from intccms.schema.base import FunctorConfig, SubscriptableModel


class SkimOutputConfig(SubscriptableModel):
    """Configuration for a single skimmed output artifact."""

    format: Annotated[
        Literal["parquet", "ttree", "rntuple"],
        Field(default="parquet", description="Output format for skimmed events"),
    ]
    output_dir: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Base output directory for skimmed events. Can be a local path "
                "or a remote URI (e.g., root://server/path). When None, falls "
                "back to {root_output_dir}/skimmed/ via OutputDirectoryManager."
            ),
        ),
    ]
    to_kwargs: Annotated[
        Dict[str, Any],
        Field(
            default_factory=dict,
            description=(
                "Additional keyword arguments forwarded to the writer "
                "function (e.g., ak.to_parquet)."
            ),
        ),
    ]
    from_kwargs: Annotated[
        Dict[str, Any],
        Field(
            default_factory=dict,
            description=(
                "Additional keyword arguments forwarded to the reader "
                "function (e.g., ak.from_parquet)."
            ),
        ),
    ]


class SkimmingConfig(FunctorConfig):
    """Configuration for workitem-based skimming selections and output"""

    # File handling configuration
    chunk_size: Annotated[
        int,
        Field(default=100_000, description="Number of events to process per chunk (used for configuration compatibility)")
    ]
    tree_name: Annotated[
        str,
        Field(default="Events", description="ROOT tree name for input and output files")
    ]

    # Retry configuration
    max_retries: Annotated[
        int,
        Field(default=3, description="Maximum number of retry attempts for failed workitems")
    ]
    output: Annotated[
        SkimOutputConfig,
        Field(
            default_factory=SkimOutputConfig,
            description="Output format and destination for skimmed data",
        ),
    ]


class PreprocessConfig(SubscriptableModel):
    """Configuration for preprocessing and branch selection."""

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
