"""
Pydantic schemas for validating the analysis configuration.
"""

from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import Field, model_validator, field_validator

from intccms.schema.base import FunctorConfig, ObjVar, SubscriptableModel


class DatasetConfig(SubscriptableModel):
    """Configuration for individual dataset paths, cross-sections, and metadata"""
    name: Annotated[str, Field(description="Dataset name/identifier")]
    directories: Annotated[Union[str, tuple[str, ...]], Field(description="Directories containing dataset files")]
    cross_sections: Annotated[Union[float, tuple[float, ...]], Field(description="Cross-sections in picobarns")]
    file_pattern: Annotated[str, Field(default="*.root", description="File pattern for dataset files")]
    tree_name: Annotated[str, Field(default="Events", description="ROOT tree name")]
    weight_branch: Annotated[Optional[str], Field(default="genWeight", description="Branch name for event weights")]
    redirector: Annotated[Optional[str], Field(default=None, description="redirector to prefix ROOT file-paths")]
    is_data: Annotated[bool, Field(default=False, description="Flag indicating whether dataset represents real data")]
    lumi_mask: Annotated[
        Optional[Union[FunctorConfig, Tuple[FunctorConfig, ...], List[FunctorConfig]]],
        Field(
            default=None,
            description="Optional lumi mask configuration(s) applied to data datasets. "
                        "Can be a single FunctorConfig or a sequence of FunctorConfig (one per directory).",
        ),
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

    @model_validator(mode="after")
    def validate_general(self) -> "DatasetManagerConfig":
        """Validate the dataset manager configuration settings."""
        # Check number of directories and number of cross-sections
        for dataset_config in self.datasets:
            dirs = dataset_config.directories
            xss = dataset_config.cross_sections
            lumi_masks = dataset_config.lumi_mask

            # Get the count of directories
            num_dirs = len(dirs) if isinstance(dirs, tuple) else 1
            # Get the count of cross-sections
            num_xss = len(xss) if isinstance(xss, tuple) else 1
            # Get the count of lumi_masks (handle both tuple and list from OmegaConf)
            num_lumi_masks = len(lumi_masks) if isinstance(lumi_masks, (tuple, list)) else (1 if lumi_masks is not None else 0)

            # Validate cross-sections
            # Valid cases:
            # 1. Single dir, single xsec
            # 2. Multiple dirs, single xsec (will be replicated)
            # 3. Multiple dirs, matching number of xsecs
            if num_xss > 1 and num_xss != num_dirs:
                raise ValueError(
                    f"Dataset '{dataset_config.name}': You must provide either a single cross-section "
                    f"or an equal number of cross-sections ({num_dirs}) to match the number of directories. "
                    f"Got {num_dirs} directories and {num_xss} cross-sections."
                )

            # Validate lumi_masks
            # Valid cases:
            # 1. No lumi_mask (None)
            # 2. Single lumi_mask (will be used for single dir or replicated for multiple dirs)
            # 3. Multiple lumi_masks matching number of directories
            if num_lumi_masks > 1 and num_lumi_masks != num_dirs:
                raise ValueError(
                    f"Dataset '{dataset_config.name}': You must provide either a single lumi_mask "
                    f"or an equal number of lumi_masks ({num_dirs}) to match the number of directories. "
                    f"Got {num_dirs} directories and {num_lumi_masks} lumi_masks."
                )

        return self
