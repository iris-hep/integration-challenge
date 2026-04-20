"""Pre-skimming configuration models.

This module defines configuration classes for data Pre-skimming operations.
"""

from typing import Annotated, List

from pydantic import Field

from intccms.schema.base import SubscriptableModel


class PreSkimConfig(SubscriptableModel):
    """Configuration for preprocessing and branch selection."""

    selections: Annotated[
        List[str],
        Field(
            description="UprootRaw ServiceX selections"
        ),
    ]
    temp_dir: Annotated[
        str,
        Field(
            description="Location to save the servicex output. Not implemented yet."
        ),
    ]
    futures_max_workers: Annotated[
        int,
        Field(
            description="Number of servicex requests at a time. Controlled by Futures ThrealPoolExecutor Workers"
        ),
    ]
