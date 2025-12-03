"""Data models for dataset management."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Dataset:
    """
    Represents a logical dataset that may span multiple fileset entries.

    A Dataset encapsulates all information about a physics process, including
    its fileset keys and cross-sections. When multiple directories are provided,
    events are processed separately but histograms naturally accumulate.

    Attributes
    ----------
    name : str
        Logical name of the dataset (e.g., "signal", "ttbar_semilep")
    fileset_keys : List[str]
        List of fileset keys this dataset spans
        (e.g., ["signal_0__nominal", "signal_1__nominal", "signal_2__nominal"])
    process : str
        Process name (e.g., "signal")
    variation : str
        Systematic variation label (e.g., "nominal")
    cross_sections : List[float]
        Cross-section in picobarns for each fileset entry
    is_data : bool
        Flag indicating whether dataset represents real data
    lumi_mask_configs : List[Optional[Any]]
        List of luminosity mask configurations (FunctorConfig) for data, one per fileset_key.
        None entries for MC or if not configured. Length must match fileset_keys.
    events : Optional[List[Tuple[Any, Dict]]]
        Processed events from skimming, added after skimming completes.
        Each tuple contains (events_array, metadata_dict) for one fileset entry.
    """

    name: str
    fileset_keys: List[str]
    process: str
    variation: str
    cross_sections: List[float]
    is_data: bool = False
    lumi_mask_configs: List[Optional[Any]] = field(default_factory=list)
    events: Optional[List[Tuple[Any, Dict]]] = field(default=None)

    def __repr__(self) -> str:
        """String representation for debugging."""
        events_info = f"{len(self.events)} splits" if self.events else "no events"
        return (
            f"Dataset(name={self.name}, fileset_keys={self.fileset_keys}, "
            f"{events_info})"
        )
