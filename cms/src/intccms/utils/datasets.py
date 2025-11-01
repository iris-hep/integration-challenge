"""
Centralized dataset management with configurable paths, cross-sections, and metadata.

This module provides a configurable dataset manager that replaces hardcoded paths
and cross-sections throughout the codebase, making the framework more flexible
and maintainable.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from intccms.utils.schema import DatasetConfig, DatasetManagerConfig

logger = logging.getLogger(__name__)


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
    lumi_mask_config : Optional[Any]
        Luminosity mask configuration for data (FunctorConfig), None for MC
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
    lumi_mask_config: Optional[Any] = None
    events: Optional[List[Tuple[Any, Dict]]] = field(default=None)

    def __repr__(self) -> str:
        """String representation for debugging."""
        events_info = f"{len(self.events)} splits" if self.events else "no events"
        return (
            f"Dataset(name={self.name}, fileset_keys={self.fileset_keys}, "
            f"{events_info})"
        )


class ConfigurableDatasetManager:
    """
    Manages dataset paths, cross-sections, and metadata from configuration.

    This class replaces hardcoded dataset directories and cross-section maps
    with a flexible, configuration-driven approach.
    """

    def __init__(self, config: DatasetManagerConfig):
        """
        Initialize the dataset manager with configuration.

        Parameters
        ----------
        config : DatasetManagerConfig
            Configuration containing dataset definitions and paths.
        """
        self.config = config
        self.datasets = {ds.name: ds for ds in config.datasets}
        logger.info(f"Initialized dataset manager with {len(self.datasets)} datasets")

    def get_cross_section(self, process: str) -> List[float]:
        """
        Get cross-section(s) from config as a list.

        Parameters
        ----------
        process : str
            Process name (e.g., 'signal', 'ttbar_semilep', etc.)

        Returns
        -------
        List[float]
            List of cross-section(s) in picobarns. Always returns a list.
            If a single cross-section is configured but multiple directories exist,
            the cross-section is replicated for each directory.

        Raises
        ------
        KeyError
            If process is not found in configuration
        """
        if process not in self.datasets:
            raise KeyError(f"Process '{process}' not found in dataset configuration")

        xsecs = self.datasets[process].cross_sections
        dirs = self.datasets[process].directories

        # Normalize to lists
        if isinstance(xsecs, float):
            xsecs = [xsecs]
        else:
            xsecs = list(xsecs)

        if isinstance(dirs, str):
            num_dirs = 1
        else:
            num_dirs = len(dirs)

        # If single xsec but multiple directories, replicate the xsec
        if len(xsecs) == 1 and num_dirs > 1:
            xsecs = xsecs * num_dirs

        return xsecs

    def get_dataset_directories(self, process: str) -> List[Path]:
        """
        Get dataset directory/directories containing text files with file lists.

        Parameters
        ----------
        process : str
            Process name

        Returns
        -------
        List[Path]
            List of Path(s) to directory/directories containing .txt files with file lists.
            Always returns a list.
        """
        if process not in self.datasets:
            raise KeyError(f"Process '{process}' not found in dataset configuration")

        dirs = self.datasets[process].directories
        if isinstance(dirs, str):
            return [Path(dirs)]
        else:
            return [Path(directory) for directory in dirs]

    def get_tree_name(self, process: str) -> str:
        """
        Get ROOT tree name from config.

        Parameters
        ----------
        process : str
            Process name

        Returns
        -------
        str
            ROOT tree name
        """
        if process not in self.datasets:
            raise KeyError(f"Process '{process}' not found in dataset configuration")
        return self.datasets[process].tree_name
    
    def get_redirector(self, process: str) -> str:
        """
        Get ROOT file redirector (prefix)

        Parameters
        ----------
        process : str
            Process name

        Returns
        -------
        str
            ROOT file redirector (prefix)
        """
        if process not in self.datasets:
            raise KeyError(f"Process '{process}' not found in dataset configuration")
        return self.datasets[process].redirector

    def is_data_dataset(self, process: str) -> bool:
        """
        Check if a process represents real data (not MC).

        Parameters
        ----------
        process : str
            Process name

        Returns
        -------
        bool
            True if this is a data dataset, False for MC
        """
        return self.datasets[process].is_data

    def get_lumi_mask_config(self, process: str) -> Optional[Any]:
        """
        Get luminosity mask configuration for a process.

        Parameters
        ----------
        process : str
            Process name

        Returns
        -------
        Optional[Any]
            Luminosity mask FunctorConfig for data, None for MC or if not configured
        """
        dataset_config = self.datasets[process]
        if dataset_config.is_data:
            return dataset_config.lumi_mask
        return None

    def list_processes(self) -> List[str]:
        """
        Get list of all configured process names.

        Returns
        -------
        list
            List of process names
        """
        return list(self.datasets.keys())

    def validate_process(self, process: str) -> bool:
        """
        Check if a process is configured.

        Parameters
        ----------
        process : str
            Process name to check

        Returns
        -------
        bool
            True if process is configured, False otherwise
        """
        return process in self.datasets
