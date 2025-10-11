"""
Centralized dataset management with configurable paths, cross-sections, and metadata.

This module provides a configurable dataset manager that replaces hardcoded paths
and cross-sections throughout the codebase, making the framework more flexible
and maintainable.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.schema import DatasetConfig, DatasetManagerConfig

logger = logging.getLogger(__name__)


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

    def get_cross_section(self, process: str) -> float:
        """
        Get cross-section from config instead of hardcoded map.

        Parameters
        ----------
        process : str
            Process name (e.g., 'signal', 'ttbar_semilep', etc.)

        Returns
        -------
        float
            Cross-section in picobarns

        Raises
        ------
        KeyError
            If process is not found in configuration
        """
        if process not in self.datasets:
            raise KeyError(f"Process '{process}' not found in dataset configuration")
        if isinstance(self.datasets[process].cross_sections, float):
            return (self.datasets[process].cross_sections,)
        else:
            return (xs for xs in self.datasets[process].cross_sections)

    def get_dataset_directories(self, process: str) -> Path:
        """
        Get dataset directory containing text files with file lists.

        Parameters
        ----------
        process : str
            Process name

        Returns
        -------
        Path
            Path to directory containing .txt files with file lists
        """
        if process not in self.datasets:
            raise KeyError(f"Process '{process}' not found in dataset configuration")
        if isinstance(self.datasets[process].directories, str):
            return (Path(self.datasets[process].directories),)
        else:
            return (Path(directory) for directory in self.datasets[process].directories)

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
