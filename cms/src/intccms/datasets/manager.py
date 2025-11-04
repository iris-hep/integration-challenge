"""Dataset manager for configuration-driven dataset handling."""

import logging
from pathlib import Path
from typing import Any, List, Optional

from intccms.schema import DatasetConfig, DatasetManagerConfig

from .models import Dataset
from .utils import count_directories, index_or_scalar, normalize_to_list, replicate_single

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages dataset paths, cross-sections, and metadata from configuration.

    This class provides a configuration-driven approach to accessing dataset
    information, replacing hardcoded paths and cross-sections throughout the codebase.
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

    def _validate_process(self, process: str) -> DatasetConfig:
        """
        Validate that process exists and return its configuration.

        Parameters
        ----------
        process : str
            Process name to validate

        Returns
        -------
        DatasetConfig
            The dataset configuration for the process

        Raises
        ------
        KeyError
            If process is not found in configuration
        """
        if process not in self.datasets:
            raise KeyError(f"Process '{process}' not found in dataset configuration")
        return self.datasets[process]

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
        config = self._validate_process(process)
        xsecs = normalize_to_list(config.cross_sections)
        num_dirs = count_directories(config.directories)
        return replicate_single(xsecs, num_dirs)

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

        Raises
        ------
        KeyError
            If process is not found in configuration
        """
        config = self._validate_process(process)
        return normalize_to_list(config.directories, transform=Path)

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

        Raises
        ------
        KeyError
            If process is not found in configuration
        """
        config = self._validate_process(process)
        return config.tree_name

    def get_redirector(self, process: str) -> str:
        """
        Get ROOT file redirector (prefix).

        Parameters
        ----------
        process : str
            Process name

        Returns
        -------
        str
            ROOT file redirector (prefix)

        Raises
        ------
        KeyError
            If process is not found in configuration
        """
        config = self._validate_process(process)
        return config.redirector

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

        Raises
        ------
        KeyError
            If process is not found in configuration
        """
        config = self._validate_process(process)
        return config.is_data

    def get_lumi_mask_config(
        self, process: str, directory_index: Optional[int] = None
    ) -> Optional[Any]:
        """
        Get luminosity mask configuration for a process.

        For multi-directory datasets, you can specify which directory's lumi_mask to retrieve.
        If multiple lumi_masks are configured and directory_index is None, returns the first one.

        Parameters
        ----------
        process : str
            Process name
        directory_index : int, optional
            Index of directory for multi-directory datasets (0-based)

        Returns
        -------
        Optional[Any]
            Luminosity mask FunctorConfig for data, None for MC or if not configured

        Raises
        ------
        KeyError
            If process is not found in configuration
        ValueError
            If directory_index is out of bounds
        """
        config = self._validate_process(process)

        if not config.is_data:
            return None

        lumi_mask = config.lumi_mask
        if lumi_mask is None:
            return None

        return index_or_scalar(
            lumi_mask, index=directory_index, context=f"lumi_masks for process '{process}'"
        )

    def list_processes(self) -> List[str]:
        """
        Get list of all configured process names.

        Returns
        -------
        List[str]
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
