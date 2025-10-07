"""
Centralized output directory management for the Z-prime ttbar analysis.

This module provides a unified interface for managing all output directories,
with support for user-specified paths for metadata and skimmed files.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OutputDirectoryManager:
    """
    Centralized manager for all output directories in the analysis.

    Provides a single source of truth for output paths with fallback logic
    for metadata and skimmed files.
    """

    def __init__(
        self,
        root_output_dir: str,
        cache_dir: Optional[str] = None,
        metadata_dir: Optional[str] = None,
        skimmed_dir: Optional[str] = None
    ):
        """
        Initialize the output directory manager.

        Parameters
        ----------
        root_output_dir : str
            Root directory for all analysis outputs
        cache_dir : str, optional
            Cache directory for temporary files. If None, uses system temp directory with 'graep' subdirectory.
        metadata_dir : str, optional
            Directory containing metadata JSON files. If None, looks under root_output_dir/metadata/
        skimmed_dir : str, optional
            Directory containing skimmed ROOT files. If None, looks under root_output_dir/skimmed/
        """
        # Normalize root output directory path
        self.root_output_dir = Path(root_output_dir).expanduser().resolve()

        # Normalize cache directory path or use system temp directory
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "graep"

        # User-specified directories (if provided) - normalize paths
        self._user_metadata_dir = Path(metadata_dir).expanduser().resolve() if metadata_dir else None
        self._user_skimmed_dir = Path(skimmed_dir).expanduser().resolve() if skimmed_dir else None

        # Standard subdirectory structure
        self._subdirs = {
            "metadata": "metadata",
            "skimmed": "skimmed",
            "plots": "plots",
            "models": "models",
            "histograms": "histograms",
            "statistics": "statistics"
        }

        # Create root and cache directories
        self.root_output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory manager initialized with root: {self.root_output_dir}")
        if self._user_metadata_dir:
            logger.info(f"Using user-specified metadata directory: {self._user_metadata_dir}")
        if self._user_skimmed_dir:
            logger.info(f"Using user-specified skimmed directory: {self._user_skimmed_dir}")

    def get_root_dir(self) -> Path:
        """Get the root output directory."""
        return self.root_output_dir

    def get_cache_dir(self) -> Path:
        """Get the cache directory."""
        return self.cache_dir

    def get_metadata_dir(self) -> Path:
        """
        Get the metadata directory with fallback logic.

        Returns
        -------
        Path
            Path to metadata directory

        Raises
        ------
        FileNotFoundError
            If directory doesn't exist and no user-specified path
        """
        if self._user_metadata_dir:
            # User specified a metadata directory - check if it exists and is a directory
            if not self._user_metadata_dir.exists():
                raise FileNotFoundError(
                    f"Metadata path not found: {self._user_metadata_dir}"
                )
            if not self._user_metadata_dir.is_dir():
                raise NotADirectoryError(
                    f"Metadata path is not a directory: {self._user_metadata_dir}"
                )
            return self._user_metadata_dir
        else:
            # Use standard location under root
            metadata_dir = self.root_output_dir / self._subdirs["metadata"]
            if not metadata_dir.exists():
                raise FileNotFoundError(
                    f"Metadata directory does not exist: {metadata_dir}. "
                    f"Either run metadata generation or specify an existing metadata directory."
                )
            return metadata_dir

    def get_metadata_dir_for_writing(self) -> Path:
        """
        Get the metadata directory for writing (creates if needed).

        Returns
        -------
        Path
            Path to metadata directory
        """
        if self._user_metadata_dir:
            metadata_dir = self._user_metadata_dir
        else:
            metadata_dir = self.root_output_dir / self._subdirs["metadata"]

        metadata_dir.mkdir(parents=True, exist_ok=True)
        return metadata_dir

    def get_skimmed_dir(self) -> Path:
        """
        Get the skimmed files directory with fallback logic.

        Returns
        -------
        Path
            Path to skimmed files directory

        Raises
        ------
        FileNotFoundError
            If directory doesn't exist and no user-specified path
        """
        if self._user_skimmed_dir:
            # User specified a skimmed directory - check if it exists and is a directory
            if not self._user_skimmed_dir.exists():
                raise FileNotFoundError(
                    f"Skimmed path not found: {self._user_skimmed_dir}"
                )
            if not self._user_skimmed_dir.is_dir():
                raise NotADirectoryError(
                    f"Skimmed path is not a directory: {self._user_skimmed_dir}"
                )
            return self._user_skimmed_dir
        else:
            # Use standard location under root
            skimmed_dir = self.root_output_dir / self._subdirs["skimmed"]
            if not skimmed_dir.exists():
                raise FileNotFoundError(
                    f"Skimmed directory does not exist: {skimmed_dir}. "
                    f"Either run skimming or specify an existing skimmed directory."
                )
            return skimmed_dir

    def get_skimmed_dir_for_writing(self) -> Path:
        """
        Get the skimmed files directory for writing (creates if needed).

        Returns
        -------
        Path
            Path to skimmed files directory
        """
        if self._user_skimmed_dir:
            skimmed_dir = self._user_skimmed_dir
        else:
            skimmed_dir = self.root_output_dir / self._subdirs["skimmed"]

        skimmed_dir.mkdir(parents=True, exist_ok=True)
        return skimmed_dir

    def get_plots_dir(self, subdir: Optional[str] = None) -> Path:
        """
        Get the plots directory, optionally with a subdirectory.

        Parameters
        ----------
        subdir : str, optional
            Subdirectory under plots/ (e.g., 'features', 'scores', 'optimisation', 'fit')

        Returns
        -------
        Path
            Path to plots directory or subdirectory
        """
        plots_dir = self.root_output_dir / self._subdirs["plots"]
        if subdir:
            plots_dir = plots_dir / subdir

        plots_dir.mkdir(parents=True, exist_ok=True)
        return plots_dir

    def get_models_dir(self) -> Path:
        """Get the models directory (creates if needed)."""
        models_dir = self.root_output_dir / self._subdirs["models"]
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def get_histograms_dir(self) -> Path:
        """Get the histograms directory (creates if needed)."""
        histograms_dir = self.root_output_dir / self._subdirs["histograms"]
        histograms_dir.mkdir(parents=True, exist_ok=True)
        return histograms_dir

    def get_statistics_dir(self) -> Path:
        """Get the statistics directory (creates if needed)."""
        statistics_dir = self.root_output_dir / self._subdirs["statistics"]
        statistics_dir.mkdir(parents=True, exist_ok=True)
        return statistics_dir

    def get_dataset_dir(self, dataset_name: str) -> Path:
        """
        Get a dataset-specific directory under the root (creates if needed).

        Parameters
        ----------
        dataset_name : str
            Name of the dataset

        Returns
        -------
        Path
            Path to dataset directory
        """
        dataset_dir = self.root_output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def get_custom_dir(self, subpath: str) -> Path:
        """
        Get a custom directory under the root output directory (creates if needed).

        Parameters
        ----------
        subpath : str
            Relative path under the root directory

        Returns
        -------
        Path
            Path to custom directory
        """
        custom_dir = self.root_output_dir / subpath
        custom_dir.mkdir(parents=True, exist_ok=True)
        return custom_dir

    def list_structure(self) -> Dict[str, Any]:
        """
        Get a summary of the current directory structure.

        Returns
        -------
        dict
            Dictionary describing the directory structure
        """
        structure = {
            "root_output_dir": str(self.root_output_dir),
            "cache_dir": str(self.cache_dir),
            "metadata_dir": str(self._user_metadata_dir or self.root_output_dir / "metadata"),
            "skimmed_dir": str(self._user_skimmed_dir or self.root_output_dir / "skimmed"),
            "standard_subdirs": {
                name: str(self.root_output_dir / path)
                for name, path in self._subdirs.items()
            }
        }
        return structure
