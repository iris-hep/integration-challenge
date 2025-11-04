"""Fileset and metadata builders.

This module provides builders that coordinate core functions and I/O operations
to construct coffea-compatible filesets and Dataset objects.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from intccms.datasets import DatasetManager, Dataset
from intccms.metadata_extractor.core import (
    format_dataset_key,
    build_fileset_entry,
)
from intccms.metadata_extractor.io import collect_file_paths, save_json
from intccms.utils.filters import should_process

logger = logging.getLogger(__name__)


class FilesetBuilder:
    """
    Builds coffea-compatible filesets from dataset configurations.

    This class reads dataset listings and constructs a fileset dictionary
    suitable for coffea processors, along with Dataset objects for the
    analysis pipeline.

    Attributes
    ----------
    dataset_manager : DatasetManager
        Manages dataset configurations, including paths and tree names
    output_manager : OutputDirectoryManager
        Manages output directory paths
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        output_manager: Any,
    ) -> None:
        """
        Initialize FilesetBuilder.

        Parameters
        ----------
        dataset_manager : DatasetManager
            Dataset manager instance
        output_manager : OutputDirectoryManager
            Output directory manager
        """
        self.dataset_manager = dataset_manager
        self.output_manager = output_manager

    def build_fileset(
        self,
        identifiers: Optional[Union[int, List[int]]] = None,
        processes_filter: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dataset]]:
        """
        Build coffea-compatible fileset and Dataset objects from configurations.

        Iterates through configured processes, collects ROOT file paths from listing
        files, and constructs both a fileset dict for coffea preprocessing and Dataset
        objects for the analysis pipeline. Handles multi-directory datasets by creating
        separate fileset entries with index suffixes (e.g., "signal_0__nominal").

        Parameters
        ----------
        identifiers : int or list of ints, optional
            Specific listing file IDs to process. If None, uses all .txt files.
        processes_filter : list of str, optional
            Only build fileset for these processes. If None, builds all.

        Returns
        -------
        Tuple[Dict[str, Dict[str, Any]], List[Dataset]]
            (fileset_dict, datasets_list) where fileset_dict maps dataset keys to
            {"files": {path: treename}, "metadata": {...}} and datasets_list contains
            Dataset objects with cross-sections and process metadata.

        Raises
        ------
        ValueError
            If max_files is configured but <= 0
        """
        fileset: Dict[str, Dict[str, Any]] = {}
        datasets: List[Dataset] = []

        max_files = self.dataset_manager.config.max_files

        if max_files and max_files <= 0:
            raise ValueError("max_files must be None or a positive integer.")

        # Iterate over each process configured in the dataset manager
        for process_name in self.dataset_manager.list_processes():
            # Check if processes filter is configured
            if not should_process(process_name, processes_filter):
                logger.info(f"Skipping {process_name} (not in processes filter)")
                continue

            logger.info(f"Building fileset for process: {process_name}")

            try:
                # Build fileset and dataset for this process
                process_fileset, process_dataset = self._build_process_fileset(
                    process_name, identifiers, max_files
                )

                # Add to results
                fileset.update(process_fileset)
                datasets.append(process_dataset)

            except FileNotFoundError as fnf:
                logger.error(f"Could not build fileset for {process_name}: {fnf}")
                continue

        logger.info(f"Built fileset with {len(fileset)} dataset keys from {len(datasets)} processes")
        return fileset, datasets

    def _build_process_fileset(
        self,
        process_name: str,
        identifiers: Optional[Union[int, List[int]]],
        max_files: Optional[int],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dataset]:
        """
        Build fileset entries and Dataset object for a single process.

        Parameters
        ----------
        process_name : str
            Name of the process to build
        identifiers : int or list of ints, optional
            Listing file IDs to process
        max_files : int, optional
            Maximum number of files per directory

        Returns
        -------
        Tuple[Dict[str, Dict[str, Any]], Dataset]
            (fileset_entries, dataset_object)

        Raises
        ------
        FileNotFoundError
            If listing files are not found
        """
        # Get process configuration
        listing_dirs = self.dataset_manager.get_dataset_directories(process_name)
        cross_sections = self.dataset_manager.get_cross_section(process_name)
        tree_name = self.dataset_manager.get_tree_name(process_name)
        redirector = self.dataset_manager.get_redirector(process_name)
        is_data = self.dataset_manager.is_data_dataset(process_name)

        # Validate configuration
        if len(listing_dirs) != len(cross_sections):
            raise ValueError(
                f"Mismatch between number of directories ({len(listing_dirs)}) "
                f"and cross-sections ({len(cross_sections)}) for process {process_name}"
            )

        # Build fileset entries for each directory
        fileset_entries = {}
        fileset_keys = []
        lumi_mask_configs = []
        variation_label = "nominal"

        for idx, (listing_dir, xsec) in enumerate(zip(listing_dirs, cross_sections)):
            # Get lumi_mask_config for this directory
            directory_index = idx if len(listing_dirs) > 1 else None
            lumi_mask_config = self.dataset_manager.get_lumi_mask_config(
                process_name, directory_index=directory_index
            )
            lumi_mask_configs.append(lumi_mask_config)

            # Collect file paths
            file_paths = collect_file_paths(listing_dir, identifiers, redirector)

            # Apply max_files limit
            if max_files:
                file_paths = file_paths[:max_files]

            # Create dataset key
            dataset_key = format_dataset_key(
                process_name,
                variation=variation_label,
                directory_index=directory_index,
                is_data=is_data,
            )

            # Build fileset entry
            fileset_entries[dataset_key] = build_fileset_entry(
                file_paths=file_paths,
                tree_name=tree_name,
                process_name=process_name,
                variation=variation_label,
                xsec=xsec,
                is_data=is_data,
            )

            fileset_keys.append(dataset_key)
            logger.debug(f"Added {len(file_paths)} files for {dataset_key} with xsec={xsec}")

        # Create Dataset object
        dataset = Dataset(
            name=process_name,
            fileset_keys=fileset_keys,
            process=process_name,
            variation=variation_label,
            cross_sections=cross_sections,
            is_data=is_data,
            lumi_mask_configs=lumi_mask_configs,
            events=None,  # Will be populated during skimming
        )

        logger.debug(f"Created Dataset: {dataset}")

        return fileset_entries, dataset

    def save_fileset(self, fileset: Dict[str, Dict[str, Any]]) -> None:
        """
        Save the built fileset to a JSON file.

        Parameters
        ----------
        fileset : Dict[str, Dict[str, Any]]
            The fileset mapping to save
        """
        output_dir = Path(self.output_manager.metadata_dir)
        fileset_path = output_dir / "fileset.json"

        save_json(fileset, fileset_path)
