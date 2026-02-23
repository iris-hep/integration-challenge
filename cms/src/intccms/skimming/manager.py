"""High-level skimming workflow manager.

This module provides the main user-facing interface for the skimming workflow,
orchestrating workitem processing, file discovery, loading, and caching.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import awkward as ak
from coffea.processor.executor import WorkItem
from coffea.nanoevents import NanoAODSchema

from intccms.schema import SkimmingConfig
from intccms.skimming.dask import WorkitemSkimmingManager
from intccms.skimming.cacher import (
    compute_cache_key,
    load_cached_events,
    save_cached_events,
)
from intccms.skimming.pipeline.stages import load_events
from intccms.skimming.io.readers import get_reader
from intccms.skimming.workitem import resolve_lazy_values

# Suppress NanoAOD cross-reference warnings
NanoAODSchema.warn_missing_crossrefs = False

logger = logging.getLogger(__name__)


class SkimmingManager:
    """High-level manager for the complete skimming workflow.

    This manager orchestrates:
    1. Parallel skimming of NanoAOD files (if needed)
    2. Discovery of skimmed output files
    3. Loading and merging events per dataset
    4. Caching for fast subsequent runs

    This is the main user-facing API for skimming operations.

    Attributes
    ----------
    config : SkimmingConfig
        Skimming configuration with selection functions and output settings
    output_manager : OutputDirectoryManager
        Centralized output directory manager
    workitem_manager : WorkitemSkimmingManager
        Manager for parallel workitem processing

    Examples
    --------
    >>> manager = SkimmingManager(config, output_manager)
    >>> datasets = manager.run(workitems, configuration, datasets)
    >>> # datasets now contain merged events ready for analysis
    """

    def __init__(self, config: SkimmingConfig, output_manager: Any):
        """Initialize the skimming manager.

        Args:
            config: Skimming configuration with selection functions and output settings
            output_manager: OutputDirectoryManager for path resolution
        """
        self.config = config
        self.output_manager = output_manager
        self.workitem_manager = WorkitemSkimmingManager(config, output_manager)
        logger.info("Initialized high-level skimming manager")

    def run(
        self,
        workitems: List[WorkItem],
        configuration: Any,
        datasets: List[Any],
        metadata_lookup: Dict[str, Dict[str, Any]],
        skip_skimming: bool = False,
        use_cache: bool = True,
    ) -> List[Any]:
        """Execute the complete skimming workflow.

        This is the main entry point that orchestrates:
        1. Optional skimming phase (parallel workitem processing)
        2. Discovery of skimmed files
        3. Loading and merging events per dataset with caching

        Args:
            workitems: List of WorkItems to process (if skimming)
            configuration: Main analysis configuration object
            datasets: Dataset objects with metadata (fileset_keys, cross_sections, etc.)
            metadata_lookup: Pre-built metadata lookup from NanoAODMetadataGenerator.build_metadata_lookup()
                Maps fileset_key -> {process, variation, xsec, nevts, is_data, dataset} (REQUIRED)
            skip_skimming: If True, skip skimming and load existing files
            use_cache: If True, use cached merged events when available

        Returns:
            List of Dataset objects with events loaded and metadata attached
        """
        # Phase 1: Skimming (if needed)
        if not skip_skimming:
            logger.info("=== Phase 1: Skimming ===")
            self._run_skimming(workitems, configuration, datasets)
        else:
            logger.info("Skipping skimming phase, will load existing files")

        # Phase 2: Loading and merging
        logger.info("=== Phase 2: Loading and Merging ===")
        self._load_and_merge(workitems, configuration, datasets, use_cache, metadata_lookup)

        return datasets

    def _run_skimming(
        self,
        workitems: List[WorkItem],
        configuration: Any,
        datasets: List[Any],
    ) -> None:
        """Run the parallel skimming phase.

        Args:
            workitems: List of WorkItems to process
            configuration: Main analysis configuration object
            datasets: Dataset objects with metadata
        """
        result = self.workitem_manager.process_workitems(
            workitems,
            configuration,
            datasets,
        )

        # Log summary
        logger.info(
            f"Skimming complete: {result['processed_events']} events, "
            f"{len(result['output_files'])} files created"
        )

        if result.get("failed_items"):
            logger.warning(
                f"Failed to process {len(result['failed_items'])} workitems"
            )

    def _load_and_merge(
        self,
        workitems: List[WorkItem],
        configuration: Any,
        datasets: List[Any],
        use_cache: bool,
        metadata_lookup: Dict[str, Dict[str, Any]],
    ) -> None:
        """Load and merge skimmed files per dataset.

        Discovers output files, loads them using our reader infrastructure,
        merges per dataset, and handles caching.

        Args:
            workitems: List of WorkItems (for discovering outputs)
            configuration: Main analysis configuration object
            datasets: Dataset objects to populate with events
            use_cache: Whether to use cached merged events
            metadata_lookup: Pre-built metadata lookup mapping fileset_key to metadata dict (REQUIRED)
        """
        # Group workitems by dataset/fileset_key
        workitems_by_dataset = self._group_workitems_by_dataset(workitems)

        # Build mapping from fileset_key to Dataset object
        fileset_key_to_dataset = self._build_fileset_key_mapping(datasets)

        # Initialize events list for each Dataset
        for dataset in datasets:
            dataset.events = []

        # Get cache directory
        cache_dir = self.output_manager.get_cache_dir()

        # Resolve lazy values in reader kwargs
        reader_kwargs = resolve_lazy_values(
            self.config.output.from_kwargs or {}
        )

        # Process each fileset
        for fileset_key, dataset_workitems in workitems_by_dataset.items():
            # Get the Dataset object this fileset_key belongs to
            if fileset_key not in fileset_key_to_dataset:
                logger.warning(
                    f"Fileset key '{fileset_key}' not found in any Dataset object, skipping"
                )
                continue

            dataset_obj = fileset_key_to_dataset[fileset_key]

            # Skip datasets not explicitly requested in config
            if self._should_skip_dataset(configuration, dataset_obj):
                logger.info(
                    f"Skipping {fileset_key} (process {dataset_obj.process} not requested)"
                )
                continue

            # Discover output files for this fileset_key
            output_files = self.workitem_manager.discover_workitem_outputs(
                dataset_workitems
            )

            if not output_files:
                logger.warning(f"No output files found for {fileset_key}")
                continue

            # Get metadata for this fileset
            if fileset_key not in metadata_lookup:
                raise ValueError(
                    f"Fileset key '{fileset_key}' not found in metadata_lookup. "
                    "Ensure NanoAODMetadataGenerator.build_metadata_lookup() was called "
                    "and passed to SkimmingManager.run()."
                )

            metadata = metadata_lookup[fileset_key].copy()

            # Load and merge events (with caching)
            merged_events = self._load_and_merge_with_cache(
                fileset_key=fileset_key,
                output_files=output_files,
                cache_dir=cache_dir,
                use_cache=use_cache,
                reader_kwargs=reader_kwargs,
            )

            if merged_events is not None:
                dataset_obj.events.append((merged_events, metadata.copy()))
                logger.info(
                    f"Loaded {len(merged_events)} events for {fileset_key}"
                )
            else:
                logger.error(f"Failed to load events for {fileset_key}")

    def _load_and_merge_with_cache(
        self,
        fileset_key: str,
        output_files: List[str],
        cache_dir: Path,
        use_cache: bool,
        reader_kwargs: Dict[str, Any],
    ) -> Optional[Any]:
        """Load and merge events for a fileset, using cache if available.

        Uses stages.load_events() and ak.concatenate() from existing infrastructure.

        Args:
            fileset_key: Dataset/fileset identifier
            output_files: List of skimmed output file paths
            cache_dir: Directory for cache files
            use_cache: Whether to read from and write to cache
            reader_kwargs: Additional keyword arguments for the reader

        Returns:
            Merged events, or None if loading failed
        """
        # Compute cache key
        cache_key = compute_cache_key(fileset_key, output_files)
        cache_file = cache_dir / f"{fileset_key}__{cache_key}.pkl"

        # Try to load from cache
        if use_cache:
            cached_events = load_cached_events(cache_file)
            if cached_events is not None:
                logger.info(f"Using cached events for {fileset_key}")
                return cached_events

        # Load files using stages.load_events()
        reader = get_reader(self.config.output.format)
        all_events = []

        for file_path in output_files:
            try:
                # Determine tree_name for ROOT files
                if self.config.output.format in ("ttree", "rntuple"):
                    tree_name = reader_kwargs.get("tree_name", self.config.tree_name)
                else:
                    tree_name = None

                events = load_events(
                    reader=reader,
                    path=file_path,
                    tree_name=tree_name,
                    **reader_kwargs
                )
                all_events.append(events)
            except Exception as e:
                logger.error(f"Failed to load events from {file_path}: {e}")
                continue

        if not all_events:
            return None

        # Merge using ak.concatenate()
        try:
            if len(all_events) == 1:
                merged_events = all_events[0]
            else:
                merged_events = ak.concatenate(all_events, axis=0)

            logger.info(
                f"Merged {len(output_files)} files → "
                f"{len(merged_events)} events for {fileset_key}"
            )

            # Save to cache
            if use_cache:
                save_cached_events(merged_events, cache_file)

            return merged_events

        except Exception as e:
            logger.error(f"Failed to merge events for {fileset_key}: {e}")
            return None

    def _group_workitems_by_dataset(
        self, workitems: List[WorkItem]
    ) -> Dict[str, List[WorkItem]]:
        """Group workitems by their dataset/fileset_key.

        Args:
            workitems: List of WorkItems to group

        Returns:
            Dictionary mapping fileset_key -> list of workitems
        """
        workitems_by_dataset = defaultdict(list)
        for workitem in workitems:
            workitems_by_dataset[workitem.dataset].append(workitem)
        return workitems_by_dataset

    def _build_fileset_key_mapping(
        self, datasets: List[Any]
    ) -> Dict[str, Any]:
        """Build mapping from fileset_key to Dataset object.

        Args:
            datasets: List of Dataset objects

        Returns:
            Dictionary mapping fileset_key -> Dataset object
        """
        fileset_key_to_dataset = {}
        for dataset in datasets:
            for fileset_key in dataset.fileset_keys:
                fileset_key_to_dataset[fileset_key] = dataset
        return fileset_key_to_dataset

    def _should_skip_dataset(
        self, configuration: Any, dataset_obj: Any
    ) -> bool:
        """Check if dataset should be skipped based on configuration.

        Args:
            configuration: Main analysis configuration
            dataset_obj: Dataset object to check

        Returns:
            True if dataset should be skipped, False otherwise
        """
        if hasattr(configuration, "general") and hasattr(
            configuration.general, "processes"
        ):
            if configuration.general.processes:
                return dataset_obj.process not in configuration.general.processes
        return False

