"""Coffea processor for unified skimming and analysis workflow.

This module provides the UnifiedProcessor class that integrates skimming
and analysis into a single distributed coffea-based workflow. The processor
respects configuration flags to control which stages run.
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import awkward as ak
from coffea.processor import ProcessorABC
from roastcoffea import track_metrics, track_time

from intccms.analysis.nondiff import NonDiffAnalysis
from intccms.skimming.io.writers import get_writer
from intccms.skimming.pipeline.stages import (
    build_column_list,
    extract_columns,
    save_events,
)
from intccms.skimming.workitem import resolve_lazy_values
from intccms.utils.output import (
    OutputDirectoryManager,
    save_histograms_to_pickle,
    save_histograms_to_root,
)
from intccms.schema import Config
from intccms.utils.functors import SelectionExecutor

logger = logging.getLogger(__name__)


class UnifiedProcessor(ProcessorABC):
    """Coffea processor for distributed skimming and/or analysis.

    This processor integrates the skimming pipeline with the analysis workflow,
    controlled by configuration flags. It supports two primary workflows:

    **Workflow 1: Skim NanoAOD files**
        Process original NanoAOD → Apply event selection → Save filtered events to disk
        Use when: First time processing, want to create skimmed files for later analysis
        Config: save_skimmed_output=True, run_analysis=False

    **Workflow 2: Analyze pre-skimmed files**
        Load skimmed files → Run analysis → Fill histograms
        Use when: Analyzing previously skimmed data multiple times
        Config: save_skimmed_output=False, run_analysis=True

    Pipeline Stages
    ---------------
    - **Skimming** (via intccms.skimming.pipeline):
      - Event selection filter ALWAYS applies
      - When save_skimmed_output=True: Saves filtered events to disk in configured format
        (Parquet or ROOT). Writer automatically appends correct file extension.
      - When save_skimmed_output=False: No disk I/O (useful for analyzing pre-skimmed data)

    - **Analysis** (via NonDiffAnalysis):
      - When run_analysis=True: Object selection, corrections, observable calculations
      - When run_histogramming=True: Fill histograms during analysis
      - When run_systematics=True: Apply systematic variations

    - **Output Saving**:
      - Skimmed files: Saved during process() with manifest.json for tracking
      - Histograms: Auto-saved in postprocess() to:
        * processor_histograms.pkl (for caching, load with run_processor=False)
        * histograms.root (for downstream analysis tools)

    Configuration Flags
    -------------------
    config.general flags control workflow behavior:
    - save_skimmed_output: Save filtered events to disk (filter always applies)
    - run_analysis: Run analysis (object selection, corrections, observables)
    - run_histogramming: Fill histograms during analysis
    - run_systematics: Apply systematic variations
    - run_statistics: (handled post-processing, not in processor)

    User controls executor, fileset, and chunk size externally in their script.

    Attributes
    ----------
    config : Config
        Full analysis configuration
    output_manager : OutputDirectoryManager
        Manager for output directory paths
    metadata_lookup : Dict[str, Dict[str, Any]]
        Pre-built metadata lookup mapping fileset_key to metadata dict

    Examples
    --------
    **Example 1: Skim original NanoAOD files**

    >>> from coffea.processor import Runner, FuturesExecutor
    >>> from intccms.analysis import UnifiedProcessor
    >>> from intccms.metadata_extractor import DatasetMetadataManager
    >>>
    >>> # Generate metadata from NanoAOD files
    >>> generator = DatasetMetadataManager(dataset_manager, output_manager)
    >>> generator.run(generate_metadata=True)
    >>> metadata_lookup = generator.build_metadata_lookup()
    >>> fileset = generator.get_coffea_fileset()  # Original NanoAOD
    >>>
    >>> # Configure for skimming
    >>> config.general.save_skimmed_output = True
    >>> config.general.run_analysis = False
    >>>
    >>> processor = UnifiedProcessor(
    ...     config=config,
    ...     output_manager=output_manager,
    ...     metadata_lookup=metadata_lookup,
    ... )
    >>>
    >>> runner = Runner(executor=FuturesExecutor(), schema=NanoAODSchema)
    >>> output = runner(fileset, "Events", processor_instance=processor)
    >>> # Skimmed files saved to output_manager.skimmed_dir

    **Example 2: Analyze pre-skimmed files**

    >>> from intccms.skimming import FilesetManager
    >>>
    >>> # Load metadata from original run
    >>> generator = DatasetMetadataManager(dataset_manager, output_manager)
    >>> generator.run(generate_metadata=False)  # Read existing metadata
    >>> metadata_lookup = generator.build_metadata_lookup()
    >>>
    >>> # Build fileset from skimmed files
    >>> skimmed_dir = output_manager.skimmed_dir
    >>> fileset_manager = FilesetManager(skimmed_dir, format="parquet")
    >>> skimmed_fileset = fileset_manager.build_fileset_from_datasets(generator.datasets)
    >>>
    >>> # Convert to workitems
    >>> from coffea.processor.executor import WorkItem
    >>> workitems = []
    >>> for dataset_name, info in skimmed_fileset.items():
    ...     for file_path in info["files"]:
    ...         workitems.append(WorkItem(
    ...             dataset=dataset_name,
    ...             filename=file_path,
    ...             treename=info["metadata"].get("treename", "Events"),
    ...             entrystart=0,
    ...             entrystop=-1,
    ...         ))
    >>>
    >>> # Configure for analysis only
    >>> config.general.save_skimmed_output = False  # Don't re-skim
    >>> config.general.run_analysis = True
    >>> config.general.run_histogramming = True
    >>>
    >>> processor = UnifiedProcessor(
    ...     config=config,
    ...     output_manager=output_manager,
    ...     metadata_lookup=metadata_lookup,
    ... )
    >>>
    >>> runner = Runner(executor=FuturesExecutor(), schema=NanoAODSchema)
    >>> output = runner({}, "Events", processor_instance=processor, items=workitems)
    >>> # Histograms auto-saved to processor_histograms.pkl and histograms.root
    """

    def __init__(
        self,
        config: Config,
        output_manager: OutputDirectoryManager,
        metadata_lookup: Dict[str, Dict[str, Any]],
    ):
        """Initialize UnifiedProcessor.

        Parameters
        ----------
        config : Config
            Full analysis configuration with general.run_* flags
        output_manager : OutputDirectoryManager
            For resolving output paths
        metadata_lookup : Dict[str, Dict[str, Any]]
            Pre-built metadata lookup from NanoAODMetadataGenerator.build_metadata_lookup()
            Maps fileset_key -> {process, variation, xsec, nevts, is_data, dataset}
        """
        self.config = config
        self.output_manager = output_manager
        self.metadata_lookup = metadata_lookup

        # Initialize skimming components (always needed for filtering)
        self._init_skimming_components()

        # Always create NonDiffAnalysis instance
        # The run_analysis flag controls whether we execute its methods
        self.analysis = NonDiffAnalysis(
            config=config,
            output_manager=output_manager,
        )

        logger.info(
            f"Initialized UnifiedProcessor: "
            f"save_skimmed_output={config.general.save_skimmed_output}, "
            f"analysis={config.general.run_analysis}, "
            f"histogramming={config.general.run_histogramming}, "
            f"systematics={config.general.run_systematics}"
        )


    def _init_skimming_components(self):
        """Initialize skimming pipeline components."""
        self.skim_config = self.config.preprocess.skimming

        # Pre-build column lists
        preprocess_cfg = self.config.preprocess
        self.columns_to_keep, self.mc_only_columns = build_column_list(
            preprocess_cfg.branches,
            preprocess_cfg.get("mc_branches"),
            is_data=False,  # Will filter per-chunk
        )

        # Initialize writer if we need to save to disk
        self.writer = get_writer(self.skim_config.output.format)

        # Resolve lazy values in writer kwargs
        self.writer_kwargs = resolve_lazy_values(self.skim_config.output.to_kwargs or {})

    @property
    def accumulator(self):
        """Define accumulator structure based on enabled stages.

        Returns
        -------
        dict
            Accumulator dict with histograms, counters, etc.
        """
        acc = {"processed_events": 0}

        if self.config.general.run_histogramming:
            # Use histograms from NonDiffAnalysis instance
            # Coffea will automatically merge these across chunks via hist.Hist.__add__
            acc["histograms"] = self.analysis.nD_hists_per_region

        if self.config.general.save_skimmed_output:
            acc["skimmed_events"] = 0
            acc["manifest_entries"] = []

        return acc

    @track_metrics
    def process(self, events: ak.Array) -> Dict[str, Any]:
        """Process a single chunk of events.

        This is called by coffea's Runner for each chunk from the fileset.

        Parameters
        ----------
        events : ak.Array
            NanoEvents array from coffea (has NanoAODSchema applied)

        Returns
        -------
        dict
            Accumulator dict with histograms, counters, etc.
        """
        # Initialize output accumulator
        output = self.accumulator

        # Get chunk metadata
        dataset_name = events.metadata["dataset"]
        metadata = self.metadata_lookup.get(dataset_name)

        if not metadata:
            logger.warning(f"No metadata found for dataset {dataset_name}, skipping chunk")
            output["processed_events"] = 0
            return output

        # Track total input events
        input_events_count = len(events)

        # Step 1: Apply skim selection (always applies filter)
        with track_time(self, "skim_selection"):
            events = self._apply_skim_selection(events)
        output["skimmed_events"] = len(events)

        # Save filtered events to disk only if save_skimmed_output is enabled
        if self.config.general.save_skimmed_output and len(events) > 0:
            with track_time(self, "save_skimmed"):
                manifest_entry = self._save_skimmed_events(events, metadata)
            output["manifest_entries"] = [manifest_entry]

        # Step 2: Run analysis if enabled
        if self.config.general.run_analysis and len(events) > 0:
            # NonDiffAnalysis.process() handles:
            # - Object selection and corrections
            # - Histogram filling (if run_histogramming=True)
            # - Systematics (if run_systematics=True)
            with track_time(self, "analysis"):
                self.analysis.process(events=events, metadata=metadata)

            # Step 3: Collect histograms if histogramming was enabled
            if self.config.general.run_histogramming:
                # Histograms are filled in-place in self.analysis.nD_hists_per_region
                # Coffea will merge them across chunks
                output["histograms"] = self.analysis.nD_hists_per_region

        # Track total events processed (input events, not after filtering)
        output["processed_events"] = input_events_count
        return output

    def _apply_skim_selection(self, events: ak.Array) -> ak.Array:
        """Apply skim selection to events.

        Parameters
        ----------
        events : ak.Array
            Input events

        Returns
        -------
        ak.Array
            Filtered events after skim selection
        """
        executor = SelectionExecutor(self.skim_config)
        selection_mask = executor.execute(events)
        filtered_events = events[selection_mask]

        logger.debug(
            f"Skim selection: {len(events)} → {len(filtered_events)} events"
        )

        return filtered_events

    def _save_skimmed_events(self, events: ak.Array, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Save filtered events to disk and return a manifest entry.

        Parameters
        ----------
        events : ak.Array
            Filtered events to save
        metadata : dict
            Metadata dict with process, variation, dataset info

        Returns
        -------
        dict
            ManifestEntry-compatible dict tracking the output file
        """
        # Extract columns based on is_data flag
        is_data = metadata.get("is_data", False)
        output_columns = extract_columns(
            events,
            self.columns_to_keep,
            mc_only_columns=self.mc_only_columns,
            is_data=is_data,
        )

        dataset_name = metadata.get("dataset", "unknown")
        process = metadata.get("process", "unknown")
        variation = metadata.get("variation", "nominal")

        chunk_id = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]

        base_dir = (
            self.skim_config.output.output_dir
            or str(self.output_manager.skimmed_dir)
        ).rstrip("/")
        output_path = f"{base_dir}/{dataset_name}/{chunk_id}"

        # Prepare writer kwargs
        writer_kwargs = self.writer_kwargs.copy()
        if self.skim_config.output.format in ("ttree", "rntuple"):
            writer_kwargs["tree_name"] = self.skim_config.tree_name

        # Save events and capture actual output path (with extension)
        actual_path = save_events(self.writer, output_columns, output_path, **writer_kwargs)

        logger.debug(
            f"Saved {len(events)} events for {process}/{variation} to {actual_path}"
        )

        return {
            "source_file": events.metadata["filename"],
            "entrystart": events.metadata["entrystart"],
            "entrystop": events.metadata["entrystop"],
            "dataset": dataset_name,
            "treename": self.skim_config.tree_name,
            "output_file": actual_path,
            "processed_events": len(events),
            "total_events": events.metadata["entrystop"] - events.metadata["entrystart"],
        }

    def _write_manifests(self, manifest_entries: List[Dict[str, Any]]) -> None:
        """Write manifest JSON files grouping output files by dataset.

        Mirrors the logic in ``skimming/dask.py._save_manifest()``.
        Manifests are always written locally to ``output_manager.skimmed_dir``,
        even when output files are on remote storage.

        Parameters
        ----------
        manifest_entries : list of dict
            ManifestEntry-compatible dicts from all chunks
        """
        if not manifest_entries:
            logger.info("No manifest entries to save")
            return

        by_dataset = defaultdict(list)
        for entry in manifest_entries:
            by_dataset[entry["dataset"]].append(entry)

        base_dir = Path(self.output_manager.skimmed_dir)
        for dataset, entries in by_dataset.items():
            manifest_path = base_dir / dataset / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(manifest_path, "w") as f:
                json.dump(entries, f, indent=2)

            logger.info(f"Saved manifest for {dataset}: {manifest_path} ({len(entries)} entries)")

    def postprocess(self, accumulator: Dict) -> Dict:
        """Finalize accumulator after all chunks processed.

        Called once by coffea after all chunks are merged.
        Saves manifests and histograms to disk, and optionally runs statistical analysis.

        Parameters
        ----------
        accumulator : dict
            Merged accumulator from all chunks

        Returns
        -------
        dict
            Final accumulator
        """
        logger.info(
            f"Postprocessing complete: {accumulator.get('processed_events', 0)} total events"
        )

        # Write skimming manifests if any were produced
        if accumulator.get("manifest_entries"):
            self._write_manifests(accumulator["manifest_entries"])

        # Save histograms to disk if they were produced
        if self.config.general.run_histogramming and "histograms" in accumulator:
            # Save pickle format (for loading when run_processor=False)
            histograms_pkl = self.output_manager.histograms_dir / "processor_histograms.pkl"
            save_histograms_to_pickle(
                accumulator["histograms"],
                output_file=histograms_pkl,
            )
            logger.info(f"Saved processor histograms (pickle) to {histograms_pkl}")

            # Save ROOT format (for downstream tools/visualization)
            histograms_root = self.output_manager.histograms_dir / "histograms.root"
            save_histograms_to_root(
                accumulator["histograms"],
                output_file=histograms_root,
            )
            logger.info(f"Saved processor histograms (ROOT) to {histograms_root}")

            # Run statistical analysis if enabled
            if (self.config.general.run_statistics
                and self.config.statistics
                and self.config.statistics.cabinetry_config):

                logger.info("Running statistical analysis...")

                # Verify cabinetry config exists
                cabinetry_config_path = Path(self.config.statistics.cabinetry_config)
                if not cabinetry_config_path.exists():
                    logger.warning(
                        f"Cabinetry config not found: {cabinetry_config_path}. "
                        "Skipping statistics step."
                    )
                    return accumulator

                # Set histograms in the analysis instance
                self.analysis.nD_hists_per_region = accumulator["histograms"]

                # Run the statistics
                try:
                    self.analysis.run_statistics(str(cabinetry_config_path))
                    stats_dir = self.output_manager.statistics_dir
                    logger.info(f"Statistical analysis complete. Results saved to {stats_dir}")
                except Exception as e:
                    logger.error(f"Statistics failed: {e}", exc_info=True)
                    # Don't raise - allow workflow to complete

        return accumulator
