"""Coffea processor for unified skimming and analysis workflow.

This module provides the UnifiedProcessor class that integrates skimming
and analysis into a single distributed coffea-based workflow. The processor
respects configuration flags to control which stages run.
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import awkward as ak
from coffea.processor import ProcessorABC

from intccms.analysis.nondiff import NonDiffAnalysis
from intccms.skimming.io.writers import get_writer
from intccms.skimming.pipeline.stages import (
    apply_selection,
    build_column_list,
    extract_columns,
    save_events,
)
from intccms.skimming.workitem import resolve_lazy_values
from intccms.utils.output_files import (
    save_histograms_to_pickle,
    save_histograms_to_root,
)
from intccms.utils.output_manager import OutputDirectoryManager
from intccms.utils.schema import Config
from intccms.utils.tools import get_function_arguments

logger = logging.getLogger(__name__)


class UnifiedProcessor(ProcessorABC):
    """Coffea processor for distributed skimming and/or analysis.

    This processor integrates the skimming pipeline with the analysis workflow,
    controlled by configuration flags. It delegates to existing implementations:
    - Skimming: Uses pipeline stages from intccms.skimming.pipeline
      - Event selection filter ALWAYS applies
      - When run_skimming=True: Saves filtered events to disk in configured format
      - When run_skimming=False: No disk I/O (useful for analyzing pre-skimmed data)
    - Analysis: Delegates to NonDiffAnalysis.process()

    The processor respects these config.general flags:
    - run_skimming: Save filtered events to disk (filter always applies)
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
    >>> # User's script (analysis.py)
    >>> from coffea.processor import Runner, IterativeExecutor
    >>> from intccms.analysis import UnifiedProcessor
    >>>
    >>> # Generate metadata first
    >>> generator = NanoAODMetadataGenerator(...)
    >>> generator.run(generate_metadata=True)
    >>> metadata_lookup = generator.build_metadata_lookup()
    >>> fileset = generator.get_coffea_fileset()
    >>>
    >>> processor = UnifiedProcessor(
    ...     config=config,
    ...     output_manager=output_manager,
    ...     metadata_lookup=metadata_lookup,
    ... )
    >>>
    >>> runner = Runner(executor=IterativeExecutor(), schema=NanoAODSchema)
    >>> output = runner(fileset, "Events", processor_instance=processor)
    >>>
    >>> # Save histograms
    >>> save_histograms_to_root(output["histograms"], ...)
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
            f"skimming={config.general.run_skimming}, "
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

        if self.config.general.run_skimming:
            acc["skimmed_events"] = 0

        return acc

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
        events = self._apply_skim_selection(events)
        output["skimmed_events"] = len(events)

        # Save filtered events to disk only if run_skimming is enabled
        if self.config.general.run_skimming and len(events) > 0:
            self._save_skimmed_events(events, metadata)

        # Step 2: Run analysis if enabled
        if self.config.general.run_analysis and len(events) > 0:
            # NonDiffAnalysis.process() handles:
            # - Object selection and corrections
            # - Histogram filling (if run_histogramming=True)
            # - Systematics (if run_systematics=True)
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
        selection_func = self.skim_config.function
        selection_use = self.skim_config.use

        selection_args, selection_kwargs = get_function_arguments(
            selection_use,
            events,
            function_name=selection_func.__name__,
            static_kwargs=self.skim_config.get("static_kwargs"),
        )

        filtered_events = apply_selection(
            events, selection_func, selection_args, selection_kwargs
        )

        logger.debug(
            f"Skim selection: {len(events)} → {len(filtered_events)} events"
        )

        return filtered_events

    def _save_skimmed_events(self, events: ak.Array, metadata: Dict[str, Any]) -> None:
        """Save filtered events to disk in the configured format.

        Parameters
        ----------
        events : ak.Array
            Filtered events to save
        metadata : dict
            Metadata dict with process, variation, dataset info
        """
        # Extract columns based on is_data flag
        is_data = metadata.get("is_data", False)
        output_columns = extract_columns(
            events,
            self.columns_to_keep,
            mc_only_columns=self.mc_only_columns,
            is_data=is_data,
        )

        # Build output path
        # Use dataset name + unique chunk identifier
        dataset_name = metadata.get("dataset", "unknown")
        process = metadata.get("process", "unknown")
        variation = metadata.get("variation", "nominal")

        # Generate unique chunk ID
        chunk_id = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]

        # Determine file extension
        extension_map = {
            "parquet": ".parquet",
            "root_ttree": ".root",
            "rntuple": ".ntuple",
            "safetensors": ".safetensors",
        }
        extension = extension_map.get(self.skim_config.output.format, ".parquet")

        # Build path: skimmed_dir/dataset/chunk_id.ext
        output_dir = Path(self.output_manager.get_skimmed_dir()) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{chunk_id}{extension}"

        # Prepare writer kwargs
        writer_kwargs = self.writer_kwargs.copy()
        if self.skim_config.output.format == "root_ttree":
            writer_kwargs["tree_name"] = self.skim_config.tree_name

        # Save events
        save_events(self.writer, output_columns, str(output_path), **writer_kwargs)

        logger.debug(
            f"Saved {len(events)} events for {process}/{variation} to {output_path}"
        )

    def postprocess(self, accumulator: Dict) -> Dict:
        """Finalize accumulator after all chunks processed.

        Called once by coffea after all chunks are merged.
        Saves histograms to disk for later use without re-running processor.

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

        # Save histograms to disk if they were produced
        if self.config.general.run_histogramming and "histograms" in accumulator:
            # Save pickle format (for loading when run_processor=False)
            histograms_pkl = self.output_manager.get_histograms_dir() / "processor_histograms.pkl"
            save_histograms_to_pickle(
                accumulator["histograms"],
                output_file=histograms_pkl,
            )
            logger.info(f"Saved processor histograms (pickle) to {histograms_pkl}")

            # Save ROOT format (for downstream tools/visualization)
            histograms_root = self.output_manager.get_histograms_dir() / "histograms.root"
            save_histograms_to_root(
                accumulator["histograms"],
                output_file=histograms_root,
            )
            logger.info(f"Saved processor histograms (ROOT) to {histograms_root}")

        return accumulator
