"""Coffea processor for unified skimming and analysis workflow.

This module provides the UnifiedProcessor class that integrates skimming
and analysis into a single distributed coffea-based workflow. The processor
respects configuration flags to control which stages run.
"""

import logging
from typing import Any, Dict, List, Optional

import awkward as ak
from coffea.processor import ProcessorABC

from intccms.analysis.nondiff import NonDiffAnalysis
from intccms.utils.schema import Config
from intccms.utils.output_manager import OutputDirectoryManager
from intccms.utils.metadata import build_dataset_metadata_lookup

logger = logging.getLogger(__name__)


class UnifiedProcessor(ProcessorABC):
    """Coffea processor for distributed skimming and/or analysis.

    This processor integrates the skimming pipeline with the analysis workflow,
    controlled by configuration flags. It delegates to existing implementations:
    - Skimming: Uses pipeline stages from intccms.skimming.pipeline
    - Analysis: Delegates to NonDiffAnalysis.process()

    The processor respects these config.general flags:
    - run_skimming: Apply event selection (skim)
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
    datasets_metadata : List[Dataset]
        Dataset metadata for xsec, nevts, is_data, etc.

    Examples
    --------
    >>> # User's script (analysis.py)
    >>> from coffea.processor import Runner, IterativeExecutor
    >>> from intccms.analysis import UnifiedProcessor
    >>>
    >>> processor = UnifiedProcessor(
    ...     config=config,
    ...     output_manager=output_manager,
    ...     datasets_metadata=datasets,
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
        datasets_metadata: List[Any],
        nanoaods_summary: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize UnifiedProcessor.

        Parameters
        ----------
        config : Config
            Full analysis configuration with general.run_* flags
        output_manager : OutputDirectoryManager
            For resolving output paths
        datasets_metadata : List[Dataset]
            Dataset objects with metadata (xsec, nevts, is_data, etc.)
        nanoaods_summary : dict, optional
            Summary with nevts_total per dataset/variation from metadata generation
        """
        self.config = config
        self.output_manager = output_manager
        self.datasets_metadata = datasets_metadata
        self.nanoaods_summary = nanoaods_summary or {}

        # Build metadata lookup: dataset_name -> metadata dict
        self._metadata_lookup = self._build_metadata_lookup()

        # Initialize components based on enabled stages
        if config.general.run_skimming:
            self._init_skimming_components()

        if config.general.run_analysis:
            # Create NonDiffAnalysis instance
            # It handles run_histogramming and run_systematics internally
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

    def _build_metadata_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Build mapping from dataset name to metadata dict.

        Extracts nevts from nanoaods_summary (same logic as SkimmingManager).

        Returns
        -------
        dict
            Mapping of dataset_name -> {process, variation, xsec, nevts, is_data, ...}
        """
        lookup = {}
        for dataset in self.datasets_metadata:
            for fileset_key in dataset.fileset_keys:
                idx = dataset.fileset_keys.index(fileset_key)

                # Extract nevts from NanoAODs summary
                nevts = 0
                if self.nanoaods_summary:
                    # Extract dataset name from fileset_key (format: "datasetname__variation")
                    dataset_name_from_key = fileset_key.rsplit("__", 1)[0]
                    if dataset_name_from_key in self.nanoaods_summary:
                        if dataset.variation in self.nanoaods_summary[dataset_name_from_key]:
                            nevts = self.nanoaods_summary[dataset_name_from_key][
                                dataset.variation
                            ].get("nevts_total", 0)

                if nevts == 0:
                    logger.warning(f"No nevts found for {fileset_key}, using 0")

                lookup[fileset_key] = {
                    "process": dataset.process,
                    "variation": dataset.variation,
                    "xsec": dataset.cross_sections[idx],
                    "nevts": nevts,
                    "is_data": dataset.is_data,
                    "dataset": fileset_key,
                }
        return lookup

    def _init_skimming_components(self):
        """Initialize skimming pipeline components."""
        from intccms.skimming.pipeline.stages import build_column_list

        self.skim_config = self.config.preprocess.skimming

        # Pre-build column lists
        preprocess_cfg = self.config.preprocess
        self.columns_to_keep, self.mc_only_columns = build_column_list(
            preprocess_cfg.branches,
            preprocess_cfg.get("mc_branches"),
            is_data=False,  # Will filter per-chunk
        )

    @property
    def accumulator(self):
        """Define accumulator structure based on enabled stages.

        Returns
        -------
        dict
            Accumulator dict with histograms, counters, etc.
        """
        acc = {"processed_events": 0}

        if self.config.general.run_histogramming and hasattr(self, "analysis"):
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
        metadata = self._metadata_lookup.get(dataset_name)

        if not metadata:
            logger.warning(f"No metadata found for dataset {dataset_name}, skipping chunk")
            output["processed_events"] = 0
            return output

        # Step 1: Apply skimming if enabled
        if self.config.general.run_skimming:
            events = self._apply_skim_selection(events)
            output["skimmed_events"] = len(events)

        # Step 2: Run analysis if enabled
        if self.config.general.run_analysis and len(events) > 0:
            # NonDiffAnalysis.process() handles:
            # - Object selection and corrections
            # - Histogram filling (if run_histogramming=True)
            # - Systematics (if run_systematics=True)
            self.analysis.process(events=events, metadata=metadata)

            if self.config.general.run_histogramming:
                # Histograms are filled in-place in self.analysis.nD_hists_per_region
                # Coffea will merge them across chunks
                output["histograms"] = self.analysis.nD_hists_per_region

        output["processed_events"] = len(events)
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
        from intccms.skimming.pipeline.stages import apply_selection
        from intccms.utils.tools import get_function_arguments

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

    def postprocess(self, accumulator: Dict) -> Dict:
        """Finalize accumulator after all chunks processed.

        Called once by coffea after all chunks are merged.

        Parameters
        ----------
        accumulator : dict
            Merged accumulator from all chunks

        Returns
        -------
        dict
            Final accumulator
        """
        # For now, just return the merged accumulator
        # Future: Could add post-processing logic here (e.g., normalization)
        logger.info(
            f"Postprocessing complete: {accumulator.get('processed_events', 0)} total events"
        )
        return accumulator
