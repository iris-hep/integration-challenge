"""High-level orchestration for processor-based workflow.

This module provides a clean entry point for running the UnifiedProcessor
workflow, handling both the full processor execution and the histogram loading
path (for iterating on statistical models without re-processing).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from coffea.nanoevents import NanoAODSchema
from coffea.processor import Runner, WorkItem

from intccms.analysis.processor import UnifiedProcessor
from intccms.utils.output_files import load_histograms_from_pickle
from intccms.utils.output_manager import OutputDirectoryManager
from intccms.utils.schema import Config

logger = logging.getLogger(__name__)


def run_processor_workflow(
    config: Config,
    output_manager: OutputDirectoryManager,
    metadata_lookup: Dict[str, Dict[str, Any]],
    workitems: List[WorkItem],
    executor: Any,
    schema: Any = NanoAODSchema,
    chunksize: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute processor workflow or load saved histograms.

    This function provides a unified entry point for the processor-based workflow.
    When run_processor=True, it runs the UnifiedProcessor over data and saves
    histograms. When run_processor=False, it loads previously saved histograms,
    enabling fast iteration on statistical models without re-processing events.

    Parameters
    ----------
    config : Config
        Full analysis configuration with general.run_processor flag
    output_manager : OutputDirectoryManager
        Manager for output directory paths
    metadata_lookup : Dict[str, Dict[str, Any]]
        Pre-built metadata lookup from DatasetMetadataManager.build_metadata_lookup()
        Maps dataset_key -> {process, variation, xsec, nevts, is_data, dataset}
    workitems : List[WorkItem]
        Pre-generated work items from DatasetMetadataManager.workitems
    executor : Any
        Coffea executor (DaskExecutor, FuturesExecutor, etc.)
        User controls which executor to use
    schema : Any, optional
        NanoAOD schema for coffea, by default NanoAODSchema
    chunksize : int, optional
        Number of events per chunk, by default None (uses config value or 100k)

    Returns
    -------
    Dict[str, Any]
        Output dictionary containing:
        - "histograms": Filled histograms (if run_histogramming=True)
        - "processed_events": Number of input events (if run_processor=True)
        - "skimmed_events": Number of events after filtering (if run_processor=True)

    Raises
    ------
    FileNotFoundError
        If run_processor=False but no saved histograms exist

    Examples
    --------
    >>> # Full processor run
    >>> config.general.run_processor = True
    >>> output = run_processor_workflow(
    ...     config=config,
    ...     output_manager=output_manager,
    ...     metadata_lookup=metadata_lookup,
    ...     workitems=workitems,
    ...     executor=DaskExecutor(client=client),
    ... )
    >>> # Histograms saved automatically, ready for statistics

    >>> # Load saved histograms (iterate on statistics)
    >>> config.general.run_processor = False
    >>> output = run_processor_workflow(
    ...     config=config,
    ...     output_manager=output_manager,
    ...     metadata_lookup=metadata_lookup,
    ...     workitems=workitems,  # Not used when loading
    ...     executor=DaskExecutor(client=client),  # Not used when loading
    ... )
    >>> # Fast! No event processing, just loads histograms
    """
    if config.general.run_processor:
        logger.info("Running processor over data...")

        # Initialize UnifiedProcessor
        unified_processor = UnifiedProcessor(
            config=config,
            output_manager=output_manager,
            metadata_lookup=metadata_lookup,
        )

        # Determine chunksize
        if chunksize is None:
            chunksize = config.general.chunksize if hasattr(config.general, 'chunksize') else 100_000

        # Create coffea Runner
        runner = Runner(
            executor=executor,
            schema=schema,
            chunksize=chunksize,
        )

        # Run processor over workitems
        logger.info(f"Processing {len(workitems)} work items with chunksize={chunksize}")
        output = runner.run(
            workitems,
            processor_instance=unified_processor,
        )

        logger.info(
            f"Processor complete: {output.get('processed_events', 0):,} events processed, "
            f"{output.get('skimmed_events', 0):,} events after skim"
        )

        return output

    else:
        # Skip processor and load saved histograms
        logger.info("Skipping processor (run_processor=False)")
        logger.info("Loading previously saved histograms from disk...")

        histograms_pkl = output_manager.get_histograms_dir() / "processor_histograms.pkl"

        if not histograms_pkl.exists():
            raise FileNotFoundError(
                f"No saved histograms found at {histograms_pkl}. "
                f"Run with config.general.run_processor=True first to generate histograms."
            )

        histograms = load_histograms_from_pickle(histograms_pkl)
        logger.info(f"Loaded histograms from {histograms_pkl}")

        # Return in same format as processor output
        return {"histograms": histograms}
