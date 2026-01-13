"""High-level orchestration for processor-based workflow.

This module provides a clean entry point for running the UnifiedProcessor
workflow, handling both the full processor execution and the histogram loading
path (for iterating on statistical models without re-processing).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from lzma import LZMAError

from coffea.nanoevents import NanoAODSchema
from coffea.processor import Runner
from coffea.processor.executor import WorkItem
from coffea.processor.executor import UprootMissTreeError

from intccms.analysis.processor import UnifiedProcessor
from intccms.skimming import FilesetManager
from intccms.utils.filters import filter_by_process
from intccms.utils.output import (
    OutputDirectoryManager,
    load_histograms_from_pickle,
)
from intccms.schema import Config

logger = logging.getLogger(__name__)


def run_processor_workflow(
    config: Config,
    output_manager: OutputDirectoryManager,
    metadata_lookup: Dict[str, Dict[str, Any]],
    workitems: Optional[List[WorkItem]] = None,
    executor: Any = None,
    schema: Any = NanoAODSchema,
    chunksize: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute processor workflow or load saved histograms.

    This function provides a unified entry point for the processor-based workflow.
    When run_processor=True, it runs the UnifiedProcessor over data and saves
    histograms. When run_processor=False, it loads previously saved histograms,
    enabling fast iteration on statistical models without re-processing events.

    Metrics collection should be handled externally using roastcoffea's
    MetricsCollector context manager wrapping this function call.

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
    output : Dict[str, Any]
        Output dictionary containing:
        - "histograms": Filled histograms (if run_histogramming=True)
        - "processed_events": Number of input events (if run_processor=True)
        - "skimmed_events": Number of events after filtering (if run_processor=True)
    report : Dict[str, Any]
        Coffea's performance report (bytesread, entries, processtime, chunks)

    Raises
    ------
    FileNotFoundError
        If run_processor=False but no saved histograms exist

    Examples
    --------
    >>> # Full processor run with metrics via roastcoffea
    >>> from roastcoffea import MetricsCollector
    >>> with MetricsCollector(client=client, processor_instance=processor) as collector:
    ...     output, report = run_processor_workflow(
    ...         config=config,
    ...         output_manager=output_manager,
    ...         metadata_lookup=metadata_lookup,
    ...         workitems=workitems,
    ...         executor=DaskExecutor(client=client),
    ...     )
    ...     collector.extract_metrics_from_output(output)
    ...     collector.set_coffea_report(report)
    >>> metrics = collector.get_metrics()

    >>> # Load saved histograms (iterate on statistics)
    >>> config.general.run_processor = False
    >>> output, report = run_processor_workflow(
    ...     config=config,
    ...     output_manager=output_manager,
    ...     metadata_lookup=metadata_lookup,
    ... )
    >>> # Fast! No event processing, just loads histograms
    """
    if config.general.run_processor:
        logger.info("Running processor over data...")

        # Auto-build fileset from skimmed files if use_skimmed_input=True
        if config.general.use_skimmed_input:
            logger.info("Auto-detecting skimmed files (use_skimmed_input=True)...")

            skimmed_dir = Path(output_manager.skimmed_dir)
            if not skimmed_dir.exists():
                raise FileNotFoundError(
                    f"Skimmed directory does not exist: {skimmed_dir}\n"
                    "Run with use_skimmed_input=False and save_skimmed_output=True first "
                    "to create skimmed files."
                )

            # Build fileset from skimmed files
            fileset_manager = FilesetManager(
                skimmed_dir=skimmed_dir,
                format=config.preprocess.skimming.output.format
            )

            # Get datasets from metadata_lookup
            datasets = list(set(md['dataset'] for md in metadata_lookup.values()))
            fileset = fileset_manager.build_fileset(datasets)

            logger.info(f"Built fileset from {len(fileset)} skimmed datasets")

            # Convert fileset to coffea format and let coffea preprocess it
            # We'll use runner.run() with fileset instead of workitems
            use_fileset = True
        else:
            # Validate workitems exist when not using skimmed input
            if workitems is None:
                raise ValueError(
                    "No workitems provided and use_skimmed_input=False. "
                    "Either provide workitems or set use_skimmed_input=True."
                )
            use_fileset = False

        # Filter by process if configured
        if hasattr(config.general, 'processes') and config.general.processes:
            if use_fileset:
                fileset = filter_by_process(fileset, config.general.processes, metadata_lookup)
                if not fileset:
                    logger.warning("No datasets remain after process filtering")
                    return {"histograms": {}, "processed_events": 0, "skimmed_events": 0}
            else:
                workitems = filter_by_process(workitems, config.general.processes)
                if not workitems:
                    logger.warning("No workitems remain after process filtering")
                    return {"histograms": {}, "processed_events": 0, "skimmed_events": 0}

        # Initialize UnifiedProcessor
        unified_processor = UnifiedProcessor(
            config=config,
            output_manager=output_manager,
            metadata_lookup=metadata_lookup,
        )

        # Determine chunksize
        if chunksize is None:
            if hasattr(config, 'preprocess') and hasattr(config.preprocess, 'skimming'):
                chunksize = config.preprocess.skimming.chunk_size
            else:
                chunksize = 100_000

        # Create coffea Runner
        runner = Runner(
            executor=executor,
            schema=schema,
            chunksize=chunksize,
            savemetrics=True,
            skipbadfiles=(OSError, LZMAError, UprootMissTreeError, Exception),
        )

        # Run processor over fileset or workitems
        if use_fileset:
            logger.info(f"Processing fileset with {len(fileset)} datasets, chunksize={chunksize}")
            # Convert our fileset format to coffea format
            coffea_fileset = {}
            for dataset_name, dataset_info in fileset.items():
                files = dataset_info["files"]
                treename = dataset_info["metadata"].get("treename", "Events")
                coffea_fileset[dataset_name] = {treename: files}

            output, report = runner(
                coffea_fileset,
                treename="Events",  # Will be overridden by fileset structure
                processor_instance=unified_processor,
            )
        else:
            logger.info(f"Processing {len(workitems)} work items with chunksize={chunksize}")
            output, report = runner(
                workitems,
                processor_instance=unified_processor,
            )

        logger.info(
            f"Processor complete: {output.get('processed_events', 0):,} events processed, "
            f"{output.get('skimmed_events', 0):,} events after skim"
        )

        return output, report

    else:
        # Skip processor and load saved histograms
        logger.info("Skipping processor (run_processor=False)")
        logger.info("Loading previously saved histograms from disk...")

        histograms_pkl = output_manager.histograms_dir / "processor_histograms.pkl"

        if not histograms_pkl.exists():
            raise FileNotFoundError(
                f"No saved histograms found at {histograms_pkl}. "
                f"Run with config.general.run_processor=True first to generate histograms."
            )

        histograms = load_histograms_from_pickle(histograms_pkl)
        logger.info(f"Loaded histograms from {histograms_pkl}")

        # Return in same format as processor output (no report when loading)
        return {"histograms": histograms}, {}
