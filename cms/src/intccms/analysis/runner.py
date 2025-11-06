"""High-level orchestration for processor-based workflow.

This module provides a clean entry point for running the UnifiedProcessor
workflow, handling both the full processor execution and the histogram loading
path (for iterating on statistical models without re-processing).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from lzma import LZMAError

from coffea.nanoevents import NanoAODSchema
from coffea.processor import Runner
from coffea.processor.executor import WorkItem
from coffea.processor.executor import UprootMissTreeError
from dask.distributed import performance_report

from intccms.analysis.processor import UnifiedProcessor
from intccms.skimming import FilesetManager
from intccms.utils.filters import filter_by_process
from intccms.utils.output import (
    OutputDirectoryManager,
    load_histograms_from_pickle,
)
from intccms.schema import Config
from intccms.metrics import (
    start_tracking,
    stop_tracking,
    save_worker_timeline,
    collect_processing_metrics,
    save_measurement,
)

logger = logging.getLogger(__name__)


def run_processor_workflow(
    config: Config,
    output_manager: OutputDirectoryManager,
    metadata_lookup: Dict[str, Dict[str, Any]],
    workitems: Optional[List[WorkItem]] = None,
    executor: Any = None,
    schema: Any = NanoAODSchema,
    chunksize: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """Execute processor workflow or load saved histograms with optional metrics collection.

    This function provides a unified entry point for the processor-based workflow.
    When run_processor=True, it runs the UnifiedProcessor over data and saves
    histograms. When run_processor=False, it loads previously saved histograms,
    enabling fast iteration on statistical models without re-processing events.

    Metrics collection is controlled by config.general.metrics.enable. When enabled,
    collects throughput, event rates, worker utilization, and efficiency metrics.

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
    metrics : Dict[str, Any] or None
        Processing metrics (if config.general.metrics.enable=True), otherwise None

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

        # Setup metrics collection if enabled
        metrics_enabled = hasattr(config.general, 'metrics') and config.general.metrics.enable
        t0 = None
        measurement_path = None

        if metrics_enabled:
            t0 = time.perf_counter()
            measurement_path = output_manager.benchmarks_dir / "latest"

            # Start worker tracking if configured
            if config.general.metrics.track_workers and hasattr(executor, 'client'):
                client = executor.client
                try:
                    client.run_on_scheduler(start_tracking, interval=1.0)
                    logger.info("Started worker tracking on scheduler")
                except Exception as e:
                    logger.warning(f"Failed to start worker tracking: {e}")

        # Run processor over fileset or workitems
        if use_fileset:
            logger.info(f"Processing fileset with {len(fileset)} datasets, chunksize={chunksize}")
            # Convert our fileset format to coffea format
            coffea_fileset = {}
            for dataset_name, dataset_info in fileset.items():
                files = dataset_info["files"]
                treename = dataset_info["metadata"].get("treename", "Events")
                coffea_fileset[dataset_name] = {treename: files}

            # Wrap with performance report if metrics enabled
            if metrics_enabled:
                perf_report_path = measurement_path / "dask_performance.html"
                with performance_report(filename=str(perf_report_path)):
                    output, report = runner(
                        coffea_fileset,
                        treename="Events",  # Will be overridden by fileset structure
                        processor_instance=unified_processor,
                    )
            else:
                output, report = runner(
                    coffea_fileset,
                    treename="Events",  # Will be overridden by fileset structure
                    processor_instance=unified_processor,
                )
        else:
            logger.info(f"Processing {len(workitems)} work items with chunksize={chunksize}")

            # Wrap with performance report if metrics enabled
            if metrics_enabled:
                perf_report_path = measurement_path / "dask_performance.html"
                with performance_report(filename=str(perf_report_path)):
                    output, report = runner(
                        workitems,
                        processor_instance=unified_processor,
                    )
            else:
                output, report = runner(
                    workitems,
                    processor_instance=unified_processor,
                )

        logger.info(
            f"Processor complete: {output.get('processed_events', 0):,} events processed, "
            f"{output.get('skimmed_events', 0):,} events after skim"
        )

        # Collect metrics if enabled
        metrics = None
        if metrics_enabled:
            t1 = time.perf_counter()

            # Stop worker tracking and save timeline
            if config.general.metrics.track_workers and hasattr(executor, 'client'):
                try:
                    tracking_data = client.run_on_scheduler(stop_tracking)
                    save_worker_timeline(tracking_data, measurement_path)
                    logger.info(f"Saved worker timeline to {measurement_path}")
                except Exception as e:
                    logger.warning(f"Failed to save worker tracking data: {e}")

            # Collect all processing metrics
            try:
                metrics = collect_processing_metrics(
                    coffea_report=report,
                    t_start=t0,
                    t_end=t1,
                    custom_metrics=output.get("_metrics", None),
                    measurement_path=measurement_path if config.general.metrics.track_workers else None,
                )

                # Save measurements if configured
                if config.general.metrics.save_measurements:
                    save_measurement(metrics, t0, t1, measurement_path)
                    logger.info(f"Saved metrics measurement to {measurement_path}")

                # Log performance report location
                if config.general.metrics.track_workers:
                    logger.info(f"📊 Dask performance report: {perf_report_path}")

                # Log key metrics
                logger.info(
                    f"📈 Metrics: {metrics.get('overall_rate_gbps', 0):.2f} Gbps, "
                    f"{metrics.get('event_rate_wall_khz', 0):.1f} kHz"
                )

            except Exception as e:
                logger.error(f"Failed to collect metrics: {e}")
                metrics = None

        return output, report, metrics

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

        # Return in same format as processor output (no report or metrics when loading)
        return {"histograms": histograms}, {}, None
