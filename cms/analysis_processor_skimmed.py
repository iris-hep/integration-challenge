#!/usr/bin/env python3

"""
Processor-based analysis workflow using pre-skimmed files.

This script demonstrates how to run analysis on previously skimmed data.
Instead of processing original NanoAOD files, it loads skimmed files created
by a previous run with run_skimming=True.

Usage:
------
# First run: Create skimmed files
python analysis_processor.py general.run_skimming=True general.run_analysis=False

# Second run: Analyze skimmed files (this script)
python analysis_processor_skimmed.py general.run_analysis=True general.run_histogramming=True
"""
import logging
import sys
from pathlib import Path

from coffea import processor
from coffea.nanoevents import NanoAODSchema

from intccms.analysis import run_processor_workflow
from example_opendata.configs.configuration import config as ZprimeConfig
from intccms.datasets import DatasetManager
from intccms.utils.logging import setup_logging, log_banner
from intccms.utils.schema import Config, load_config_with_restricted_cli
from intccms.metadata_extractor import DatasetMetadataManager
from intccms.skimming import FilesetManager
from intccms.utils.output_manager import OutputDirectoryManager

# -----------------------------
# Logging Configuration
# -----------------------------
setup_logging()
logger = logging.getLogger("ProcessorAnalysisDriver")


# -----------------------------
# Main Driver
# -----------------------------
def main():
    """
    Main driver for running analysis on pre-skimmed files.

    This workflow:
    1. Loads metadata from the original skimming run
    2. Builds fileset from skimmed files (using FilesetManager)
    3. Runs UnifiedProcessor with run_skimming=False
    4. Saves histograms from accumulated output
    """
    cli_args = sys.argv[1:]
    full_config = load_config_with_restricted_cli(ZprimeConfig, cli_args)
    config = Config(**full_config)
    logger.info(f"Luminosity: {config.general.lumi}")

    # Create centralized output directory manager
    output_manager = OutputDirectoryManager(
        root_output_dir=config.general.output_dir,
        cache_dir=config.general.cache_dir,
        metadata_dir=config.general.metadata_dir,
        skimmed_dir=config.general.skimmed_dir
    )
    logger.info(f"Output directory structure: {output_manager.list_structure()}")

    # Verify skimmed files exist
    skimmed_dir = Path(output_manager.get_skimmed_dir())
    if not skimmed_dir.exists():
        logger.error(
            f"Skimmed directory does not exist: {skimmed_dir}\n"
            "Please run skimming first with: python analysis_processor.py "
            "general.run_skimming=True general.run_analysis=False"
        )
        sys.exit(1)

    logger.info(log_banner("LOADING METADATA FROM ORIGINAL RUN"))

    # Dataset manager still needed for metadata lookup
    if not config.datasets:
        logger.error("Missing 'datasets' configuration; required for metadata lookup.")
        sys.exit(1)
    dataset_manager = DatasetManager(config.datasets)

    # Load metadata from original run (without regenerating)
    generator = DatasetMetadataManager(
        dataset_manager=dataset_manager,
        output_manager=output_manager
    )
    # Read existing metadata (generate_metadata=False)
    generator.run(generate_metadata=False)

    if not generator.datasets:
        logger.error(
            "No datasets found in metadata cache. "
            "Please ensure the initial skimming run completed successfully."
        )
        sys.exit(1)

    # Build metadata lookup (needed by processor for cross-sections, etc.)
    metadata_lookup = generator.build_metadata_lookup()
    logger.info(f"Loaded metadata for {len(metadata_lookup)} dataset variations")

    logger.info(log_banner("BUILDING FILESET FROM SKIMMED FILES"))

    # Build fileset from skimmed files using FilesetManager
    fileset_manager = FilesetManager(
        skimmed_dir=skimmed_dir,
        format=config.preprocess.skimming.output.format
    )
    skimmed_fileset = fileset_manager.build_fileset_from_datasets(generator.datasets)

    logger.info(f"Built fileset from skimmed files: {len(skimmed_fileset)} dataset variations")

    # Convert skimmed fileset to workitems
    # Extract file paths and build workitems from skimmed files
    import uuid
    from coffea.processor.executor import WorkItem
    workitems = []
    for dataset_name, dataset_info in skimmed_fileset.items():
        files = dataset_info["files"]
        metadata = dataset_info["metadata"]
        treename = metadata.get("treename", "Events")

        for file_path in files:
            # Create workitem for each skimmed file
            # Note: Skimmed files are typically smaller, so we can process entire files
            workitems.append(
                WorkItem(
                    dataset=dataset_name,
                    filename=file_path,
                    treename=treename,
                    entrystart=0,
                    entrystop=-1,  # Process entire file
                    fileuuid=uuid.uuid4(),
                )
            )

    logger.info(f"Created {len(workitems)} workitems from skimmed files")

    logger.info(log_banner("PROCESSOR-BASED WORKFLOW ON SKIMMED DATA"))

    # Ensure skimming is disabled (we're reading already-skimmed files)
    if config.general.run_skimming:
        logger.warning(
            "Forcing run_skimming=False because we're processing pre-skimmed files"
        )
        config.general.run_skimming = False

    logger.info(f"Workflow configuration:")
    logger.info(f"  - run_processor: {config.general.run_processor}")
    logger.info(f"  - run_skimming: {config.general.run_skimming} (forced False)")
    logger.info(f"  - run_analysis: {config.general.run_analysis}")
    logger.info(f"  - run_histogramming: {config.general.run_histogramming}")
    logger.info(f"  - run_systematics: {config.general.run_systematics}")

    # Choose executor based on configuration
    if hasattr(config.general, 'executor') and config.general.executor == 'dask':
        from coffea.processor import DaskExecutor
        from dask.distributed import Client

        logger.info("Using Dask executor")
        client = Client()  # Or connect to existing cluster
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        executor = DaskExecutor(client=client)
    else:
        logger.info("Using FuturesExecutor (default)")
        executor = processor.FuturesExecutor()

    # Run the processor workflow on skimmed files
    output = run_processor_workflow(
        config=config,
        output_manager=output_manager,
        metadata_lookup=metadata_lookup,
        workitems=workitems,  # Workitems from skimmed files
        executor=executor,
        schema=NanoAODSchema,
    )

    logger.info(log_banner("RESULTS"))

    # Log summary (histograms auto-saved by processor)
    if output and "histograms" in output:
        num_histograms = sum(len(hists) for hists in output["histograms"].values())
        logger.info(f"📊 Total histograms produced: {num_histograms}")
        logger.info(f"✅ Histograms auto-saved to: {output_manager.get_histograms_dir()}")
        logger.info(f"   - processor_histograms.pkl (for loading with run_processor=False)")
        logger.info(f"   - histograms.root (for downstream tools)")
    else:
        logger.info("No histograms produced (run_histogramming may be disabled)")

    logger.info(log_banner("WORKFLOW COMPLETE"))
    logger.info("✅ Analysis of skimmed files completed successfully!")


if __name__ == "__main__":
    main()
