#!/usr/bin/env python3

"""
Processor-based analysis workflow using UnifiedProcessor.

This script supports a flexible 2-stage workflow for efficient analysis:

**Stage 1: Skim NanoAOD files (First Pass)**
    python analysis_processor.py \
        general.use_skimmed_input=False \
        general.save_skimmed_output=True \
        general.run_analysis=False

    Reads original NanoAOD, applies event selection, saves filtered events to disk.

**Stage 2: Analyze pre-skimmed files (Second Pass)**
    python analysis_processor.py \
        general.use_skimmed_input=True \
        general.run_analysis=True \
        general.run_histogramming=True

    Automatically loads skimmed files, runs analysis, fills histograms.
    No manual fileset building needed - everything is automated!

**Alternative: Single-pass workflow (no skimming)**
    python analysis_processor.py \
        general.use_skimmed_input=False \
        general.save_skimmed_output=False \
        general.run_analysis=True

    Processes NanoAOD directly, filter in-memory, analyze (no disk I/O).
"""
import logging
import sys

from coffea import processor
from coffea.nanoevents import NanoAODSchema

from intccms.analysis import run_processor_workflow
from example_opendata.configs.configuration import config as ZprimeConfig
from intccms.datasets import DatasetManager
from intccms.utils.logging import setup_logging, log_banner
from intccms.utils.schema import Config, load_config_with_restricted_cli
from intccms.metadata_extractor import DatasetMetadataManager
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
    Main driver function for processor-based analysis workflow.

    Supports two workflows:
    - NanoAOD → Skim → Analyze (2-stage, efficient for multiple analyses)
    - NanoAOD → Analyze (single-pass, quick for one-off analyses)

    When use_skimmed_input=True, automatically loads pre-skimmed files.
    When use_skimmed_input=False, processes original NanoAOD files.
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

    # Check datasets config
    if not config.datasets:
        logger.error("Missing 'datasets' configuration; required for metadata.")
        sys.exit(1)
    dataset_manager = DatasetManager(config.datasets)

    # Metadata generation needed for both workflows (provides metadata_lookup)
    logger.info(log_banner("METADATA GENERATION"))
    generator = DatasetMetadataManager(
        dataset_manager=dataset_manager,
        output_manager=output_manager
    )

    # Generate metadata if needed, otherwise load from cache
    # For skimmed input, we still need metadata but not workitems from NanoAOD
    if config.general.use_skimmed_input:
        logger.info("Using skimmed input - loading metadata from cache")
        generator.run(generate_metadata=False)
    else:
        logger.info("Using NanoAOD input - generating/loading metadata and workitems")
        generator.run(generate_metadata=config.general.run_metadata_generation)

    if not generator.datasets:
        logger.error("No datasets available. Please ensure metadata exists.")
        sys.exit(1)

    # Build metadata lookup (needed for both workflows)
    metadata_lookup = generator.build_metadata_lookup()
    logger.info(f"Loaded metadata for {len(metadata_lookup)} dataset variations")

    # Get workitems only if not using skimmed input
    if config.general.use_skimmed_input:
        workitems = None  # Will be auto-generated from skimmed files
        logger.info("Workitems will be auto-generated from skimmed files")
    else:
        if not generator.workitems:
            logger.error("No workitems available from NanoAOD metadata generation.")
            sys.exit(1)
        workitems = generator.workitems
        logger.info(f"Generated {len(workitems)} workitems from NanoAOD files")

    logger.info(log_banner("PROCESSOR-BASED WORKFLOW"))

    logger.info(f"Workflow configuration:")
    logger.info(f"  - use_skimmed_input: {config.general.use_skimmed_input}")
    logger.info(f"  - save_skimmed_output: {config.general.save_skimmed_output}")
    logger.info(f"  - run_processor: {config.general.run_processor}")
    logger.info(f"  - run_analysis: {config.general.run_analysis}")
    logger.info(f"  - run_histogramming: {config.general.run_histogramming}")
    logger.info(f"  - run_systematics: {config.general.run_systematics}")

    # Choose executor based on configuration
    # User has full control over the executor choice
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

    # Run the processor workflow (or load saved histograms)
    # If use_skimmed_input=True and workitems=None, fileset auto-built from skimmed files
    output = run_processor_workflow(
        config=config,
        output_manager=output_manager,
        metadata_lookup=metadata_lookup,
        workitems=workitems,
        executor=executor,
        schema=NanoAODSchema,
    )

    logger.info(log_banner("RESULTS"))

    # Log summary (histograms and statistics auto-saved by processor)
    if output and "histograms" in output:
        num_histograms = sum(len(hists) for hists in output["histograms"].values())
        logger.info(f"📊 Total histograms produced: {num_histograms}")
        logger.info(f"✅ Histograms auto-saved to: {output_manager.get_histograms_dir()}")
        logger.info(f"   - processor_histograms.pkl (for loading with run_processor=False)")
        logger.info(f"   - histograms.root (for downstream tools)")

        # Statistics runs automatically if run_statistics=True
        if config.general.run_statistics:
            logger.info(f"📈 Statistical analysis auto-saved to: {output_manager.get_statistics_dir()}")
            logger.info(f"   - workspace.json (cabinetry workspace)")
            logger.info(f"   - Pre-fit and post-fit plots")
    else:
        logger.info("No histograms produced (run_histogramming may be disabled)")

    logger.info(log_banner("WORKFLOW COMPLETE"))
    logger.info("✅ Processor-based analysis completed successfully!")


if __name__ == "__main__":
    main()
