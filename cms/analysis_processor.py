#!/usr/bin/env python3

"""
Processor-based analysis workflow using UnifiedProcessor.

This script demonstrates the new processor-based workflow that unifies
skimming and analysis into a single distributed coffea pipeline. The processor
handles both event selection and histogram filling in one pass over the data.
"""
import logging
import sys

from coffea import processor
from coffea.nanoevents import NanoAODSchema

from intccms.analysis.processor import UnifiedProcessor
from example_opendata.configs.configuration import config as ZprimeConfig
from intccms.utils.datasets import ConfigurableDatasetManager
from intccms.utils.logging import setup_logging, log_banner
from intccms.utils.schema import Config, load_config_with_restricted_cli
from intccms.metadata_extractor import DatasetMetadataManager
from intccms.utils.output_manager import OutputDirectoryManager
from intccms.utils.output_files import save_histograms_to_root

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
    Main driver function for running the processor-based analysis workflow.

    This workflow:
    1. Generates metadata and fileset from NanoAODs
    2. Builds metadata lookup for the processor
    3. Runs UnifiedProcessor with coffea.processor.Runner
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

    if not config.datasets:
        logger.error("Missing 'datasets' configuration; required for metadata/skimming.")
        sys.exit(1)
    dataset_manager = ConfigurableDatasetManager(config.datasets)

    logger.info(log_banner("METADATA AND FILESET GENERATION"))
    # Generate metadata and fileset from NanoAODs
    generator = DatasetMetadataManager(
        dataset_manager=dataset_manager,
        output_manager=output_manager
    )
    generator.run(generate_metadata=config.general.run_metadata_generation)

    if not generator.workitems:
        logger.error("No workitems available. Please ensure metadata generation completed successfully.")
        sys.exit(1)
    if not generator.datasets:
        logger.error("No datasets available. Please ensure metadata generation completed successfully.")
        sys.exit(1)

    # Build metadata lookup and get coffea fileset
    metadata_lookup = generator.build_metadata_lookup()
    fileset = generator.get_coffea_fileset()

    logger.info(f"Generated fileset with {len(fileset)} dataset variations")
    logger.info(f"Total workitems: {len(generator.workitems)}")

    logger.info(log_banner("PROCESSOR-BASED WORKFLOW"))

    # Initialize the unified processor
    unified_processor = UnifiedProcessor(
        config=config,
        output_manager=output_manager,
        metadata_lookup=metadata_lookup,
    )

    logger.info("Initialized UnifiedProcessor with:")
    logger.info(f"  - run_skimming: {config.general.run_skimming}")
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

    # Create the processor runner
    runner = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=config.general.chunksize if hasattr(config.general, 'chunksize') else 100_000,
    )

    # Run the processor over the fileset
    logger.info("Running processor over fileset...")
    output = runner(
        fileset,
        treename="Events",
        processor_instance=unified_processor,
    )

    logger.info(log_banner("SAVING OUTPUTS"))

    # Save histograms if they were produced
    if output and "histograms" in output:
        histograms_output = output_manager.get_histograms_dir() / "histograms.root"
        save_histograms_to_root(
            output["histograms"],
            output_file=histograms_output,
        )
        logger.info(f"✅ Histograms saved to: {histograms_output}")

        # Log summary
        num_histograms = sum(len(hists) for hists in output["histograms"].values())
        logger.info(f"📊 Total histograms produced: {num_histograms}")
    else:
        logger.info("No histograms produced (run_histogramming may be disabled)")

    logger.info(log_banner("WORKFLOW COMPLETE"))
    logger.info("✅ Processor-based analysis completed successfully!")


if __name__ == "__main__":
    main()
