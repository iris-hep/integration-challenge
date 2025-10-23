#!/usr/bin/env python3

"""
ZprimeAnalysis framework for applying object and event-level systematic corrections
on NanoAOD ROOT files and producing histograms of observables like mtt. Supports both
correctionlib-based and function-based corrections.
"""
import logging
import sys
import warnings

from analysis.nondiff import NonDiffAnalysis
from example_opendata.configs.configuration import config as ZprimeConfig
from utils.datasets import ConfigurableDatasetManager
from utils.logging import setup_logging, log_banner
from utils.schema import Config, load_config_with_restricted_cli
from utils.metadata_extractor import NanoAODMetadataGenerator
from utils.skimming import process_and_load_events
from utils.output_manager import OutputDirectoryManager

# -----------------------------
# Logging Configuration
# -----------------------------
setup_logging()

logger = logging.getLogger("AnalysisDriver")

# -----------------------------
# Main Driver
# -----------------------------
def main():
    """
    Main driver function for running the Zprime analysis framework.
    Loads configuration, runs preprocessing, and dispatches analysis over datasets.
    """
    cli_args = sys.argv[1:]
    full_config = load_config_with_restricted_cli(ZprimeConfig, cli_args)
    config = Config(**full_config)  # Pydantic validation
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

    logger.info(log_banner("metadata and workitems extraction"))
    # Generate metadata and fileset from NanoAODs
    generator = NanoAODMetadataGenerator(dataset_manager=dataset_manager, output_manager=output_manager)
    generator.run(generate_metadata=config.general.run_metadata_generation)
    datasets = generator.datasets
    workitems = generator.workitems
    if not workitems:
        logger.error("No workitems available. Please ensure metadata generation completed successfully.")
        sys.exit(1)
    if not datasets:
        logger.error("No datasets available. Please ensure metadata generation completed successfully.")
        sys.exit(1)

    logger.info(log_banner("SKIMMING AND PROCESSING"))
    logger.info(f"Processing {len(workitems)} workitems across {len(datasets)} datasets")

    # Process workitems and populate Dataset objects with events
    datasets = process_and_load_events(workitems, config, output_manager, datasets, generator.nanoaods_summary)


    analysis_mode = config.general.analysis
    if analysis_mode == "skip":
        logger.info(log_banner("Skim-Only Mode: Skimming Complete"))
        logger.info("✅ Skimming completed successfully. Analysis skipped as requested.")
        logger.info(f"Skimmed files are available in the configured output directories.")
        return
    elif analysis_mode == "nondiff":
        logger.info(log_banner("Running Non-Differentiable Analysis"))
        nondiff_analysis = NonDiffAnalysis(config, datasets, output_manager)
        nondiff_analysis.run_analysis_chain()


if __name__ == "__main__":
    main()
