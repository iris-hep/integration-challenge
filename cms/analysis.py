#!/usr/bin/env python3

"""
ZprimeAnalysis framework for applying object and event-level systematic corrections
on NanoAOD ROOT files and producing histograms of observables like mtt. Supports both
correctionlib-based and function-based corrections.
"""
import logging
import sys
import warnings

from intccms.analysis.nondiff import NonDiffAnalysis
from example_opendata.configs.configuration import config as ZprimeConfig
from intccms.datasets import DatasetManager
from intccms.utils.logging import setup_logging, log_banner
from intccms.schema import Config, load_config_with_restricted_cli
from intccms.metadata_extractor import DatasetMetadataManager
from intccms.skimming.manager import SkimmingManager
from intccms.utils.output import OutputDirectoryManager

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
    dataset_manager = DatasetManager(config.datasets)

    logger.info(log_banner("metadata and workitems extraction"))
    # Generate metadata and fileset from NanoAODs
    generator = DatasetMetadataManager(dataset_manager=dataset_manager, output_manager=output_manager)
    generator.run(generate_metadata=config.general.run_metadata_generation)
    datasets = generator.datasets
    workitems = generator.workitems
    if not workitems:
        logger.error("No workitems available. Please ensure metadata generation completed successfully.")
        sys.exit(1)
    if not datasets:
        logger.error("No datasets available. Please ensure metadata generation completed successfully.")
        sys.exit(1)

    # Build metadata lookup for downstream processing
    metadata_lookup = generator.build_metadata_lookup()

    logger.info(log_banner("SKIMMING AND PROCESSING"))
    logger.info(f"Processing {len(workitems)} workitems across {len(datasets)} datasets")

    # Process workitems and populate Dataset objects with events
    skimming_manager = SkimmingManager(
        config=config.preprocess.skimming,
        output_manager=output_manager,
    )
    datasets = skimming_manager.run(
        workitems=workitems,
        configuration=config,
        datasets=datasets,
        metadata_lookup=metadata_lookup,
        skip_skimming=not config.general.run_skimming,
        use_cache=config.general.read_from_cache,
    )


    analysis_mode = config.general.analysis
    if analysis_mode == "skip":
        logger.info(log_banner("Skim-Only Mode: Skimming Complete"))
        logger.info("✅ Skimming completed successfully. Analysis skipped as requested.")
        logger.info(f"Skimmed files are available in the configured output directories.")
        return
    elif analysis_mode == "nondiff":
        logger.info(log_banner("Running Non-Differentiable Analysis"))
        from intccms.utils.output import save_histograms_to_root

        nondiff_analysis = NonDiffAnalysis(config, output_manager)
        logger.info(f"Analysis initialized for {len(datasets)} datasets")

        # Process each dataset through the analysis pipeline
        histogram_count = 0
        for dataset in datasets:
            if dataset.events:
                for events, metadata in dataset.events:
                    logger.info(f"Processing {len(events)} events for {metadata['process']}")
                    nondiff_analysis.process(events, metadata)
                    histogram_count += 1

        # Save histograms to ROOT file
        if config.general.run_histogramming:
            histograms_output = output_manager.get_histograms_dir() / "histograms.root"
            save_histograms_to_root(
                nondiff_analysis.nD_hists_per_region,
                output_file=histograms_output,
            )
            logger.info(f"✅ Histograms saved to: {histograms_output}")
            logger.info(f"📈 Generated histograms for channels: {[ch.name for ch in config.channels]}")
            logger.info(f"📊 Processed {histogram_count} dataset chunks")

        logger.info(log_banner("ANALYSIS COMPLETE"))
        logger.info("✅ Non-differentiable analysis completed successfully!")


if __name__ == "__main__":
    main()
