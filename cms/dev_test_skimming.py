"""
Development test script for skimming workflow.

This script provides a minimal setup for testing skimming changes locally
with just 1 file per dataset. Use this for rapid iteration during development.

Usage:
    python dev_test_skimming.py
"""

import copy
from dask.distributed import Client, LocalCluster

from user.configuration_demo import config as original_config
from utils.schema import Config, load_config_with_restricted_cli
from utils.output_manager import OutputDirectoryManager
from utils.metadata_extractor import NanoAODMetadataGenerator
from utils.datasets import ConfigurableDatasetManager
from utils.skimming import process_workitems_with_skimming


def main():
    print("=" * 60)
    print("Development Skimming Test")
    print("=" * 60)

    # Configuration setup
    config = copy.deepcopy(original_config)

    # Limit to 1 file per dataset for fast testing
    config["datasets"]["max_files"] = 1

    # Use local output directory for development
    config["general"]["output_dir"] = "dev_outputs"

    # Disable caching to test fresh runs
    config["general"]["read_from_cache"] = False
    config["general"]["run_skimming"] = True
    config["general"]["run_metadata_generation"] = True

    # Disable subsequent steps (only test skimming)
    config["general"]["run_histogramming"] = False
    config["general"]["run_statistics"] = False
    config["general"]["run_systematics"] = False

    # Optional: limit to specific datasets for even faster testing
    config["general"]["processes"] = ["signal", "ttbar_semilep"]

    cli_args = []
    full_config = load_config_with_restricted_cli(config, cli_args)
    validated_config = Config(**full_config)

    print(f"\n✅ Configuration loaded with max_files={validated_config.datasets.max_files}")

    # Output manager setup
    output_manager = OutputDirectoryManager(
        root_output_dir=validated_config.general.output_dir,
        cache_dir=validated_config.general.cache_dir,
        metadata_dir=validated_config.general.metadata_dir,
        skimmed_dir=validated_config.general.skimmed_dir
    )
    print(f"✅ Output directory: {output_manager.root_output_dir}")

    # Setup local dask cluster
    print("\n🔧 Setting up local Dask cluster...")
    cluster = LocalCluster(n_workers=2, processes=True, threads_per_worker=1)
    client = Client(cluster)
    print(f"✅ Dask dashboard: {cluster.dashboard_link}")

    try:
        # Metadata extraction
        print("\n📋 Extracting metadata...")
        dataset_manager = ConfigurableDatasetManager(validated_config.datasets)
        metadata_generator = NanoAODMetadataGenerator(
            dataset_manager=dataset_manager,
            output_manager=output_manager,
            dask=(client, cluster),
        )
        metadata = metadata_generator.run(
            processes_filter=validated_config.general.processes if hasattr(validated_config.general, 'processes') else None
        )

        datasets = metadata_generator.datasets
        workitems = metadata_generator.workitems

        print(f"✅ Generated {len(workitems)} workitems across {len(datasets)} datasets")

        # Skimming
        print("\n⚡ Running skimming...")
        datasets = process_workitems_with_skimming(
            workitems,
            validated_config,
            output_manager,
            datasets,
            metadata_generator.nanoaods_summary
        )

        # Display results
        print("\n" + "=" * 60)
        print("✅ Skimming complete! Results:")
        print("=" * 60)

        total_events = 0
        for dataset in datasets:
            if dataset.events:
                for events, metadata in dataset.events:
                    num_events = len(events)
                    total_events += num_events
                    print(f"  📁 {dataset.name}: {num_events:,} events, {len(events.fields)} branches")

        print(f"\n📊 Total: {total_events:,} events processed")
        print(f"💾 Skimmed files: {output_manager.get_skimmed_dir()}")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        client.close()
        cluster.close()
        print("✅ Done!")


if __name__ == "__main__":
    main()
