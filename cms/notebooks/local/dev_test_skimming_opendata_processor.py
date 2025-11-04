"""
Development test script for processor-based workflow.

This script demonstrates the UnifiedProcessor workflow with coffea.processor.Runner,
including skimming, analysis, histogramming, and statistics steps.

Usage:
    python dev_test_skimming_opendata_processor.py
"""

import copy
from pathlib import Path

from coffea.nanoevents import NanoAODSchema
from coffea.processor import DaskExecutor
from dask.distributed import Client, LocalCluster

from example_opendata.configs.configuration import config as original_config
from intccms.analysis import run_processor_workflow
from intccms.metadata_extractor import DatasetMetadataManager
from intccms.datasets import DatasetManager
from intccms.utils.output import OutputDirectoryManager
from intccms.utils.schema import Config, load_config_with_restricted_cli


def main():
    print("=" * 60)
    print("Development Processor-Based Workflow Test")
    print("=" * 60)

    # Configuration setup
    config = copy.deepcopy(original_config)

    # Limit files for fast testing
    config["datasets"]["max_files"] = None

    # Use local output directory for development
    config["general"]["output_dir"] = "dev/dev_outputs_opendata_processor"

    # Disable caching to test fresh runs
    config["general"]["read_from_cache"] = False
    config["general"]["run_metadata_generation"] = False

    # Enable/disable stages
    config["general"]["run_processor"] = True  # Set to False to skip processor and load saved histograms
    config["general"]["save_skimmed_output"] = False
    config["general"]["run_analysis"] = True
    config["general"]["run_histogramming"] = True
    config["general"]["run_systematics"] = False
    config["general"]["run_statistics"] = True

    # Test only the signal dataset
    config["general"]["processes"] = ["signal"]

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
    cluster = LocalCluster(n_workers=8, processes=True, threads_per_worker=1)
    client = Client(cluster)
    print(f"✅ Dask dashboard: {cluster.dashboard_link}")

    try:
        # Step 1: Metadata extraction
        print("\n📋 Extracting metadata...")
        dataset_manager = DatasetManager(validated_config.datasets)
        metadata_generator = DatasetMetadataManager(
            dataset_manager=dataset_manager,
            output_manager=output_manager,
            executor=DaskExecutor(client=client),
            config=validated_config,
        )
        metadata_generator.run(
            generate_metadata=validated_config.general.run_metadata_generation,
            processes_filter=validated_config.general.processes if hasattr(validated_config.general, 'processes') else None
        )

        metadata_lookup = metadata_generator.build_metadata_lookup()
        workitems = metadata_generator.workitems

        print(f"✅ Generated {len(workitems)} workitems")

        # Diagnostic: Show workitem details
        print("\n🔍 Workitem Details:")
        for i, wi in enumerate(workitems[:5]):  # Show first 5
            print(f"  {i}: dataset='{wi.dataset}' process='{wi.usermeta.get('process', 'N/A')}'")
        if len(workitems) > 5:
            print(f"  ... and {len(workitems) - 5} more")

        # Step 2: Run processor workflow (or load saved histograms)
        print("\n🚀 Running processor workflow...")
        print(f"Configuration:")
        print(f"  - run_processor: {validated_config.general.run_processor}")
        print(f"  - save_skimmed_output: {validated_config.general.save_skimmed_output}")
        print(f"  - run_analysis: {validated_config.general.run_analysis}")
        print(f"  - run_histogramming: {validated_config.general.run_histogramming}")
        print(f"  - run_systematics: {validated_config.general.run_systematics}")

        output = run_processor_workflow(
            config=validated_config,
            output_manager=output_manager,
            metadata_lookup=metadata_lookup,
            workitems=workitems,
            executor=DaskExecutor(client=client),
            schema=NanoAODSchema,
        )

        # Step 3: Display results
        print("\n" + "=" * 60)
        print("📊 Results:")
        print("=" * 60)

        if validated_config.general.run_processor:
            print(f"📊 Total events processed: {output.get('processed_events', 0):,}")
            if 'skimmed_events' in output:
                print(f"✂️  Events after skim: {output.get('skimmed_events', 0):,}")

        # Histograms are auto-saved by processor
        if output and "histograms" in output:
            num_histograms = sum(len(hists) for hists in output["histograms"].values())
            print(f"📈 Total histograms: {num_histograms}")
            print(f"📈 Channels: {list(output['histograms'].keys())}")
            print(f"✅ Histograms auto-saved to: {output_manager.get_histograms_dir()}")
            print(f"   - processor_histograms.pkl (for loading with run_processor=False)")
            print(f"   - histograms.root (for downstream tools)")
        else:
            print("⚠️  No histograms produced (run_histogramming may be disabled)")

        # Statistics run automatically in processor.postprocess() if run_statistics=True
        if validated_config.general.run_statistics and output and "histograms" in output:
            print(f"\n📈 Statistical analysis auto-saved to: {output_manager.get_statistics_dir()}")
            print(f"   - workspace.json (cabinetry workspace)")
            print(f"   - Pre-fit and post-fit plots")

        print("\n" + "=" * 60)
        print("✅ Complete processor workflow finished!")
        print("=" * 60)

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        client.close()
        cluster.close()
        print("✅ Done!")


if __name__ == "__main__":
    main()
