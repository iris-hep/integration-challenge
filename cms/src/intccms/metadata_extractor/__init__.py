"""Dataset metadata extraction and management.

This package provides tools for extracting metadata from ROOT files (NanoAOD, MiniAOD, etc.),
building coffea-compatible filesets, and managing dataset metadata for analysis workflows.

The package is organized into modules with clear separation of concerns:
- core: Pure functions for data transformation and aggregation
- io: File system operations (reading listings, saving/loading JSON)
- builders: Coordinating builders that compose core + io functions
- extractor: Coffea integration for WorkItem extraction
- manager: High-level workflow orchestration

Main exports for users:
- DatasetMetadataManager: Main entry point for metadata generation
- FilesetBuilder: Build coffea filesets from dataset configurations
- CoffeaMetadataExtractor: Extract WorkItems from files

Usage:
------
>>> from intccms.metadata_extractor import DatasetMetadataManager
>>> from intccms.datasets import DatasetManager
>>>
>>> dataset_manager = DatasetManager(config)
>>> metadata_manager = DatasetMetadataManager(dataset_manager, output_manager)
>>> metadata_manager.run(generate_metadata=True)
>>>
>>> # Get outputs
>>> fileset = metadata_manager.get_coffea_fileset()
>>> metadata_lookup = metadata_manager.build_metadata_lookup()
"""

# Import main classes
from intccms.metadata_extractor.manager import DatasetMetadataManager
from intccms.metadata_extractor.builders import FilesetBuilder
from intccms.metadata_extractor.extractor import CoffeaMetadataExtractor
from intccms.metadata_extractor.core import parse_dataset_key, format_dataset_key

__all__ = [
    # Main classes (primary API)
    "DatasetMetadataManager",
    "FilesetBuilder",
    "CoffeaMetadataExtractor",

    # Core functions (for advanced users)
    "parse_dataset_key",
    "format_dataset_key",
]

# Version info
__version__ = "2.0.0"  # Major refactoring to modular structure
