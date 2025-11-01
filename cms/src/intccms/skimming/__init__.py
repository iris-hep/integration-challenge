"""Skimming utilities for event filtering and I/O operations.

This module provides tools for skimming NanoAOD events, including:
- I/O operations for reading and writing events in various formats
- Pipeline stages for processing events
- Workflow management for parallel processing

The refactored architecture emphasizes modularity, testability, and extensibility.
"""

# Version info
__version__ = "2.0.0"  # Refactored version

# Main exports
from intccms.skimming.manager import SkimmingManager
from intccms.skimming.fileset_manager import FilesetManager

__all__ = ["SkimmingManager", "FilesetManager"]
