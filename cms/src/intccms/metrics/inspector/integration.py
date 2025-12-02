"""Integration with DatasetManager - extract files for inspection.

This module bridges DatasetManager (which has dataset configurations and file listings)
to the inspector module. It uses existing infrastructure from metadata_extractor.

IMPORTANT: This does NOT require metadata preprocessing - it works directly with
the raw dataset configuration.
"""

from typing import Dict, List, Optional, Tuple

from intccms.datasets import DatasetManager
from intccms.metadata_extractor.io import collect_file_paths
from intccms.utils.filters import should_process


def extract_files_from_dataset_manager(
    dataset_manager: DatasetManager,
    processes: Optional[List[str]] = None,
    identifiers: Optional[List[int]] = None,
    max_files_per_process: Optional[int] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Extract file paths from DatasetManager for inspection.

    Reads .txt files from dataset directories to get actual ROOT file paths.
    Uses the same infrastructure as metadata_extractor.

    Parameters
    ----------
    dataset_manager : DatasetManager
        Dataset manager with configuration
    processes : List[str], optional
        List of process names to inspect (e.g., ["signal", "ttbar_semilep"])
        If None, inspects all processes
    identifiers : List[int], optional
        Specific listing file IDs to process (e.g., [0, 1] reads 0.txt, 1.txt)
        If None, reads all .txt files
    max_files_per_process : int, optional
        Limit files per process (useful for quick sampling)

    Returns
    -------
    file_list : List[str]
        List of ROOT file paths (XRootD URLs or local paths)
    dataset_map : Dict[str, str]
        Mapping from filepath to dataset/process name

    Examples
    --------
    >>> from intccms.datasets import DatasetManager
    >>> dm = DatasetManager(config)
    >>>
    >>> # Get all files
    >>> files, dataset_map = extract_files_from_dataset_manager(dm)
    >>>
    >>> # Or just specific processes
    >>> files, dataset_map = extract_files_from_dataset_manager(
    ...     dm, processes=["signal", "ttbar_semilep"]
    ... )
    >>>
    >>> # Or sample first 10 files per process
    >>> files, dataset_map = extract_files_from_dataset_manager(
    ...     dm, max_files_per_process=10
    ... )
    >>>
    >>> # Or specific listing files
    >>> files, dataset_map = extract_files_from_dataset_manager(
    ...     dm, identifiers=[0, 1]  # Only reads 0.txt and 1.txt
    ... )
    """
    file_list = []
    dataset_map = {}

    # Iterate over each configured process
    for process_name in dataset_manager.list_processes():
        # Check if processes filter is configured
        if not should_process(process_name, processes):
            continue

        try:
            # Get configuration for this process
            directories = dataset_manager.get_dataset_directories(process_name)
            redirector = dataset_manager.get_redirector(process_name)

            # Collect files from all directories for this process
            process_files = []
            for directory in directories:
                # Use existing collect_file_paths infrastructure
                files = collect_file_paths(directory, identifiers, redirector)
                process_files.extend(files)

            # Apply max_files limit if specified
            if max_files_per_process is not None:
                process_files = process_files[:max_files_per_process]

            # Add to file_list and dataset_map
            for filepath in process_files:
                file_list.append(filepath)
                dataset_map[filepath] = process_name

        except (KeyError, FileNotFoundError):
            # Process not found or no files found, skip
            continue

    return file_list, dataset_map


def get_dataset_file_counts(dataset_manager: DatasetManager) -> Dict[str, int]:
    """Get file count per dataset without full inspection.

    Quick summary of how many files per process.

    Parameters
    ----------
    dataset_manager : DatasetManager
        Dataset manager with configuration

    Returns
    -------
    file_counts : Dict[str, int]
        Mapping from process name to number of files

    Examples
    --------
    >>> counts = get_dataset_file_counts(dm)
    >>> for dataset, count in counts.items():
    ...     print(f"{dataset}: {count} files")
    """
    file_list, dataset_map = extract_files_from_dataset_manager(dataset_manager)

    # Count files per dataset
    from collections import defaultdict
    counts = defaultdict(int)

    for filepath, dataset in dataset_map.items():
        counts[dataset] += 1

    return dict(counts)
