"""Integration with DatasetManager - extract files for inspection.

This module bridges DatasetManager (which has dataset configurations and file listings)
to the inspector module. It reads .txt file lists and prepares them for inspection.

IMPORTANT: This does NOT require metadata preprocessing - it works directly with
the raw dataset configuration.
"""

from typing import Dict, List, Tuple
from pathlib import Path

from intccms.datasets import DatasetManager


def extract_files_from_dataset_manager(
    dataset_manager: DatasetManager,
    processes: List[str] = None,
    max_files_per_process: int = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Extract file paths from DatasetManager for inspection.

    Reads .txt files from dataset directories to get actual ROOT file paths.

    Parameters
    ----------
    dataset_manager : DatasetManager
        Dataset manager with configuration
    processes : List[str], optional
        List of process names to inspect (e.g., ["signal", "ttbar_semilep"])
        If None, inspects all processes
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
    """
    # Determine which processes to inspect
    if processes is None:
        processes = list(dataset_manager.datasets.keys())

    file_list = []
    dataset_map = {}

    for process_name in processes:
        # Get directories for this process
        try:
            directories = dataset_manager.get_dataset_directories(process_name)
            file_pattern = dataset_manager.get_file_pattern(process_name)
        except KeyError:
            # Process not found, skip
            continue

        # Read .txt files from each directory
        process_files = []
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue

            # Find .txt files matching pattern
            txt_files = list(dir_path.glob(file_pattern))

            # Read each .txt file to get ROOT file paths
            for txt_file in txt_files:
                with open(txt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and line.endswith('.root'):
                            process_files.append(line)

        # Apply max_files limit if specified
        if max_files_per_process is not None:
            process_files = process_files[:max_files_per_process]

        # Add to file_list and dataset_map
        for filepath in process_files:
            file_list.append(filepath)
            dataset_map[filepath] = process_name

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
