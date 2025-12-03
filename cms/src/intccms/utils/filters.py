"""Utilities for filtering datasets, workitems, and filesets by process."""

import logging
from typing import Any, Dict, List, Optional

from coffea.processor.executor import WorkItem

logger = logging.getLogger(__name__)


def filter_by_process(
    items: Any,
    processes: List[str],
    metadata_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Any:
    """Filter datasets, workitems, or filesets by process names.

    This function provides a unified interface for filtering different data structures
    based on process names, reducing code duplication across the codebase.

    Args:
        items: Items to filter. Can be:
            - List[WorkItem]: Coffea workitems with usermeta
            - Dict[str, dict]: Fileset dictionary (dataset -> info)
            - List[str]: Process names
        processes: Process names to keep (e.g., ['ttbar', 'wjets'])
        metadata_lookup: Optional metadata dict for fileset filtering.
            Maps dataset -> {'process': str, ...}

    Returns:
        Filtered items of the same type as input

    Examples:
        >>> # Filter workitems
        >>> filtered = filter_by_process(workitems, ['ttbar', 'wjets'])

        >>> # Filter fileset
        >>> filtered = filter_by_process(fileset, ['ttbar'], metadata_lookup)

        >>> # Filter process list
        >>> filtered = filter_by_process(['ttbar', 'wjets', 'zjets'], ['ttbar'])
    """
    if not processes:
        return items

    # Filter list of WorkItems
    if isinstance(items, list) and items and isinstance(items[0], WorkItem):
        filtered = [wi for wi in items if wi.usermeta.get('process') in processes]
        logger.info(f"Filtered {len(items)} workitems → {len(filtered)} (processes: {processes})")
        return filtered

    # Filter fileset dictionary
    elif isinstance(items, dict):
        if metadata_lookup is None:
            raise ValueError("metadata_lookup required for filtering filesets")
        filtered = {
            dataset: info for dataset, info in items.items()
            if metadata_lookup.get(dataset, {}).get('process') in processes
        }
        logger.info(f"Filtered {len(items)} datasets → {len(filtered)} (processes: {processes})")
        return filtered

    # Filter list of process names
    elif isinstance(items, list) and items and isinstance(items[0], str):
        filtered = [p for p in items if p in processes]
        logger.info(f"Filtered {len(items)} processes → {len(filtered)} (keeping: {processes})")
        return filtered

    else:
        raise TypeError(f"Unsupported type for filtering: {type(items)}")


def should_process(process_name: str, processes_filter: Optional[List[str]]) -> bool:
    """Check if a process should be processed based on filter.

    Args:
        process_name: Name of the process to check
        processes_filter: Optional list of process names to include.
            If None, all processes are included.

    Returns:
        bool: True if process should be processed, False otherwise

    Examples:
        >>> should_process('ttbar', ['ttbar', 'wjets'])
        True
        >>> should_process('zjets', ['ttbar', 'wjets'])
        False
        >>> should_process('ttbar', None)
        True
    """
    if processes_filter is None:
        return True
    return process_name in processes_filter
