"""Caching for skimmed events.

This module handles only caching operations:
- Computing cache keys
- Loading from cache
- Saving to cache

Event loading and merging use existing infrastructure:
- stages.load_events() for loading files
- ak.concatenate() for merging arrays
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, List, Optional

import cloudpickle

logger = logging.getLogger(__name__)


def compute_cache_key(fileset_key: str, output_files: List[str]) -> str:
    """Compute deterministic cache key from fileset and file list.

    The cache key is computed as MD5("{fileset_key}::{file1}::{file2}::...").
    Files are sorted to ensure same key regardless of discovery order.
    If any output file changes or is regenerated, cache is invalidated.

    Args:
        fileset_key: Dataset/fileset identifier
        output_files: List of output file paths

    Returns:
        MD5 hash string (32 hex characters)
    """
    sorted_files = sorted(output_files)
    cache_input = f"{fileset_key}::{':'.join(sorted_files)}"
    return hashlib.md5(cache_input.encode()).hexdigest()


def load_cached_events(
    cache_file: Path,
) -> Optional[Any]:
    """Load events from cache file.

    Args:
        cache_file: Path to cache file

    Returns:
        Cached events if successful, None if loading fails
    """
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            events = cloudpickle.load(f)
        logger.info(f"Loaded {len(events)} events from cache")
        return events
    except Exception as e:
        logger.error(f"Failed to load cached events from {cache_file}: {e}")
        return None


def save_cached_events(
    events: Any,
    cache_file: Path,
) -> bool:
    """Save events to cache file.

    Args:
        events: Events to cache
        cache_file: Path to cache file

    Returns:
        True if successful, False otherwise
    """
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            cloudpickle.dump(events, f)
        logger.info(f"Cached {len(events)} events to {cache_file}")
        return True
    except Exception as e:
        logger.warning(f"Failed to cache events to {cache_file}: {e}")
        return False
