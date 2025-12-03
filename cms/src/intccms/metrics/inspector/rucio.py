"""Rucio-backed helpers for inspector size metrics.

This module lets users pull authoritative file-size information from Rucio and
optionally feed those numbers into the inspector statistics/plotting pipeline.
The goal is to keep the heavy Rucio lookups opt-in while providing a convenient
API that mirrors the rest of the inspector tooling.

Typical usage
-------------
>>> from intccms.datasets import DatasetManager
>>> from intccms.metrics.inspector import rucio as inspector_rucio
>>> dm = DatasetManager(config.datasets)
>>> size_summary = inspector_rucio.fetch_file_sizes(dm, processes=["signal"])
>>> console.print(inspector_rucio.format_dataset_size_table(size_summary))
>>> stats = aggregate_statistics(results, size_summary=size_summary)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from intccms.datasets import DatasetManager
from intccms.utils.filters import should_process

try:  # coffea is an optional dependency in some environments
    from coffea.dataset_tools import rucio_utils  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    rucio_utils = None  # type: ignore

logger = logging.getLogger(__name__)


def _read_listing_files(directory: Path, identifiers: Optional[Sequence[int]]) -> List[str]:
    """Return LFNs from listing files inside *directory*."""
    if identifiers is None:
        listing_files = sorted(directory.glob("*.txt"))
    else:
        listing_files = [directory / f"{idx}.txt" for idx in identifiers]

    lfns: List[str] = []
    for listing in listing_files:
        if not listing.is_file():
            logger.debug("Skipping missing listing file: %s", listing)
            continue
        for line in listing.read_text().splitlines():
            path = line.strip()
            if path:
                lfns.append(path)
    return lfns


def _build_filepath(lfn: str, redirector: Optional[str]) -> str:
    """Combine redirector prefix (if any) with logical file name."""
    if redirector:
        return f"{redirector}{lfn}"
    return lfn


def _strip_redirector(filepath: str) -> str:
    """Remove XRootD redirector from a path to recover the logical file name."""
    if filepath.startswith("root://"):
        stripped = filepath[len("root://") :]
        if "/" not in stripped:
            return filepath
        _, rest = stripped.split("/", 1)
        return "/" + rest.lstrip("/")
    return filepath


def _format_bytes(num_bytes: float) -> str:
    """Human-readable byte formatting."""
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    magnitude = 0
    while value >= 1024 and magnitude < len(units) - 1:
        value /= 1024.0
        magnitude += 1
    return f"{value:.2f} {units[magnitude]}"


def fetch_file_sizes(
    dataset_manager: DatasetManager,
    *,
    processes: Optional[Iterable[str]] = None,
    identifiers: Optional[Iterable[int]] = None,
    max_files_per_process: Optional[int] = None,
    scope: str = "cms",
    rucio_client=None,
) -> Dict:
    """Collect per-file byte information for datasets managed by DatasetManager.

    Parameters
    ----------
    dataset_manager
        Dataset manager created from the analysis configuration.
    processes
        Optional iterable of process names to restrict the lookup; if None,
        all configured processes are queried.
    identifiers
        Optional iterable of listing file indices (e.g. [0, 1]) to restrict
        the lookup to specific `N.txt` files inside each dataset directory.
    max_files_per_process
        Optional cap on the number of files per process to query (useful for quick
        spot checks).
    scope
        Rucio scope to use for file queries (defaults to ``cms``).
    rucio_client
        Pre-initialized Rucio client. If omitted, ``coffea.dataset_tools.rucio_utils``
        is used to create one lazily.

    Returns
    -------
    dict
        Summary structure with the following keys:
        - ``files``: list of ``{"dataset", "lfn", "filepath", "bytes"}`` entries
        - ``datasets``: mapping of dataset -> aggregated info (total bytes, files,…)
        - ``total_bytes`` / ``total_files``: overall aggregates
        - ``scope``: Rucio scope used for lookups
        - ``errors``: list of ``{"lfn", "error"}`` entries for failed lookups

    Notes
    -----
    The helper queries Rucio replicas for each logical file name. For large campaigns
    you may want to cache the returned summary to avoid repeated lookups.
    """
    if rucio_client is None:
        if rucio_utils is None:
            raise RuntimeError(
                "coffea.dataset_tools.rucio_utils is required to create a Rucio client"
            )
        rucio_client = rucio_utils.get_rucio_client()

    process_filter = set(processes) if processes is not None else None
    summary = {
        "files": [],
        "datasets": {},
        "total_bytes": 0,
        "total_files": 0,
        "scope": scope,
        "errors": [],
    }

    for process in dataset_manager.list_processes():
        if not should_process(process, process_filter):
            continue

        try:
            directories = dataset_manager.get_dataset_directories(process)
            redirector = dataset_manager.get_redirector(process)
        except KeyError as exc:
            logger.warning("Skipping process '%s': %s", process, exc)
            continue

        lfns: List[str] = []
        for directory in directories:
            lfns.extend(_read_listing_files(Path(directory), identifiers))

        if not lfns:
            continue

        if max_files_per_process is not None:
            lfns = lfns[:max_files_per_process]

        # Group LFNs by dataset name derived from the LFN structure
        dataset_groups: Dict[str, Dict[str, any]] = {}
        for lfn in lfns:
            parts = lfn.strip("/").split("/")
            if len(parts) < 6:
                summary["errors"].append({"dataset": process, "lfn": lfn, "error": "unable_to_parse_dataset"})
                continue
            prefix = "/" + "/".join(parts[:3])  # /store/mc/Run...
            dataset_name = "/" + "/".join(parts[3:6])  # /ZPrime.../NANOAODSIM/...
            dataset_groups.setdefault(dataset_name, {"prefix": prefix, "lfns": set()})["lfns"].add(lfn)

        size_map: Dict[str, int] = {}
        for dataset_name, info in dataset_groups.items():
            try:
                files_iter = rucio_client.list_files(scope=scope, name=dataset_name)
            except Exception as error:
                logger.warning("Failed to list files for dataset %s: %s", dataset_name, error)
                for lfn in info["lfns"]:
                    summary["errors"].append({"dataset": process, "lfn": lfn, "error": str(error)})
                continue

            for entry in files_iter:
                name = entry.get("name")
                if not name:
                    continue
                full_lfn = f"{info['prefix']}/{dataset_name.strip('/')}/{name}"
                if full_lfn in info["lfns"]:
                    try:
                        size_map[full_lfn] = int(entry.get("bytes", 0))
                    except Exception:
                        size_map[full_lfn] = 0

        dataset_files = []
        for lfn in lfns:
            filepath = _build_filepath(lfn, redirector)
            size_bytes = size_map.get(lfn, 0)
            if not size_bytes:
                summary["errors"].append({"dataset": process, "lfn": lfn, "error": "size_not_found"})

            entry = {
                "dataset": process,
                "lfn": lfn,
                "filepath": filepath,
                "bytes": size_bytes,
            }
            dataset_files.append(entry)
            summary["files"].append(entry)

        total_bytes_dataset = sum(item["bytes"] for item in dataset_files)
        dataset_info = {
            "dataset": process,
            "files": dataset_files,
            "num_files": len(dataset_files),
            "total_bytes": total_bytes_dataset,
            "avg_bytes_per_file": (
                total_bytes_dataset / len(dataset_files) if dataset_files else 0
            ),
        }
        summary["datasets"][process] = dataset_info
        summary["total_bytes"] += total_bytes_dataset
        summary["total_files"] += len(dataset_files)

    return summary


def format_dataset_size_table(size_summary: Dict):
    """Return a Rich table summarising total bytes per dataset."""
    from rich.table import Table  # Local import to avoid Rich dependency during tests

    table = Table(title="Dataset File Sizes", show_header=True, header_style="bold cyan")
    table.add_column("Dataset", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Total Size", justify="right", style="magenta")
    table.add_column("Avg Size/File", justify="right", style="green")

    for dataset, info in sorted(size_summary.get("datasets", {}).items()):
        num_files = info.get("num_files", 0)
        total_bytes = info.get("total_bytes", 0)
        avg_size = info.get("avg_bytes_per_file", 0)
        table.add_row(
            dataset,
            f"{num_files:,}",
            _format_bytes(total_bytes),
            _format_bytes(avg_size),
        )

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"{size_summary.get('total_files', 0):,}",
        _format_bytes(size_summary.get("total_bytes", 0)),
        "",
    )
    return table


def build_size_lookup(size_summary: Dict) -> Dict[str, int]:
    """Construct mapping from file path / LFN to byte counts."""
    lookup: Dict[str, int] = {}
    for entry in size_summary.get("files", []):
        filepath = entry.get("filepath")
        if filepath:
            lookup[filepath] = entry.get("bytes", 0)
        lfn = entry.get("lfn")
        if lfn:
            lookup[lfn] = entry.get("bytes", 0)
        if filepath:
            lookup[_strip_redirector(filepath)] = entry.get("bytes", 0)
    return lookup
