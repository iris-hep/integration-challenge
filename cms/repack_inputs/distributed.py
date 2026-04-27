"""Distributed Dask + XRootD layer for ROOT file repacking.

Driver-side (this module on the local Python process):

- :class:`FileRecord` / :class:`ChunkSegment` / :class:`ChunkPlan` dataclasses
- :func:`load_fileset` reads the input JSON
- :func:`plan_chunks` builds chunks from the fileset using ``nevts`` from JSON
- :func:`prepare_output_dirs` ``mkdir -p``s the XRootD output directories
- :func:`upload_package` zips this package and ships it to scheduler + workers
- :func:`run_repack` submits one task per chunk, gathers with tqdm progress
- :func:`write_output_fileset` mirrors the input JSON shape with output URLs

Worker-side (executed on Dask workers):

- :func:`repack_chunk` ``xrdcp``s the chunk's input files into local scratch,
  slices / re-baskets via :mod:`root_repack`, merges into one local file, and
  ``xrdcp``s the merged file to the final XRootD URL.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from dask.distributed import as_completed
from tqdm.auto import tqdm

from repack_inputs import root_repack as _rr

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileRecord:
    """One input file with its pre-computed entry count."""

    dataset: str
    systematic: str
    url: str
    nevts: int


@dataclass(frozen=True)
class ChunkSegment:
    """One slice of one remote input file."""

    source_url: str
    start: int
    count: int
    total_entries: int


@dataclass(frozen=True)
class ChunkPlan:
    """One output file plus the segments that feed it."""

    dataset: str
    systematic: str
    index: int
    output_url: str
    segments: tuple[ChunkSegment, ...]

    @property
    def total_events(self) -> int:
        return sum(s.count for s in self.segments)

    @property
    def unique_sources(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for seg in self.segments:
            seen.setdefault(seg.source_url, None)
        return tuple(seen)


# ---------------------------------------------------------------------------
# Fileset JSON
# ---------------------------------------------------------------------------


def load_fileset(json_path: str | Path) -> list[FileRecord]:
    """Flatten the fileset JSON into one record per input file."""

    data = json.loads(Path(json_path).expanduser().read_text())
    records: list[FileRecord] = []
    for dataset, systs in data.items():
        for syst, body in systs.items():
            for entry in body["files"]:
                records.append(
                    FileRecord(
                        dataset=dataset,
                        systematic=syst,
                        url=entry["path"],
                        nevts=int(entry["nevts"]),
                    )
                )
    return records


def plan_chunks(
    files: Sequence[FileRecord],
    *,
    output_dir_url: str,
    n_events: int | None = None,
    output_subdir: str = "{dataset}/{systematic}",
) -> list[ChunkPlan]:
    """Produce one :class:`ChunkPlan` per output file.

    Files are grouped by ``(dataset, systematic)``. Within each group the
    inputs are scanned in their JSON order and split into chunks of up to
    ``n_events`` entries each.
    """

    if n_events is not None and n_events <= 0:
        raise ValueError("n_events must be greater than zero")

    grouped: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    for record in files:
        grouped[(record.dataset, record.systematic)].append(record)

    base = output_dir_url.rstrip("/")
    plans: list[ChunkPlan] = []
    for (dataset, syst), group in grouped.items():
        chunks = _chunk_by_events(group, n_events)
        subdir = output_subdir.format(dataset=dataset, systematic=syst)
        for idx, segments in enumerate(chunks, start=1):
            if not segments:
                continue
            plans.append(
                ChunkPlan(
                    dataset=dataset,
                    systematic=syst,
                    index=idx,
                    output_url=f"{base}/{subdir}/{idx:03d}.root",
                    segments=tuple(segments),
                )
            )
    return plans


def _chunk_by_events(
    files: Sequence[FileRecord],
    n_events: int | None,
) -> list[list[ChunkSegment]]:
    if n_events is None:
        segments = [
            ChunkSegment(f.url, 0, f.nevts, f.nevts) for f in files if f.nevts > 0
        ]
        return [segments] if segments else []

    chunks: list[list[ChunkSegment]] = []
    current: list[ChunkSegment] = []
    current_count = 0
    for f in files:
        if f.nevts <= 0:
            continue
        start = 0
        while start < f.nevts:
            capacity = n_events - current_count
            take = min(capacity, f.nevts - start)
            current.append(ChunkSegment(f.url, start, take, f.nevts))
            current_count += take
            start += take
            if current_count == n_events:
                chunks.append(current)
                current = []
                current_count = 0
    if current:
        chunks.append(current)
    return chunks


def write_output_fileset(
    plans: Sequence[ChunkPlan],
    results: Mapping[str, str | BaseException],
    out_json_path: str | Path,
) -> Path:
    """Write a JSON listing only the successfully-written outputs.

    The structure mirrors the input fileset:
    ``{dataset: {systematic: {"files": [{path, nevts}], "nevts_total"}}}``.
    """

    nested: dict[str, dict[str, dict[str, Any]]] = {}
    for plan in plans:
        outcome = results.get(plan.output_url)
        if not isinstance(outcome, str):
            continue
        entry = nested.setdefault(plan.dataset, {}).setdefault(
            plan.systematic, {"files": [], "nevts_total": 0}
        )
        entry["files"].append({"path": outcome, "nevts": plan.total_events})
        entry["nevts_total"] += plan.total_events

    out_path = Path(out_json_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nested, indent=2) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# XRootD helpers (subprocess; needs xrdcp + xrdfs in PATH)
# ---------------------------------------------------------------------------


def _split_xrootd_url(url: str) -> tuple[str, str]:
    """Split ``root://host[:port]//path`` into ``("root://host", "/path")``."""

    parsed = urlparse(url)
    if parsed.scheme != "root":
        raise ValueError(f"not an xrootd URL: {url}")
    host = parsed.netloc
    if not host:
        raise ValueError(f"xrootd URL missing host: {url}")
    path = "/" + parsed.path.lstrip("/")
    return f"root://{host}", path


def _xrdcp(src: str, dst: str, *, force: bool = False, timeout: float | None = None) -> None:
    cmd = ["xrdcp", "--silent"]
    if force:
        cmd.append("-f")
    cmd.extend([src, dst])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"xrdcp failed ({src} -> {dst}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _xrdfs(host: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["xrdfs", host, *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"xrdfs {' '.join(args)} failed on {host}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def _xrdfs_mkdir_p(url: str) -> None:
    host, path = _split_xrootd_url(url)
    _xrdfs(host, "mkdir", "-p", path)


def prepare_output_dirs(plans: Sequence[ChunkPlan]) -> None:
    """Driver-side: ``mkdir -p`` each distinct output parent on XRootD."""

    seen: set[str] = set()
    for plan in plans:
        parent = plan.output_url.rsplit("/", 1)[0]
        if parent in seen:
            continue
        seen.add(parent)
        _xrdfs_mkdir_p(parent)


# ---------------------------------------------------------------------------
# Scratch helpers
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                pass
    return total


def _enforce_budget(path: Path, budget_bytes: int, *, phase: str) -> None:
    used = _dir_size_bytes(path)
    if used > budget_bytes:
        raise RuntimeError(
            f"scratch budget exceeded during {phase}: "
            f"{used / 1024**3:.2f} GB used, cap is {budget_bytes / 1024**3:.2f} GB"
        )


def _unique_local_path(inputs_dir: Path, url: str) -> Path:
    base = Path(url).name or "input.root"
    candidate = inputs_dir / base
    if not candidate.exists():
        return candidate
    for i in range(1, 10_000):
        candidate = inputs_dir / f"{i:04d}_{base}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find free local name for {url}")


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------


def repack_chunk(
    chunk: ChunkPlan,
    *,
    repack_kwargs: Mapping[str, Any],
    scratch_root: str,
    max_scratch_gb: float,
    overwrite: bool = False,
) -> str:
    """``xrdcp`` inputs to local scratch, slice / merge, ``xrdcp`` out.

    The full input file is downloaded for every unique source URL in the
    chunk, even when only a slice is needed. Slicing / re-basketing uses
    :mod:`root_repack` internals against the local copies, then
    :func:`root_repack._merge_root_files` produces one merged local file.
    The merged file is ``xrdcp``-ed straight to ``chunk.output_url`` with
    ``-f`` when ``overwrite`` is set. We do not write to a ``.tmp`` and
    rename, because most CMS SEs disable user-level ``mv`` and ``rm``.

    Returns the final XRootD URL on success.
    """

    if not chunk.segments:
        raise ValueError(f"empty chunk: {chunk.output_url}")

    scratch_root_path = Path(scratch_root)
    scratch_root_path.mkdir(parents=True, exist_ok=True)
    budget_bytes = int(max_scratch_gb * 1024**3)

    event_tree = repack_kwargs.get("event_tree", "Events")
    merge_opts = _rr.MergeOptions(
        fast=bool(repack_kwargs.get("fast", False)),
        keep=bool(repack_kwargs.get("keep", False)),
        sort=str(repack_kwargs.get("sort", "branch")),
        compress=str(repack_kwargs.get("compress", "same")),
        iofeatures=tuple(repack_kwargs.get("iofeatures") or ()),
        verbose=int(repack_kwargs.get("verbose", 0)),
    )
    tree_opts = _rr.TreeWriteOptions(
        basket_sizes=_rr._normalize_basket_sizes(
            repack_kwargs.get("basket_size"),
            repack_kwargs.get("basket_sizes"),
        ),
        auto_flush=_rr._parse_auto_flush(repack_kwargs.get("auto_flush")),
    )

    prefix = f"repack_{chunk.dataset}_{chunk.systematic}_{chunk.index:03d}_"
    with TemporaryDirectory(prefix=prefix, dir=str(scratch_root_path)) as tmpdir:
        tmp = Path(tmpdir)
        inputs_dir = tmp / "inputs"
        staging_dir = tmp / "staging"
        output_dir = tmp / "output"
        inputs_dir.mkdir()
        staging_dir.mkdir()
        output_dir.mkdir()

        url_to_local: dict[str, Path] = {}
        for url in chunk.unique_sources:
            local = _unique_local_path(inputs_dir, url)
            LOGGER.info("xrdcp %s -> %s", url, local)
            _xrdcp(url, str(local))
            url_to_local[url] = local
            _enforce_budget(tmp, budget_bytes, phase="input download")

        local_segments = [
            _rr.EventSegment(
                url_to_local[seg.source_url],
                seg.start,
                seg.count,
                seg.total_entries,
            )
            for seg in chunk.segments
        ]

        staged_paths = _rr._stage_segment_files(
            local_segments, event_tree, staging_dir, tree_opts
        )
        _enforce_budget(tmp, budget_bytes, phase="segment staging")

        local_output = output_dir / Path(chunk.output_url).name
        _rr._merge_root_files(local_output, staged_paths, merge_opts)
        _enforce_budget(tmp, budget_bytes, phase="local merge")

        _xrdcp(str(local_output), chunk.output_url, force=overwrite)

    return chunk.output_url


# ---------------------------------------------------------------------------
# Driver: shipping + runner
# ---------------------------------------------------------------------------


def upload_package(client) -> Path:
    """Zip this package and upload to scheduler + workers via Dask.

    Call once per :class:`~dask.distributed.Client` lifetime, before any
    :meth:`Client.submit`. Puts ``repack_inputs`` on the scheduler's and
    each worker's ``sys.path`` so deserialization of the task graph and
    execution of :func:`repack_chunk` both work without the package being
    pre-installed on the cluster image.
    """

    pkg_dir = Path(__file__).resolve().parent
    zip_path = shutil.make_archive(
        "/tmp/repack_inputs_pkg",
        "zip",
        root_dir=str(pkg_dir.parent),
        base_dir=pkg_dir.name,
    )
    client.upload_file(zip_path)
    return Path(zip_path)


def run_repack(
    client,
    plans: Sequence[ChunkPlan],
    *,
    scratch_root: str = "/tmp/repack_inputs",
    max_scratch_gb: float = 7.0,
    overwrite: bool = False,
    event_tree: str = "Events",
    basket_size: int | str | None = None,
    basket_sizes: Any = None,
    auto_flush: int | str | None = None,
    fast: bool = False,
    keep: bool = False,
    sort: str = "branch",
    compress: str = "same",
    iofeatures: Sequence[str | int] | None = None,
    verbose: int = 0,
    progress: bool = True,
    raise_on_error: bool = True,
) -> dict[str, str | BaseException]:
    """Submit one task per :class:`ChunkPlan`; gather results as they finish.

    Returns a dict keyed by each plan's output URL. Successful values are
    the written URL. Failed values are the raised exception when
    ``raise_on_error=False``; otherwise the first failure re-raises after
    in-flight work is cancelled.
    """

    repack_kwargs = {
        "event_tree": event_tree,
        "basket_size": basket_size,
        "basket_sizes": basket_sizes,
        "auto_flush": auto_flush,
        "fast": fast,
        "keep": keep,
        "sort": sort,
        "compress": compress,
        "iofeatures": list(iofeatures) if iofeatures else None,
        "verbose": verbose,
    }

    plan_by_key: dict[str, ChunkPlan] = {}
    futures = []
    for plan in plans:
        key = f"repack-{plan.dataset}-{plan.systematic}-{plan.index:03d}"
        plan_by_key[key] = plan
        futures.append(
            client.submit(
                repack_chunk,
                plan,
                repack_kwargs=repack_kwargs,
                scratch_root=scratch_root,
                max_scratch_gb=max_scratch_gb,
                overwrite=overwrite,
                key=key,
                pure=False,
            )
        )

    results: dict[str, str | BaseException] = {}
    iterator = as_completed(futures)
    if progress:
        iterator = tqdm(iterator, total=len(futures), desc="Repacking", unit="chunk")
    for future in iterator:
        plan = plan_by_key[future.key]
        try:
            url = future.result()
            results[plan.output_url] = url
        except BaseException as err:  # noqa: BLE001
            if raise_on_error:
                for other in futures:
                    if not other.done():
                        other.cancel()
                raise
            results[plan.output_url] = err
    return results
