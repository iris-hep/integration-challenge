"""Distributed ROOT file repacking over Dask + XRootD.

Thin driver/worker layer on top of the vendored :mod:`root_repack`:

- Planner: takes a JSON fileset with pre-computed ``nevts`` per file and
  produces one :class:`ChunkPlan` per output file. Runs on the driver
  without opening remote files.
- Worker: :func:`repack_chunk` streams source files straight from XRootD
  through ROOT's native client. Whole-file pass-through segments are
  fed to ``TFileMerger`` by URL; sliced or re-basketed segments go to
  local scratch first. The merged output is written to local scratch
  (XRootD does not support the random-access writes ``TFileMerger``
  needs to finalise a file), then ``xrdcp``'d straight to the final
  XRootD URL. We do not write to a ``.tmp`` and rename, because most
  CMS SEs disable user-level ``mv`` and ``rm``. Local scratch is
  bounded by ``max_scratch_gb`` and cleaned up at task exit.
- Runner: :func:`run_repack` submits one task per chunk through a
  :class:`dask.distributed.Client`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import cloudpickle
from dask.distributed import as_completed
from tqdm.auto import tqdm

from intccms.utils.repack import root_repack as _rr

cloudpickle.register_pickle_by_value(sys.modules[__name__])

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
# Fileset loading + planning
# ---------------------------------------------------------------------------


def load_fileset(json_path: str | os.PathLike[str]) -> list[FileRecord]:
    """Flatten the nanoaods-style JSON into one record per input file."""

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
    inputs are scanned in the order they appear in ``files`` and split
    into chunks of up to ``n_events`` entries (using the same algorithm
    as :func:`root_repack._event_chunks` but reading counts from the
    JSON instead of opening files).
    """

    if n_events is not None and n_events <= 0:
        raise ValueError("n_events must be greater than zero")

    grouped: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    for record in files:
        grouped[(record.dataset, record.systematic)].append(record)

    base = output_dir_url.rstrip("/")
    plans: list[ChunkPlan] = []
    for (dataset, syst), group in grouped.items():
        segments_per_chunk = _chunk_by_events(group, n_events)
        subdir = output_subdir.format(dataset=dataset, systematic=syst)
        for idx, segments in enumerate(segments_per_chunk, start=1):
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
    """Pure-bookkeeping port of ``root_repack._event_chunks`` using JSON counts."""

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


# ---------------------------------------------------------------------------
# XRootD helpers (subprocess-based; relies on xrdcp / xrdfs on worker PATH)
# ---------------------------------------------------------------------------


def _split_xrootd_url(url: str) -> tuple[str, str]:
    """Split ``root://host[:port]//path`` into ``("root://host", "/path")``.

    XRootD uses a double-slash to separate host from path, so ``urlparse``
    alone is not enough — the path component of ``root://host//store/x``
    ends up as ``/store/x`` which we keep, but we must also handle the
    bare ``root://host/store/x`` form.
    """

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
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"xrdcp failed ({src} -> {dst}): {result.stderr.strip() or result.stdout.strip()}"
        )


def _xrdfs(host: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["xrdfs", host, *args], capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"xrdfs {' '.join(args)} failed on {host}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def _xrdfs_mkdir_p(url: str) -> None:
    host, path = _split_xrootd_url(url)
    _xrdfs(host, "mkdir", "-p", path)


def _xrdfs_mv(src_url: str, dst_url: str) -> None:
    src_host, src_path = _split_xrootd_url(src_url)
    dst_host, dst_path = _split_xrootd_url(dst_url)
    if src_host != dst_host:
        raise ValueError(
            f"xrdfs mv requires same host, got {src_host} and {dst_host}"
        )
    _xrdfs(src_host, "mv", src_path, dst_path)


def _xrdfs_rm(url: str, *, missing_ok: bool = True) -> None:
    host, path = _split_xrootd_url(url)
    result = _xrdfs(host, "rm", path, check=False)
    if result.returncode != 0 and not missing_ok:
        raise RuntimeError(
            f"xrdfs rm failed ({url}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def prepare_output_dirs(plans: Sequence[ChunkPlan]) -> None:
    """Driver-side: mkdir -p each distinct output parent on XRootD once."""

    seen: set[str] = set()
    for plan in plans:
        parent = plan.output_url.rsplit("/", 1)[0]
        if parent in seen:
            continue
        seen.add(parent)
        _xrdfs_mkdir_p(parent)


# ---------------------------------------------------------------------------
# Scratch accounting
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
    """Stream inputs from XRootD; merge to local scratch; xrdcp out.

    No ``xrdcp`` step on the input side: ``TFile``/``TFileMerger`` read
    the source URLs over the network. Whole-file pass-through segments
    feed their URL straight to ``TFileMerger.AddFile``; sliced or
    re-basketed segments write a local temp file under ``scratch_root``.
    The merged output is written to local scratch (XRootD does not
    support the random-access writes ``TFileMerger`` needs to finalise
    a file), then ``xrdcp``'d directly to ``chunk.output_url`` with
    ``-f`` if ``overwrite`` is set. We do not write to a ``.tmp`` and
    rename, because most CMS SEs disable user-level ``mv`` and ``rm``;
    a partial ``xrdcp`` from a crashed worker leaves a partial file
    that the next run will overwrite.

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
        staging_dir = tmp / "staging"
        output_dir = tmp / "output"
        staging_dir.mkdir()
        output_dir.mkdir()

        staged_inputs = _stage_streaming_segments(
            chunk.segments, event_tree, staging_dir, tree_opts
        )
        _enforce_budget(tmp, budget_bytes, phase="segment staging")

        local_output = output_dir / Path(chunk.output_url).name
        _rr._merge_root_files(local_output, staged_inputs, merge_opts)
        _enforce_budget(tmp, budget_bytes, phase="local merge")

        _xrdcp(str(local_output), chunk.output_url, force=overwrite)

    return chunk.output_url


def _stage_streaming_segments(
    segments: Sequence[ChunkSegment],
    event_tree: str,
    staging_dir: Path,
    tree_options: Any,
) -> list:
    """Produce one entry per segment for ``TFileMerger.AddFile``.

    Pass-through segments (whole file, no rewrite) yield their original
    XRootD URL string and are streamed by the merger directly. Sliced
    or re-basketed segments are written to ``staging_dir`` as local
    temp files; their local ``Path`` is yielded.
    """

    staged: list = []
    for index, seg in enumerate(segments):
        if (
            not tree_options.requires_rewrite
            and seg.start == 0
            and seg.count == seg.total_entries
        ):
            staged.append(seg.source_url)
            continue

        staged_path = staging_dir / f"segment_{index:03d}.root"
        LOGGER.info(
            "slicing %s [%d:%d] -> %s",
            seg.source_url,
            seg.start,
            seg.start + seg.count,
            staged_path,
        )
        _rr._copy_tree_range(
            seg.source_url,
            staged_path,
            event_tree,
            seg.start,
            seg.count,
            tree_options,
        )
        staged.append(staged_path)
    return staged


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_repack(
    client,
    plans: Sequence[ChunkPlan],
    *,
    scratch_root: str = "/tmp/intccms_repack",
    max_scratch_gb: float = 4.0,
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
    """Submit one task per plan and collect results as they finish.

    Returns a dict keyed by each plan's output URL. Successful values are
    the written URL (same as the key). Failed values are the raised
    exception when ``raise_on_error=False``; otherwise the first failure
    re-raises after in-flight work is cancelled.

    All keyword arguments other than ``client``, ``plans``, ``scratch_root``,
    ``max_scratch_gb``, ``progress`` and ``raise_on_error`` mirror the
    ``root_repack`` / CLI options.
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


def write_output_fileset(
    plans: Sequence[ChunkPlan],
    results: Mapping[str, str | BaseException],
    out_json_path: str | os.PathLike[str],
) -> Path:
    """Write an input-compatible JSON listing the outputs that succeeded.

    The structure mirrors the input ``nanoaods.json``:
    ``{dataset: {systematic: {"files": [{"path", "nevts"}], "nevts_total"}}}``.
    Only plans whose ``output_url`` appears as a string value in
    ``results`` are included. Failed chunks (values that are exceptions)
    are skipped.
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
