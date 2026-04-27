#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Vendored verbatim from https://github.com/pfackeldey/root_repack (commit: main)
# Do not edit in place; refresh from upstream.

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import time
from typing import Iterable, Sequence

import ROOT
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only without optional dependency
    tqdm = None

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch()

LOGGER = logging.getLogger(__name__)

COMPRESSION = {
    lut.__name__[1:].lower(): {
        k[len(pre) :]: getattr(lut, k)
        for k in dir(lut)
        if k.startswith(pre) and not k.endswith("Undefined")
    }
    for lut, pre in [
        (ROOT.RCompressionSetting.EAlgorithm, "k"),
        (ROOT.RCompressionSetting.EDefaults, "kUse"),
    ]
}
IOFEATURES = {
    name[1:]: getattr(source, name)
    for source in [ROOT.Experimental.EIOFeatures, ROOT.EIOFeatures]
    for name in dir(source)
    if name.startswith("k") and name != "kSupported"
}
SORT_MODES = {"branch", "offset", "entry"}
BYTE_SUFFIXES = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
}


@dataclass(frozen=True)
class EventSegment:
    path: Path
    start: int
    count: int
    total_entries: int


@dataclass(frozen=True)
class MergeOptions:
    fast: bool = False
    keep: bool = False
    sort: str = "branch"
    compress: str = "same"
    iofeatures: tuple[str | int, ...] = ()
    verbose: int = 0


@dataclass(frozen=True)
class TreeWriteOptions:
    basket_sizes: tuple[tuple[str, int], ...] = ()
    auto_flush: int | None = None

    @property
    def requires_rewrite(self) -> bool:
        return bool(self.basket_sizes) or self.auto_flush is not None


@dataclass(frozen=True)
class RepackPlan:
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    chunks: tuple[tuple[EventSegment, ...], ...] | None = None
    inplace: bool = False


def _progress(
    iterable,
    *,
    enabled: bool,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    leave: bool = True,
):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
    )


def _as_paths(paths: Sequence[os.PathLike[str] | str]) -> list[Path]:
    if not paths:
        raise ValueError("at least one input file is required")
    return [Path(path).expanduser() for path in paths]


def _validate_input_files(paths: Sequence[Path]) -> list[Path]:
    validated = []
    for idx, path in enumerate(paths, start=1):
        if not path.exists():
            raise FileNotFoundError(f"input file #{idx} does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"input path #{idx} is not a file: {path}")
        validated.append(path.resolve())
    return validated


def _open_root_file(path: Path, mode: str = "READ"):
    root_file = ROOT.TFile.Open(str(path), mode)
    if not root_file or root_file.IsZombie():
        raise OSError(f"failed to open ROOT file in {mode!r} mode: {path}")
    return root_file


@contextmanager
def _root_file(path: Path, mode: str = "READ"):
    root_file = _open_root_file(path, mode)
    try:
        yield root_file
    finally:
        root_file.Close()


def _tree_entries(path: Path, tree_name: str | None) -> int:
    if tree_name is None:
        raise RuntimeError("tree entry lookup requires an event tree name")
    with _root_file(path) as root_file:
        tree = root_file.Get(tree_name)
        if not tree:
            raise ValueError(f"{path} does not contain tree {tree_name!r}")
        if not tree.InheritsFrom("TTree"):
            raise TypeError(f"{tree_name!r} in {path} is not a TTree")
        return int(tree.GetEntries())


def _compression_settings(compress: str, first_input: Path) -> int:
    if compress.lower() == "same":
        with _root_file(first_input) as root_file:
            return int(root_file.GetCompressionSettings())

    comp = compress.split("=", 1)
    if len(comp) == 1:
        comp.append("")
    algo, level = comp
    lookup = COMPRESSION["algorithm" if level else "defaults"]
    for key, value in lookup.items():
        if key.lower() == algo.lower():
            if level:
                return int(ROOT.CompressionSettings(value, int(level)))
            return int(value)

    allowed = ", ".join(sorted(lookup))
    raise ValueError(f"unknown compression setting {algo!r}; allowed values: {allowed}")


def _iofeatures(features: Iterable[str | int] | None):
    features = tuple(features or ())
    if not features:
        return None

    iof = ROOT.TIOFeatures()
    for feature in features:
        if isinstance(feature, int):
            iof.Set(feature)
            continue
        if feature not in IOFEATURES:
            allowed = ", ".join(sorted(IOFEATURES))
            raise ValueError(f"unknown IO feature {feature!r}; allowed values: {allowed}")
        iof.Set(IOFEATURES[feature])
    return iof


def _check_repack_options(sort: str, n_events: int | None) -> None:
    if sort not in SORT_MODES:
        raise ValueError(f"sort must be one of: {', '.join(sorted(SORT_MODES))}")
    if n_events is not None and n_events <= 0:
        raise ValueError("n_events must be greater than zero")


def _parse_count_with_suffix(value: int | str) -> tuple[int, bool]:
    if isinstance(value, int):
        return value, False

    raw = value.strip().lower()
    sign = 1
    if raw.startswith(("+", "-")):
        if raw[0] == "-":
            sign = -1
        raw = raw[1:]

    number = raw.rstrip("abcdefghijklmnopqrstuvwxyz")
    suffix = raw[len(number) :]
    if not number or suffix not in BYTE_SUFFIXES:
        allowed = ", ".join(sorted(s for s in BYTE_SUFFIXES if s))
        raise ValueError(f"invalid byte count {value!r}; supported suffixes: {allowed}")
    return sign * int(number) * BYTE_SUFFIXES[suffix], bool(suffix)


def _parse_byte_count(value: int | str) -> int:
    size, _ = _parse_count_with_suffix(value)
    if size <= 0:
        raise ValueError("basket size must be greater than zero")
    return size


def _parse_auto_flush(value: int | str | None) -> int | None:
    if value is None:
        return None

    parsed, has_suffix = _parse_count_with_suffix(value)
    if has_suffix and parsed != 0:
        return -abs(parsed)
    return parsed


def _parse_basket_size_spec(spec: str) -> tuple[str, int]:
    if "=" in spec:
        pattern, size = spec.split("=", 1)
        pattern = pattern.strip()
        if not pattern:
            raise ValueError(f"invalid basket size spec {spec!r}: empty branch pattern")
    else:
        pattern, size = "*", spec
    return pattern, _parse_byte_count(size)


def _normalize_basket_sizes(
    basket_size: int | str | None = None,
    basket_sizes: Mapping[str, int | str]
    | Iterable[str | tuple[str, int | str]]
    | str
    | None = None,
) -> tuple[tuple[str, int], ...]:
    normalized: list[tuple[str, int]] = []
    if basket_size is not None:
        normalized.append(("*", _parse_byte_count(basket_size)))

    if basket_sizes is None:
        return tuple(normalized)

    if isinstance(basket_sizes, Mapping):
        items = basket_sizes.items()
    elif isinstance(basket_sizes, str):
        items = [basket_sizes]
    else:
        items = basket_sizes

    for item in items:
        if isinstance(item, str):
            normalized.append(_parse_basket_size_spec(item))
            continue

        pattern, size = item
        if not pattern:
            raise ValueError("basket size branch pattern must not be empty")
        normalized.append((str(pattern), _parse_byte_count(size)))
    return tuple(normalized)


def _apply_basket_sizes(tree, basket_sizes: Sequence[tuple[str, int]]) -> None:
    for pattern, size in basket_sizes:
        tree.SetBasketSize(pattern, int(size))


def _apply_tree_write_options(
    tree,
    tree_options: TreeWriteOptions,
) -> None:
    if tree_options.auto_flush is not None:
        tree.SetAutoFlush(int(tree_options.auto_flush))
    _apply_basket_sizes(tree, tree_options.basket_sizes)


def _merge_root_files(
    output: Path,
    inputs: Sequence[Path],
    merge_options: MergeOptions,
) -> None:
    if not inputs:
        raise ValueError("cannot merge an empty input list")

    fm = ROOT.TFileMerger(False, False)
    fm.SetMsgPrefix(Path(__file__).stem)
    fm.SetPrintLevel(int(merge_options.verbose))
    fm.SetFastMethod(bool(merge_options.fast))
    fm.SetMergeOptions(f"SortBasketsBy{merge_options.sort.capitalize()}")

    iof = _iofeatures(merge_options.iofeatures)
    if iof is not None:
        fm.SetIOFeatures(iof)

    comp = _compression_settings(merge_options.compress, inputs[0])
    if not fm.OutputFile(str(output), "CREATE", comp):
        raise OSError(f"failed to create output ROOT file: {output}")

    for inp in inputs:
        if not fm.AddFile(str(inp)):
            raise OSError(f"failed to add input ROOT file: {inp}")

    merge_type = ROOT.TFileMerger.EPartialMergeType
    partial_merge_type = merge_type.kAll | merge_type.kRegular
    if merge_options.keep:
        partial_merge_type |= merge_type.kKeepCompression

    if not fm.PartialMerge(partial_merge_type):
        raise RuntimeError(f"ROOT merge failed while writing: {output}")


def _atomic_merge_root_files(
    output: Path,
    inputs: Sequence[Path],
    *,
    overwrite: bool = False,
    merge_options: MergeOptions,
) -> Path:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {output}")

    with TemporaryDirectory(prefix=f".{output.stem}.", dir=output.parent) as tmpdir:
        tmp_output = Path(tmpdir) / output.name
        _merge_root_files(
            tmp_output,
            inputs,
            merge_options,
        )
        if not tmp_output.exists() or tmp_output.stat().st_size == 0:
            raise RuntimeError(f"ROOT merge did not create a valid output: {tmp_output}")
        if output.exists() and not overwrite:
            raise FileExistsError(f"output file appeared during merge: {output}")
        tmp_output.replace(output)
    return output


def _copy_tree_range(
    source: Path,
    output: Path,
    event_tree: str,
    start: int,
    count: int,
    tree_options: TreeWriteOptions,
) -> Path:
    with _root_file(source) as source_file, _root_file(output, "RECREATE") as output_file:
        _copy_keys_except(
            source_file,
            output_file,
            event_tree,
            tree_options=tree_options,
        )

        tree = source_file.Get(event_tree)
        if not tree or not tree.InheritsFrom("TTree"):
            raise ValueError(f"{source} does not contain event tree {event_tree!r}")

        tree_dir, tree_leaf = _split_tree_path(event_tree)
        destination_dir = _mkdir_p(output_file, tree_dir)
        destination_dir.cd()
        _apply_tree_write_options(tree, tree_options)
        copied = tree.CopyTree("", "", int(count), int(start))
        if not copied:
            raise RuntimeError(
                f"failed to copy entries {start}:{start + count} from {event_tree!r} in {source}"
            )
        copied.SetName(tree_leaf)
        copied.SetTitle(tree.GetTitle())
        if copied.Write(tree_leaf, ROOT.TObject.kOverwrite) <= 0:
            raise RuntimeError(f"failed to write copied tree {event_tree!r} to {output}")
        output_file.Write()
    return output


def _rewrite_file_with_tree_options(
    source: Path,
    output: Path,
    tree_options: TreeWriteOptions,
) -> Path:
    with _root_file(source) as source_file, _root_file(output, "RECREATE") as output_file:
        _copy_keys_except(
            source_file,
            output_file,
            tree_options=tree_options,
        )
        output_file.Write()
    return output


def _split_tree_path(tree_name: str) -> tuple[str, str]:
    parts = [part for part in tree_name.split("/") if part]
    if not parts:
        raise ValueError("event_tree must not be empty")
    return "/".join(parts[:-1]), parts[-1]


def _mkdir_p(root_file, directory: str):
    current = root_file
    for part in [p for p in directory.split("/") if p]:
        next_dir = current.GetDirectory(part)
        if not next_dir:
            next_dir = current.mkdir(part)
            if not next_dir:
                raise OSError(f"failed to create directory {directory!r} in output file")
        current = next_dir
    return current


def _copy_keys_except(
    source_dir,
    output_dir,
    excluded_path: str | None = None,
    prefix: str = "",
    tree_options: TreeWriteOptions = TreeWriteOptions(),
) -> None:
    for key in source_dir.GetListOfKeys():
        name = key.GetName()
        key_path = f"{prefix}/{name}" if prefix else name
        if excluded_path is not None and key_path == excluded_path:
            continue

        obj = key.ReadObj()
        if obj.InheritsFrom("TDirectory"):
            child_output = output_dir.mkdir(name)
            if not child_output:
                raise OSError(f"failed to create directory {key_path!r}")
            _copy_keys_except(
                obj,
                child_output,
                excluded_path,
                key_path,
                tree_options=tree_options,
            )
            continue

        output_dir.cd()
        if obj.InheritsFrom("TTree"):
            _apply_tree_write_options(obj, tree_options)
            copied = obj.CopyTree("", "", int(obj.GetEntries()), 0)
            if not copied:
                raise RuntimeError(f"failed to copy tree {key_path!r}")
            copied.SetName(name)
            copied.SetTitle(obj.GetTitle())
            if copied.Write(name, ROOT.TObject.kOverwrite) <= 0:
                raise RuntimeError(f"failed to copy tree {key_path!r}")
            continue

        if obj.Write(name) <= 0:
            raise RuntimeError(f"failed to copy object {key_path!r}")


def _event_chunks(
    inputs: Sequence[Path],
    n_events: int,
    event_tree: str,
    *,
    progress: bool = False,
) -> list[list[EventSegment]]:
    chunks: list[list[EventSegment]] = []
    current: list[EventSegment] = []
    current_count = 0

    for path in _progress(
        inputs,
        enabled=progress,
        desc="Counting events",
        total=len(inputs),
        unit="file",
    ):
        entries = _tree_entries(path, event_tree)
        if entries == 0:
            current.append(EventSegment(path, 0, 0, entries))
            continue

        start = 0
        while start < entries:
            capacity = n_events - current_count
            take = min(capacity, entries - start)
            current.append(EventSegment(path, start, take, entries))
            current_count += take
            start += take
            if current_count == n_events:
                chunks.append(current)
                current = []
                current_count = 0

    if current:
        chunks.append(current)

    return chunks


def _output_paths(output_dir: Path, output_count: int) -> list[Path]:
    output_dir = output_dir.expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output must be a directory path: {output_dir}")
    return [output_dir / f"{index:03d}.root" for index in range(1, output_count + 1)]


def _check_outputs(outputs: Sequence[Path], inputs: Sequence[Path], overwrite: bool) -> None:
    seen = set()
    input_set = {path.resolve() for path in inputs}
    for output in outputs:
        resolved = output.expanduser().resolve()
        if resolved in seen:
            raise ValueError(f"duplicate output path planned: {output}")
        seen.add(resolved)
        if resolved in input_set:
            raise ValueError(f"refusing to write output over an input file: {output}")
        if output.exists() and not overwrite:
            raise FileExistsError(f"output file already exists: {output}")


def _stage_segment_files(
    chunk: Sequence[EventSegment],
    event_tree: str,
    temp_dir: Path,
    tree_options: TreeWriteOptions,
    *,
    progress: bool = False,
) -> list[Path]:
    staged: list[Path] = []
    items = enumerate(chunk)
    for index, segment in _progress(
        items,
        enabled=progress and len(chunk) > 1,
        desc="Staging segments",
        total=len(chunk),
        unit="segment",
        leave=False,
    ):
        if (
            not tree_options.requires_rewrite
            and segment.start == 0
            and segment.count == segment.total_entries
        ):
            staged.append(segment.path)
            continue

        staged_path = temp_dir / f"segment_{index:03d}.root"
        _copy_tree_range(
            segment.path,
            staged_path,
            event_tree,
            segment.start,
            segment.count,
            tree_options,
        )
        staged.append(staged_path)
    return staged


def _rewrite_inputs_for_tree_options(
    inputs: Sequence[Path],
    temp_dir: Path,
    tree_options: TreeWriteOptions,
    *,
    progress: bool = False,
) -> list[Path]:
    staged = []
    items = enumerate(inputs)
    for index, source in _progress(
        items,
        enabled=progress,
        desc="Rewriting trees",
        total=len(inputs),
        unit="file",
    ):
        staged_path = temp_dir / f"rewritten_{index:03d}.root"
        _rewrite_file_with_tree_options(
            source,
            staged_path,
            tree_options,
        )
        staged.append(staged_path)
    return staged


def _copy_inputs_to_temp(
    inputs: Sequence[Path],
    temp_dir: Path,
    *,
    progress: bool = False,
) -> list[Path]:
    copied = []
    items = enumerate(inputs)
    for index, source in _progress(
        items,
        enabled=progress,
        desc="Copying inputs",
        total=len(inputs),
        unit="file",
    ):
        destination = temp_dir / f"input_{index:03d}.root"
        copyfile(source, destination)
        copied.append(destination)
    return copied


def _build_repack_plan(
    inputs: Sequence[os.PathLike[str] | str],
    output: os.PathLike[str] | str | None,
    *,
    n_events: int | None,
    event_tree: str,
    overwrite: bool,
    progress: bool = False,
) -> RepackPlan:
    input_paths = tuple(_validate_input_files(_as_paths(inputs)))

    if n_events is None:
        if output is None:
            if len(input_paths) != 1:
                raise ValueError("in-place repacking is only supported for one input file")
            return RepackPlan(inputs=input_paths, outputs=(input_paths[0],), inplace=True)

        outputs = tuple(_output_paths(Path(output), 1))
        _check_outputs(outputs, input_paths, overwrite)
        return RepackPlan(inputs=input_paths, outputs=outputs)

    if output is None:
        raise ValueError("output is required when n_events is set")

    chunks = tuple(
        tuple(chunk)
        for chunk in _event_chunks(
            input_paths, n_events, event_tree, progress=progress
        )
    )
    outputs = tuple(_output_paths(Path(output), len(chunks)))
    _check_outputs(outputs, input_paths, overwrite)
    return RepackPlan(inputs=input_paths, outputs=outputs, chunks=chunks)


def _remap_chunks_to_working_inputs(
    chunks: Sequence[Sequence[EventSegment]],
    path_map: Mapping[Path, Path],
) -> tuple[tuple[EventSegment, ...], ...]:
    return tuple(
        tuple(
            EventSegment(
                path_map[segment.path],
                segment.start,
                segment.count,
                segment.total_entries,
            )
            for segment in chunk
        )
        for chunk in chunks
    )


def _execute_repack_plan(
    plan: RepackPlan,
    *,
    event_tree: str,
    merge_options: MergeOptions,
    tree_options: TreeWriteOptions,
    temp: bool,
    overwrite: bool,
    progress: bool,
) -> list[Path]:
    if plan.inplace:
        with TemporaryDirectory(
            prefix=f".{plan.inputs[0].stem}.", dir=plan.inputs[0].parent
        ) as tmpdir:
            tmp_path = Path(tmpdir)
            working_inputs = list(plan.inputs)
            if temp:
                working_inputs = _copy_inputs_to_temp(
                    working_inputs, tmp_path, progress=progress
                )
            if tree_options.requires_rewrite:
                working_inputs = _rewrite_inputs_for_tree_options(
                    working_inputs, tmp_path, tree_options, progress=progress
                )

            tmp_output = tmp_path / plan.inputs[0].name
            _merge_root_files(tmp_output, working_inputs, merge_options)
            tmp_output.replace(plan.inputs[0])
        return list(plan.outputs)

    if plan.chunks is None:
        with TemporaryDirectory(prefix="root_repack_inputs.") as tmpdir:
            tmp_path = Path(tmpdir)
            working_inputs = (
                _copy_inputs_to_temp(plan.inputs, tmp_path, progress=progress)
                if temp
                else list(plan.inputs)
            )
            if tree_options.requires_rewrite:
                working_inputs = _rewrite_inputs_for_tree_options(
                    working_inputs, tmp_path, tree_options, progress=progress
                )
            return [
                _atomic_merge_root_files(
                    plan.outputs[0],
                    working_inputs,
                    overwrite=overwrite,
                    merge_options=merge_options,
                )
            ]

    written = []
    with TemporaryDirectory(prefix="root_repack_chunks.") as tmpdir:
        tmp_path = Path(tmpdir)
        working_inputs = (
            _copy_inputs_to_temp(plan.inputs, tmp_path, progress=progress)
            if temp
            else list(plan.inputs)
        )
        working_chunks = _remap_chunks_to_working_inputs(
            plan.chunks, dict(zip(plan.inputs, working_inputs))
        )

        output_items = enumerate(zip(working_chunks, plan.outputs))
        for chunk_index, (chunk, output_path) in _progress(
            output_items,
            enabled=progress,
            desc="Writing outputs",
            total=len(plan.outputs),
            unit="file",
        ):
            chunk_tmp = tmp_path / f"chunk_{chunk_index:03d}"
            chunk_tmp.mkdir()
            staged = _stage_segment_files(
                chunk,
                event_tree,
                chunk_tmp,
                tree_options,
                progress=progress,
            )
            written.append(
                _atomic_merge_root_files(
                    output_path,
                    staged,
                    overwrite=overwrite,
                    merge_options=merge_options,
                )
            )
    return written


def root_repack(
    inputs: Sequence[os.PathLike[str] | str],
    output: os.PathLike[str] | str | None = None,
    *,
    n_events: int | None = None,
    event_tree: str = "Events",
    basket_size: int | str | None = None,
    basket_sizes: Mapping[str, int | str]
    | Iterable[str | tuple[str, int | str]]
    | str
    | None = None,
    auto_flush: int | str | None = None,
    fast: bool = False,
    keep: bool = False,
    sort: str = "branch",
    compress: str = "same",
    iofeatures: Iterable[str | int] | None = None,
    verbose: int = 0,
    temp: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    progress: bool = False,
) -> list[Path]:
    """Repack ROOT files and optionally split/merge them into event-count chunks.

    When ``output`` is set, it is treated as an output directory and files are
    written as ``001.root``, ``002.root``, and so on. When ``output`` is not set,
    one input file can be repacked in place.

    When ``n_events`` is set, outputs contain at most that many entries from
    ``event_tree``. Consecutive inputs smaller than ``n_events`` are merged
    into the same output until the threshold is reached. The final output
    contains the remaining entries when the total entry count is not an exact
    multiple of ``n_events``.

    ``basket_size`` applies one basket size to every branch. ``basket_sizes``
    accepts repeated ``"pattern=size"`` strings, ``(pattern, size)`` pairs, or
    a mapping. ROOT wildcard patterns such as ``"*"`` and ``"Muon_*"`` are
    passed to ``TTree.SetBasketSize``.

    ``auto_flush`` is passed to ``TTree.SetAutoFlush`` when TTrees are rewritten.
    Positive unsuffixed values are entry counts. Negative unsuffixed values keep
    ROOT's byte-count convention. Suffixed values such as ``"30M"`` are treated
    as byte thresholds and passed as negative values.

    Set ``progress=True`` to show tqdm progress bars for countable phases such
    as event counting, input copying, tree rewriting, segment staging, and
    output writing.
    """

    _check_repack_options(sort, n_events)

    merge_options = MergeOptions(
        fast=fast,
        keep=keep,
        sort=sort,
        compress=compress,
        iofeatures=tuple(iofeatures or ()),
        verbose=verbose,
    )
    tree_options = TreeWriteOptions(
        basket_sizes=_normalize_basket_sizes(basket_size, basket_sizes),
        auto_flush=_parse_auto_flush(auto_flush),
    )
    plan = _build_repack_plan(
        inputs,
        output,
        n_events=n_events,
        event_tree=event_tree,
        overwrite=overwrite,
        progress=progress and not dry_run,
    )

    if dry_run:
        return list(plan.outputs)

    return _execute_repack_plan(
        plan,
        event_tree=event_tree,
        merge_options=merge_options,
        tree_options=tree_options,
        temp=temp,
        overwrite=overwrite,
        progress=progress,
    )


def repack(
    output: os.PathLike[str] | str,
    input: Sequence[os.PathLike[str] | str],
    **kwargs,
) -> list[Path]:
    """Wrapper for the original ``repack(output, input)`` argument order."""

    return root_repack(input, output, **kwargs)


def _parser() -> argparse.ArgumentParser:
    compression_algorithms = ",".join(sorted(COMPRESSION["algorithm"]))
    compression_choices = ",".join(
        [f"{{{compression_algorithms}}}={{0..99}}"] + sorted(COMPRESSION["defaults"])
    )
    iofeature_choices = ",".join(sorted(IOFEATURES))

    parser = argparse.ArgumentParser(
        description="Repack ROOT files with optional compression, basket sorting, and event chunks."
    )
    parser.add_argument("-f", "--fast", action="store_true", help="use ROOT fast merge mode")
    parser.add_argument("-k", "--keep", action="store_true", help="keep input compression")
    parser.add_argument(
        "-s",
        "--sort",
        choices=sorted(SORT_MODES),
        default="branch",
        help="basket sorting mode",
    )
    parser.add_argument("-c", "--compress", default="same", metavar=f"{{{compression_choices}}}")
    parser.add_argument(
        "-i",
        "--iofeatures",
        "--iof",
        action="append",
        default=[],
        metavar=f"{{{iofeature_choices}}}",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT_DIR",
        help="output directory; files are written as 001.root, 002.root, ...",
    )
    parser.add_argument(
        "-n",
        "--n-events",
        type=int,
        help=(
            "maximum number of entries from --event-tree per output file; "
            "smaller consecutive inputs are merged up to this threshold"
        ),
    )
    parser.add_argument(
        "--event-tree",
        default="Events",
        help="tree whose entries define the event count for --n-events",
    )
    parser.add_argument(
        "--basket-size",
        action="append",
        default=[],
        metavar="BYTES|PATTERN=BYTES",
        help=(
            "rewrite TTree baskets with this size; repeat for branch patterns, "
            "for example --basket-size 64k --basket-size Muon_*=128k"
        ),
    )
    parser.add_argument(
        "--auto-flush",
        metavar="ENTRIES|BYTES",
        help=(
            "set TTree AutoFlush while rewriting; positive unsuffixed values "
            "mean entries, negative unsuffixed values keep ROOT byte-count "
            "semantics, and suffixed values like 30M mean bytes"
        ),
    )
    parser.add_argument(
        "-t", "--temp", action="store_true", help="copy inputs to temp before processing"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bars",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="validate and print planned outputs"
    )
    parser.add_argument("input", nargs="+", help="input ROOT file(s)")
    return parser


def _configure_logging(verbose: int) -> None:
    level = logging.DEBUG if verbose and verbose > 1 else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _log_outputs(outputs: Sequence[Path], *, dry_run: bool) -> None:
    action = "would write" if dry_run else "wrote"
    if len(outputs) == 1:
        LOGGER.info("%s %s", action, outputs[0])
        return

    LOGGER.info("%s %d files:", action, len(outputs))
    for output in outputs:
        LOGGER.info("  %s", output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    started = time()
    outputs = root_repack(
        args.input,
        args.output,
        n_events=args.n_events,
        event_tree=args.event_tree,
        basket_sizes=args.basket_size,
        auto_flush=args.auto_flush,
        fast=args.fast,
        keep=args.keep,
        sort=args.sort,
        compress=args.compress,
        iofeatures=args.iofeatures,
        verbose=args.verbose,
        temp=args.temp,
        overwrite=args.force,
        dry_run=args.dry_run,
        progress=not args.no_progress and not args.dry_run,
    )

    _log_outputs(outputs, dry_run=args.dry_run)
    if not args.dry_run:
        LOGGER.info("finished in %.1f seconds", time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
