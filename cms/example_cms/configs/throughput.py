"""Helpers for picking branches to read with the throughput processor.

Closely follows iris-hep/idap-200gbps/input_files/size_per_branch.ipynb. Use
this when you want to say "read N% of a NanoAOD file" and get back branches
dicts you can hand to ``preprocess_config["branches"]`` /
``preprocess_config["mc_branches"]`` and
:class:`intccms.analysis.TwoHundredGbpsProcessor`.

The entry point :func:`get_branches_for_fraction` always returns a
``(branches, mc_branches)`` tuple. If you only pass an MC file, the second
dict is empty. If you also pass a representative data file, branches that
exist in MC but not in data are split into the second dict — ready to drop
into ``preprocess_config["mc_branches"]``.

Example::

    from example_cms.configs.throughput import get_branches_for_fraction

    branches, mc_branches = get_branches_for_fraction(
        "root://.../mc_sample.root",
        target_fraction=0.5,
        cache_path="example_cms/configs/branch_sizes.json",
        data_file="root://.../data_sample.root",  # optional
    )
    preprocess_config = {
        "branches": branches,
        "mc_branches": mc_branches,
        "skimming": skimming_config,
    }

The three smaller functions (:func:`measure_branch_sizes`,
:func:`select_branches_for_fraction`, :func:`branches_to_dict`) and
:func:`find_mc_only_branches` are also exposed if you want to reuse one
measurement for several target fractions, inspect per-branch sizes, or do
your own selection.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import uproot


def measure_branch_sizes(
    file_path: str,
    tree_name: str = "Events",
    cache_path: Optional[str] = None,
) -> Tuple[Dict[str, float], float]:
    """Measure how many MB each branch costs to read.

    Opens the file once and reads each branch on its own, recording how
    many bytes uproot pulled to satisfy that read. Returns the per-branch
    sizes and the total file size in MB.

    Parameters
    ----------
    file_path : str
        Path or URI to the NanoAOD file. Anything uproot can open works
        (local path, ``root://`` URI, etc.).
    tree_name : str, optional
        Events tree name. Defaults to ``"Events"``.
    cache_path : str or None, optional
        Optional JSON cache. If the file exists, the result is loaded from
        it instead of re-reading the NanoAOD file. If it doesn't exist, the
        result is written there after measuring so subsequent runs are fast.

    Returns
    -------
    sizes_mb : dict[str, float]
        Branch name -> size in MB.
    file_size_mb : float
        Total file size in MB.
    """
    if cache_path and Path(cache_path).exists():
        cached = json.loads(Path(cache_path).read_text())
        file_size_mb = cached.pop("_file_size_mb")
        return cached, file_size_mb

    sizes_mb: Dict[str, float] = {}
    tree = uproot.open({file_path: tree_name})
    cur_mb = tree.file.source.num_requested_bytes / 1000**2
    for key in tree.keys():
        tree.arrays([key])
        new_mb = tree.file.source.num_requested_bytes / 1000**2
        sizes_mb[key] = new_mb - cur_mb
        cur_mb = new_mb

    # File size: stat for local paths; otherwise use the sum of branch
    # sizes plus the metadata read at open time as a close approximation.
    try:
        file_size_mb = Path(file_path).stat().st_size / 1000**2
    except (OSError, ValueError):
        file_size_mb = tree.file.source.num_requested_bytes / 1000**2

    if cache_path:
        out = dict(sizes_mb)
        out["_file_size_mb"] = file_size_mb
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(out, sort_keys=True, indent=4))

    return sizes_mb, file_size_mb


def select_branches_for_fraction(
    sizes_mb: Dict[str, float],
    file_size_mb: float,
    target_fraction: float = 1.0,
    veto: Tuple[str, ...] = (),
) -> List[str]:
    """Pick the biggest branches first until they cover the target fraction.

    Walks branches from biggest to smallest, adding each one until the
    selected total reaches ``target_fraction * file_size_mb``. Branches
    listed in ``veto`` are skipped.

    Parameters
    ----------
    sizes_mb : dict[str, float]
        Per-branch sizes from :func:`measure_branch_sizes`.
    file_size_mb : float
        Total file size in MB (the second return value of
        :func:`measure_branch_sizes`).
    target_fraction : float, optional
        Fraction of the file to read. Must be in ``(0, 1.0]``. Defaults
        to ``1.0`` (every branch).
    veto : tuple of str, optional
        Branch names to skip.

    Returns
    -------
    list[str]
        The selected branch names, biggest first.
    """
    if not 0 < target_fraction <= 1.0:
        raise ValueError(
            f"target_fraction must be in (0, 1.0], got {target_fraction}"
        )

    target_mb = target_fraction * file_size_mb
    selected: List[str] = []
    accumulated = 0.0
    for key, size in sorted(sizes_mb.items(), key=lambda kv: kv[1], reverse=True):
        if key in veto:
            continue
        selected.append(key)
        accumulated += size
        if accumulated >= target_mb:
            break
    return selected


def branches_to_dict(branch_names: List[str]) -> Dict[str, List[str]]:
    """Group a flat NanoAOD branch list into the ``preprocess.branches`` shape.

    NanoAOD names branches like ``Jet_pt`` (collection prefix in upper case
    + underscore + field). Anything that doesn't fit that pattern (e.g.
    ``run``) is put under ``"event"``.

    Examples
    --------
    >>> branches_to_dict(["Jet_pt", "Muon_eta", "PSWeight", "run"])
    {'Jet': ['pt'], 'Muon': ['eta'], 'event': ['PSWeight', 'run']}
    """
    grouped: Dict[str, List[str]] = {}
    for name in branch_names:
        head, sep, tail = name.partition("_")
        if sep and head[:1].isupper():
            grouped.setdefault(head, []).append(tail)
        else:
            grouped.setdefault("event", []).append(name)
    return grouped


def find_mc_only_branches(
    mc_file: str,
    data_file: str,
    tree_name: str = "Events",
) -> Set[str]:
    """Return branches that exist in the MC file but not in the data file.

    Just opens each file and lists its branches — no branch contents are
    read, so this is cheap even over XRootD. Assumes both files are
    representative; a stripped-down test data file with missing branches
    will produce false positives.

    Parameters
    ----------
    mc_file : str
        Path or URI to a representative MC NanoAOD file.
    data_file : str
        Path or URI to a representative data NanoAOD file.
    tree_name : str, optional
        Events tree name. Defaults to ``"Events"``.

    Returns
    -------
    set[str]
        Branch names present in MC but missing from data.
    """
    with uproot.open({mc_file: tree_name}) as t:
        mc = set(t.keys())
    with uproot.open({data_file: tree_name}) as t:
        data = set(t.keys())
    return mc - data


def get_branches_for_fraction(
    file_path: str,
    target_fraction: float = 1.0,
    tree_name: str = "Events",
    cache_path: Optional[str] = None,
    veto: Tuple[str, ...] = (),
    data_file: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Measure, pick, and group branches in one call.

    The entry point for the common case: give it a file and a target
    fraction, get back a branches dict ready to drop into
    ``preprocess_config["branches"]``. Internally calls
    :func:`measure_branch_sizes`, :func:`select_branches_for_fraction`,
    and :func:`branches_to_dict`.

    If ``data_file`` is also given, the picked branches are split into
    MC-vs-data using :func:`find_mc_only_branches`, so you can plug the
    second dict straight into ``preprocess_config["mc_branches"]``.

    Parameters
    ----------
    file_path : str
        Path or URI to a NanoAOD file. If ``data_file`` is also given,
        this is treated as the MC file.
    target_fraction : float, optional
        Fraction of file size to read. Defaults to ``1.0`` (every branch).
    tree_name : str, optional
        Events tree name. Defaults to ``"Events"``.
    cache_path : str or None, optional
        See :func:`measure_branch_sizes`.
    veto : tuple of str, optional
        See :func:`select_branches_for_fraction`.
    data_file : str or None, optional
        Path or URI to a representative data NanoAOD file. When given,
        branches that exist in MC but not in data are split out into the
        second returned dict.

    Returns
    -------
    branches : dict[str, list[str]]
        Branches dict for ``preprocess_config["branches"]``.
    mc_branches : dict[str, list[str]]
        Branches dict for ``preprocess_config["mc_branches"]``. Empty
        when ``data_file`` is not given.
    """
    sizes_mb, file_size_mb = measure_branch_sizes(
        file_path, tree_name=tree_name, cache_path=cache_path
    )
    selected = select_branches_for_fraction(
        sizes_mb, file_size_mb, target_fraction=target_fraction, veto=veto
    )

    if data_file is None:
        return branches_to_dict(selected), {}

    mc_only = find_mc_only_branches(file_path, data_file, tree_name=tree_name)
    common = [b for b in selected if b not in mc_only]
    mc_only_selected = [b for b in selected if b in mc_only]
    return branches_to_dict(common), branches_to_dict(mc_only_selected)
