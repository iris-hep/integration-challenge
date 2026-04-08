"""Branch-size introspection for the throughput-measurement processor.

Modeled on iris-hep/idap-200gbps/input_files/size_per_branch.ipynb. Use this to
turn 'I want to read N% of a NanoAOD file' into a config-ready branches dict
that can be plugged straight into ``preprocess_config["branches"]`` and fed to
:class:`intccms.analysis.TwoHundredGbpsProcessor`.

Typical usage::

    from example_cms.configs.throughput import get_branches_for_fraction

    # One call: measure (cached) + pick the largest branches that sum to
    # 50% of the file + return the dict shape preprocess_config expects.
    preprocess_config = {
        "branches": get_branches_for_fraction(
            "root://.../sample.root",
            target_fraction=0.5,
            cache_path="example_cms/configs/branch_sizes.json",
        ),
        "mc_branches": {"event": ["genWeight"]},
        "skimming": skimming_config,
    }

The three lower-level primitives (:func:`measure_branch_sizes`,
:func:`select_branches_for_fraction`, :func:`branches_to_dict`) are exposed
for advanced cases — re-using one measurement for multiple target fractions,
inspecting per-branch sizes, applying custom selection logic.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import uproot


def measure_branch_sizes(
    file_path: str,
    tree_name: str = "Events",
    cache_path: Optional[str] = None,
) -> Tuple[Dict[str, float], float]:
    """Measure on-disk read cost (MB) of every branch in a NanoAOD tree.

    Opens the file once and reads each branch in isolation, tracking the
    delta in ``tree.file.source.num_requested_bytes``. The result is the
    per-branch contribution to a full read.

    Parameters
    ----------
    file_path : str
        Path or URI to the NanoAOD file. Anything uproot can open works
        (local path, ``root://`` URI, etc.).
    tree_name : str, optional
        Name of the events tree. Defaults to ``"Events"``.
    cache_path : str or None, optional
        If given and the file exists, the measurement is loaded from
        ``cache_path`` and ``file_path`` is not touched. If given and the
        file does not exist, the result is written there after measuring.
        Cache format matches IDAP's ``branch_sizes.json``: a flat JSON
        object of ``{branch_name: size_mb}`` plus a ``_file_size_mb``
        sentinel key.

    Returns
    -------
    sizes_mb : dict[str, float]
        Mapping of branch name to size in MB.
    file_size_mb : float
        Total on-disk size of the file in MB.
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
    veto: Tuple[str, ...] = ("LHEPdfWeight",),
) -> List[str]:
    """Greedy-select branches largest-first to hit a target read fraction.

    Iterates branches sorted by descending size, accumulating until the
    selected total reaches ``target_fraction * file_size_mb``. The default
    veto matches the IDAP notebook — ``LHEPdfWeight`` varies unpredictably
    across files and skews benchmarks.

    Parameters
    ----------
    sizes_mb : dict[str, float]
        Per-branch sizes from :func:`measure_branch_sizes`.
    file_size_mb : float
        Total file size in MB (the second return value of
        :func:`measure_branch_sizes`).
    target_fraction : float, optional
        Fraction of ``file_size_mb`` to read. Must be in ``(0, 1.0]``.
        Defaults to ``1.0`` (every branch).
    veto : tuple of str, optional
        Branch names to exclude regardless of size.

    Returns
    -------
    list[str]
        Selected branch names in descending size order.
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
    """Convert a flat NanoAOD branch list to ``Config.preprocess.branches`` shape.

    NanoAOD branches use ``Collection_field`` naming where ``Collection``
    starts with an uppercase letter (e.g. ``Jet_pt``, ``GenPart_eta``).
    Anything else (or names without an underscore, e.g. ``run``) is
    treated as an event-level branch and lands under ``"event"``.

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


def get_branches_for_fraction(
    file_path: str,
    target_fraction: float = 1.0,
    tree_name: str = "Events",
    cache_path: Optional[str] = None,
    veto: Tuple[str, ...] = ("LHEPdfWeight",),
) -> Dict[str, List[str]]:
    """One-call wrapper: measure → select → convert.

    Most users only need this. Measures branch sizes (with optional cache),
    selects greedy by size to hit ``target_fraction`` of the file, and
    returns the result already in the nested
    ``Config.preprocess.branches`` shape.

    Parameters
    ----------
    file_path : str
        Path or URI to the NanoAOD file.
    target_fraction : float, optional
        Fraction of file size to read. Defaults to ``1.0`` (every branch).
    tree_name : str, optional
        Events tree name. Defaults to ``"Events"``.
    cache_path : str or None, optional
        See :func:`measure_branch_sizes`.
    veto : tuple of str, optional
        See :func:`select_branches_for_fraction`.

    Returns
    -------
    dict[str, list[str]]
        Branches dict ready to plug into ``preprocess_config["branches"]``.
    """
    sizes_mb, file_size_mb = measure_branch_sizes(
        file_path, tree_name=tree_name, cache_path=cache_path
    )
    selected = select_branches_for_fraction(
        sizes_mb, file_size_mb, target_fraction=target_fraction, veto=veto
    )
    return branches_to_dict(selected)
