import json
import logging
import os
from collections import defaultdict
from pathlib import Path
import pexpect
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import awkward as ak
import uproot
import dask
from dask.distributed import as_completed

from intccms.metadata_extractor.io import collect_file_paths

logger = logging.getLogger(__name__)


def load_dotenv(path: str = ".env", overwrite: bool = True) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Ignores comments, blank lines, and keys already set in the environment.
    Does nothing if the file does not exist.

    Args:
        path: Path to the .env file (default: ".env" in current directory).
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            if key not in os.environ or overwrite: 
                os.environ[key] = value

def nested_defaultdict_to_dict(nested_structure: Any) -> dict:
    """
    Recursively convert any nested defaultdicts into standard Python dictionaries.

    Parameters
    ----------
    nested_structure : Any
        A nested structure possibly containing defaultdicts.

    Returns
    -------
    dict
        Fully converted structure using built-in dict.
    """
    if isinstance(nested_structure, defaultdict):
        return {
            key: nested_defaultdict_to_dict(value)
            for key, value in nested_structure.items()
        }
    elif isinstance(nested_structure, dict):
        return {
            key: nested_defaultdict_to_dict(value)
            for key, value in nested_structure.items()
        }
    return nested_structure


def recursive_to_backend(data_structure: Any, backend: str = "jax") -> Any:
    """
    Recursively convert all Awkward Arrays in a data structure to the specified backend.

    Parameters
    ----------
    data_structure : Any
        Input data structure possibly containing Awkward Arrays.
    backend : str
        Target backend to convert arrays to (e.g. 'jax', 'cpu').

    Returns
    -------
    Any
        Data structure with Awkward Arrays converted to the desired backend.
    """
    if isinstance(data_structure, ak.Array):
        # Convert only if not already on the target backend
        return (
            ak.to_backend(data_structure, backend)
            if ak.backend(data_structure) != backend
            else data_structure
        )
    elif isinstance(data_structure, Mapping):
        # Recurse into dictionary values
        return {
            key: recursive_to_backend(value, backend)
            for key, value in data_structure.items()
        }
    elif isinstance(data_structure, Sequence) and not isinstance(
        data_structure, (str, bytes)
    ):
        # Recurse into list or tuple elements
        return [
            recursive_to_backend(value, backend) for value in data_structure
        ]
    else:
        # Leave unchanged if not an Awkward structure
        return data_structure


# ---------------------------------------------------------------------------
# XCache warming (closely follows distributed_xrdcp.ipynb)
# ---------------------------------------------------------------------------
def warm_xcache(
    dataset_manager: Any,
    client: Any,
    redirector: Optional[str] = None,
    max_files: Optional[int] = None,
    processes: Optional[List[str]] = None,
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Warm xcache by reading all dataset files through it using dask workers.

    Dispatches ``xrdcp`` calls to dask workers (so files are pulled from
    within the cluster, close to xcache) and waits for them to finish.

    Parameters
    ----------
    dataset_manager : DatasetManager
        An initialized :class:`intccms.datasets.DatasetManager`.
    client : dask.distributed.Client
        A live dask distributed client.
    redirector : str or None, optional
        Override the per-dataset redirector. If None, each dataset's own
        redirector is used.
    max_files : int or None, optional
        Stop after this many files total. None means all files.
    processes : list of str or None, optional
        Only warm these processes. None means all.
    max_retries: the maximum number of times to retry if some futures fail

    Returns
    -------
    results : list of dict
        One dict per file with keys ``"fname"``, ``"t0"``, ``"t1"``,
        ``"GBread"``.
    meta : dict
        Summary with ``"n_files"``, ``"total_GB"``, ``"wall_time_s"``,
        ``"total_Gbps"``, ``"processtime_s"`` (sum of per-worker times),
        ``"per_worker_Gbps"``.
    """
    def _xrdcp_one(fname: str) -> Dict[str, Any]:
        """Run ``xrdcp <fname> /dev/null -f`` and return timing + size info.
    
        This is the function that gets sent to dask workers. It uses
        ``pexpect`` to capture xrdcp's progress output and parse the file
        size from it.
        """
        t0 = time.time()
        child = pexpect.spawn(f"xrdcp {fname} /dev/null -f")
        child.expect(pexpect.EOF, timeout=600) 
        t1 = time.time()
        res = child.before.decode()
        size = res.split("\r")[-2].split("/")[0][1:]
        if "MB" in size:
            size_in_GB = float(size[:-2]) / 1000
        elif "GB" in size:
            size_in_GB = float(size[:-2])
        elif "kB" in size:
            size_in_GB = float(size[:-2]) / 1000 / 1000
        else:
            raise ValueError(f"cannot handle size: {size}")
        return {"fname": fname, "t0": t0, "t1": t1, "GBread": size_in_GB}
        
    # Collect file URIs from DatasetManager
    process_names = processes or dataset_manager.list_processes()
    all_uris: List[str] = []
    for name in process_names:
        dirs = dataset_manager.get_dataset_directories(name)
        redir = redirector or dataset_manager.get_redirector(name)
        for d in dirs:
            skip = dataset_manager.config.skip_files
            all_uris.extend(collect_file_paths(d, redirector=redir, skip_files=skip))
            if max_files and len(all_uris) >= max_files:
                break
        if max_files and len(all_uris) >= max_files:
            break

    if max_files:
        all_uris = all_uris[:max_files]

    logger.info(f"Warming xcache: {len(all_uris)} files")

    # Dispatch to dask workers
    t0 = time.time()
    tasks = [dask.delayed(_xrdcp_one)(uri) for uri in all_uris]
    futures = client.compute(tasks)

    results = []
    failed = []
    to_run = futures
    nretries = 0
    while nretries < max_retries and to_run:
        try:
            for future in as_completed(to_run):
                try:
                    results.append(future.result())
                except Exception as e:
                    failed.append(future)
        except KeyboardInterrupt:
            logger.info("Interrupted — cancelling remaining futures")
            client.cancel(futures)                                                                                                                                                            
        to_run = failed
        failed = []
        nretries += 1

    t1 = time.time()
    wall_time = t1 - t0

    total_GB = sum(r["GBread"] for r in results)
    processtime = sum(r["t1"] - r["t0"] for r in results)

    meta = {
        "n_files": len(results),
        "total_GB": total_GB,
        "wall_time_s": wall_time,
        "total_Gbps": total_GB * 8 / wall_time if wall_time > 0 else 0,
        "processtime_s": processtime,
        "per_worker_Gbps": total_GB * 8 / processtime if processtime > 0 else 0,
    }
    logger.info(
        f"Warming done: {meta['total_GB']:.1f} GB in {meta['wall_time_s']:.1f}s "
        f"= {meta['total_Gbps']:.2f} Gbps"
    )

    return results, meta


# ---------------------------------------------------------------------------
# Branch-size helpers for the throughput processor
#
# Closely follows iris-hep/idap-200gbps/input_files/size_per_branch.ipynb.
# Use these when you want to say "read N% of a NanoAOD file" and get back
# branches dicts for preprocess_config["branches"] / ["mc_branches"].
# ---------------------------------------------------------------------------


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
        if any(v in key for v in veto):
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


def prepare_branches_from_list(
    branch_list: List[str],
    mc_file: Optional[str] = None,
    data_file: Optional[str] = None,
    tree_name: str = "Events",
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Group a fixed flat branch list and optionally split off the MC-only subset.

    Use this when the branch list is known up front (e.g. replicating a
    reference workflow) instead of being picked by size fraction. Internally
    calls :func:`branches_to_dict` to group the flat names into the
    ``preprocess.branches`` shape.

    When ``mc_file`` and ``data_file`` are both given, branches that exist
    in the MC file but not in the data file are returned as the second
    dict, ready to drop into ``preprocess_config["mc_branches"]``.

    Parameters
    ----------
    branch_list : list[str]
        Flat NanoAOD branch names, e.g. ``["Jet_pt", "GenPart_eta", "run"]``.
    mc_file : str or None, optional
        Path or URI to a representative MC NanoAOD file. Required with
        ``data_file`` for the MC/data split.
    data_file : str or None, optional
        Path or URI to a representative data NanoAOD file. Required with
        ``mc_file`` for the MC/data split.
    tree_name : str, optional
        Events tree name. Defaults to ``"Events"``.

    Returns
    -------
    branches : dict[str, list[str]]
        Grouped branches dict, ready for ``preprocess_config["branches"]``.
    mc_branches : dict[str, list[str]]
        MC-only subset, ready for ``preprocess_config["mc_branches"]``.
        Empty when either file is not given.
    """
    branches = branches_to_dict(branch_list)

    if mc_file is None or data_file is None:
        return branches, {}

    mc_only = find_mc_only_branches(mc_file, data_file, tree_name=tree_name)
    mc_only_selected = [b for b in branch_list if b in mc_only]
    return branches, branches_to_dict(mc_only_selected)


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
    mc_only_selected = [b for b in selected if b in mc_only]
    return branches_to_dict(selected), branches_to_dict(mc_only_selected)
