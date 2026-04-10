"""cProfile-based profiling utilities for coffea processors.

Provides MergeableProfile for aggregating cProfile stats across
distributed chunks via coffea's tree reduction, and helpers for
categorizing function times into semantic buckets.

Note: cProfile only profiles the current thread. If uproot dispatches
I/O to background threads (e.g. via ThreadPoolExecutor), network I/O
time will be undercounted. CPU-bound work (decompression, array
construction) is captured accurately.
"""

import cProfile
import marshal
import os
import pstats
import tempfile
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

# Bucket classification rules.
# Each entry is (bucket_name, list_of_substrings). A cProfile stats key
# (filename, lineno, funcname) is assigned to the FIRST bucket whose
# substrings match either the filename or funcname. Order matters:
# more specific buckets come first.
_BUCKET_RULES: List[Tuple[str, List[str]]] = [
    ("decompression", [
        "uproot/compression",
        "uproot\\compression",
        "zlib.decompress",
        "cramjam",
        "lzma.decompress",
        "xxhash",
        "zstandard",
        "lz4",
    ]),
    ("network_io", [
        "uproot/source",
        "uproot\\source",
        "fsspec/",
        "fsspec\\",
        "xrdcl",
        "_io.open",
        "_io.BufferedReader",
        "socket",
        "ssl.py",
        "http/client",
        "urllib",
    ]),
    ("array_construction", [
        "uproot/interpretation",
        "uproot\\interpretation",
        "uproot/models/TBasket",
        "uproot\\models\\TBasket",
        "awkward/",
        "awkward\\",
        "numpy/",
        "numpy\\",
    ]),
    ("uproot_overhead", [
        "uproot/reading",
        "uproot\\reading",
        "uproot/_util",
        "uproot\\_util",
        "uproot/extras",
        "uproot\\extras",
        "uproot/cache",
        "uproot\\cache",
    ]),
]


def _merge_callers(
    a: Dict[Tuple, Tuple],
    b: Dict[Tuple, Tuple],
) -> Dict[Tuple, Tuple]:
    """Merge two cProfile callers dicts by summing numeric fields.

    Each value is ``(pcalls, ncalls, tottime, cumtime)``.
    """
    merged = dict(a)
    for key, (pc2, nc2, tt2, ct2) in b.items():
        if key in merged:
            pc1, nc1, tt1, ct1 = merged[key]
            merged[key] = (pc1 + pc2, nc1 + nc2, tt1 + tt2, ct1 + ct2)
        else:
            merged[key] = (pc2, nc2, tt2, ct2)
    return merged


def _merge_stats(
    a: Dict[Tuple, Tuple],
    b: Dict[Tuple, Tuple],
) -> Dict[Tuple, Tuple]:
    """Merge two cProfile ``.stats`` dicts by summing numeric fields.

    Each value is ``(pcalls, ncalls, tottime, cumtime, callers_dict)``.
    """
    merged = dict(a)
    for key, (pc2, nc2, tt2, ct2, callers2) in b.items():
        if key in merged:
            pc1, nc1, tt1, ct1, callers1 = merged[key]
            merged[key] = (
                pc1 + pc2,
                nc1 + nc2,
                tt1 + tt2,
                ct1 + ct2,
                _merge_callers(callers1, callers2),
            )
        else:
            merged[key] = (pc2, nc2, tt2, ct2, dict(callers2))
    return merged


class MergeableProfile:
    """Wrapper around cProfile stats that merges via ``__add__``.

    cProfile stores profiling results as a dict::

        {(filename, lineno, funcname): (pcalls, ncalls, tottime, cumtime, callers)}

    All numeric fields are additive across independent runs. Coffea's
    tree reduction calls ``__add__`` on accumulator values, so 40k chunk
    profiles reduce to a single merged profile by the time results reach
    the client.

    Parameters
    ----------
    stats : dict, optional
        Raw ``cProfile.Profile.stats`` dictionary.
    """

    def __init__(self, stats: Optional[Dict] = None):
        self.stats: Dict = stats or {}

    def __add__(self, other: "MergeableProfile") -> "MergeableProfile":
        if not isinstance(other, MergeableProfile):
            return NotImplemented
        return MergeableProfile(stats=_merge_stats(self.stats, other.stats))

    def __radd__(self, other: Any) -> "MergeableProfile":
        if other == 0:
            return self
        return self.__add__(other)

    def __repr__(self) -> str:
        return f"MergeableProfile({len(self.stats)} entries)"

    def to_pstats(self, filename: str) -> None:
        """Write merged stats to a pstats-compatible ``.prof`` file.

        The resulting file can be opened with ``snakeviz``, ``flameprof``,
        or ``gprof2dot``.

        Parameters
        ----------
        filename : str
            Output path for the ``.prof`` file.
        """
        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as f:
            tmpname = f.name
            marshal.dump(self.stats, f)
        try:
            stats = pstats.Stats(tmpname)
            stats.dump_stats(filename)
        finally:
            os.unlink(tmpname)

    def print_stats(self, sort_by: str = "cumulative", limit: int = 30) -> None:
        """Print the merged profile stats to stdout.

        Parameters
        ----------
        sort_by : str
            Sort key for ``pstats.Stats.sort_stats``.
        limit : int
            Maximum number of rows to print.
        """
        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as f:
            tmpname = f.name
            marshal.dump(self.stats, f)
        try:
            stats = pstats.Stats(tmpname)
            stats.sort_stats(sort_by)
            stats.print_stats(limit)
        finally:
            os.unlink(tmpname)


def categorize_profile(stats: Dict) -> Dict[str, float]:
    """Categorize cProfile stats into semantic time buckets.

    Groups ``tottime`` (exclusive time per function) into buckets based
    on filename/funcname substring matching. Uses ``tottime`` rather
    than ``cumtime`` to avoid double-counting parent/child relationships.

    Parameters
    ----------
    stats : dict
        Raw ``cProfile.Profile.stats`` dictionary.

    Returns
    -------
    dict
        Mapping of bucket name to total seconds. Includes ``"other"``
        for uncategorized functions and ``"total_s"`` for the sum.
    """
    buckets: Dict[str, float] = {name: 0.0 for name, _ in _BUCKET_RULES}
    buckets["other"] = 0.0
    total = 0.0

    for (filename, _lineno, funcname), (_, _, tottime, _, _) in stats.items():
        total += tottime
        check_str = f"{filename} {funcname}"
        matched = False
        for bucket_name, patterns in _BUCKET_RULES:
            if any(pat in check_str for pat in patterns):
                buckets[bucket_name] += tottime
                matched = True
                break
        if not matched:
            buckets["other"] += tottime

    buckets["total_s"] = total
    return buckets
