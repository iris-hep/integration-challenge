"""Build a per-dataset processing timeline from roastcoffea chunk metrics.

Given the chunk-level records that ``roastcoffea.MetricsCollector`` collects
(``collector.chunk_metrics`` after ``extract_metrics_from_output``), this
module produces a time series of how many chunks per dataset were active in
each time bin. The output can be overlaid on a throughput time series to see
which datasets were running during throughput peaks or dips.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intccms.utils.output import OutputDirectoryManager


def strip_dataset_variation(name: str) -> str:
    """Reduce a dataset key like ``signal_0__JES_up`` to the bare process name.

    Dataset keys produced by ``format_dataset_key`` are
    ``<process>(_<idx>)?(__<variation>)?``. This drops the variation (split
    on the ``__`` delimiter) and any trailing ``_<digits>`` directory index,
    leaving the bare process name. Process names that contain underscores
    (e.g. ``qcd_2j``) are preserved since only a final ``_<digits>`` is
    stripped.
    """
    process_with_idx = name.split("__", 1)[0]
    return re.sub(r"_\d+$", "", process_with_idx)


def build_active_chunks_timeline(
    chunk_metrics: Iterable[dict[str, Any]],
    bin_seconds: float = 1.0,
    t0: Optional[float] = None,
    t1: Optional[float] = None,
    group_by: Optional[Callable[[str], str]] = strip_dataset_variation,
) -> pd.DataFrame:
    """Count chunks active per dataset (or process group) in each time bin.

    Parameters
    ----------
    chunk_metrics : iterable of dict
        Records with ``t_start``, ``t_end``, ``dataset`` (UNIX timestamps).
    bin_seconds : float
        Time bin width.
    t0, t1 : float, optional
        UNIX-timestamp range to cover. Default: span of the data.
    group_by : callable, optional
        Maps each raw dataset key to a column name. The default folds
        directory-index and variation suffixes into the bare process name
        (``signal_0__JES_up`` and ``signal_2__nominal`` both end up as
        ``signal``). Pass ``None`` to keep one column per raw dataset key.

    Returns
    -------
    pd.DataFrame
        Indexed by elapsed seconds since ``t0``. One column per group, value
        is the count of chunks whose ``[t_start, t_end)`` overlapped the bin.
    """
    records = [
        c for c in chunk_metrics
        if c.get("dataset") is not None and "t_start" in c and "t_end" in c
    ]
    if not records:
        raise ValueError("chunk_metrics has no usable records (need t_start, t_end, dataset)")

    starts = np.array([c["t_start"] for c in records], dtype=float)
    ends = np.array([c["t_end"] for c in records], dtype=float)
    if group_by is None:
        keys = np.array([c["dataset"] for c in records])
    else:
        keys = np.array([group_by(c["dataset"]) for c in records])

    if t0 is None:
        t0 = float(starts.min())
    if t1 is None:
        t1 = float(ends.max())

    n_bins = max(1, int(np.ceil((t1 - t0) / bin_seconds)))
    bin_edges = t0 + np.arange(n_bins + 1) * bin_seconds
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    unique_keys = sorted(set(keys.tolist()))
    counts = {k: np.zeros(n_bins, dtype=int) for k in unique_keys}

    for s, e, k in zip(starts, ends, keys):
        i0 = max(0, int((s - t0) / bin_seconds))
        i1 = min(n_bins, int(np.ceil((e - t0) / bin_seconds)))
        if i1 > i0:
            counts[k][i0:i1] += 1

    return pd.DataFrame(counts, index=bin_centers - t0)


def plot_dataset_timeline(
    chunk_metrics: Iterable[dict[str, Any]],
    bin_seconds: float = 1.0,
    ax: Optional[plt.Axes] = None,
    output_manager: Optional[OutputDirectoryManager] = None,
    output_path: Optional[Union[str, Path]] = None,
    filename: str = "dataset_timeline.png",
    group_by: Optional[Callable[[str], str]] = strip_dataset_variation,
    top_n: Optional[int] = 10,
) -> plt.Axes:
    """Stacked-area plot of active chunks per dataset over time.

    Parameters
    ----------
    chunk_metrics : iterable of dict
        See :func:`build_active_chunks_timeline`.
    bin_seconds : float
        Time bin width passed through to the data builder.
    ax : matplotlib.axes.Axes, optional
        Plot onto an existing axes (useful for overlaying on a throughput
        panel via ``plt.subplots(2, 1, sharex=True)``). If None a new figure
        is created.
    output_manager : OutputDirectoryManager, optional
        If set and ``output_path`` is not, the figure is saved to
        ``output_manager.plots_dir / filename``.
    output_path : str or Path, optional
        Explicit destination for the figure. Takes precedence over
        ``output_manager``.
    filename : str
        Filename used when saving via ``output_manager``.
    group_by : callable, optional
        Forwarded to :func:`build_active_chunks_timeline`. Default folds
        index/variation suffixes into the bare process name. Pass ``None``
        to keep one series per raw dataset key.
    top_n : int, optional
        Cap the number of series in the legend. Datasets are ranked by the
        sum of their active-chunk counts; everything outside the top *n*
        is summed into a single ``other`` series. Default 10. Pass ``None``
        to disable the cap.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object the plot was drawn on.
    """
    df = build_active_chunks_timeline(
        chunk_metrics, bin_seconds=bin_seconds, group_by=group_by
    )
    if top_n is not None and len(df.columns) > top_n:
        totals = df.sum(axis=0).sort_values(ascending=False)
        keep = list(totals.index[:top_n])
        other = df.drop(columns=keep).sum(axis=1)
        df = df[keep].copy()
        df["other"] = other
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(df.index, df.T.values, labels=df.columns)
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Active chunks")
    ax.set_title(f"Dataset processing timeline (bin={bin_seconds}s)")
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.margins(x=0)
    ax.figure.tight_layout()

    save_to = None
    if output_path is not None:
        save_to = Path(output_path)
    elif output_manager is not None:
        save_to = Path(output_manager.plots_dir) / filename
    if save_to is not None:
        ax.figure.savefig(save_to, dpi=150, bbox_inches="tight")
    return ax
