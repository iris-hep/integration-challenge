"""Build a per-dataset processing timeline from roastcoffea chunk metrics.

Given the chunk-level records that ``roastcoffea.MetricsCollector`` collects
(``collector.chunk_metrics`` after ``extract_metrics_from_output``), this
module produces a time series of how many chunks per dataset were active in
each time bin. The output can be overlaid on a throughput time series to see
which datasets were running during throughput peaks or dips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intccms.utils.output import OutputDirectoryManager


def build_active_chunks_timeline(
    chunk_metrics: Iterable[dict[str, Any]],
    bin_seconds: float = 1.0,
    t0: Optional[float] = None,
    t1: Optional[float] = None,
) -> pd.DataFrame:
    """Count chunks active per dataset in each time bin.

    Parameters
    ----------
    chunk_metrics : iterable of dict
        Records with ``t_start``, ``t_end``, ``dataset`` (UNIX timestamps).
    bin_seconds : float
        Time bin width.
    t0, t1 : float, optional
        UNIX-timestamp range to cover. Default: span of the data.

    Returns
    -------
    pd.DataFrame
        Indexed by elapsed seconds since ``t0``. One column per dataset, value
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
    datasets = np.array([c["dataset"] for c in records])

    if t0 is None:
        t0 = float(starts.min())
    if t1 is None:
        t1 = float(ends.max())

    n_bins = max(1, int(np.ceil((t1 - t0) / bin_seconds)))
    bin_edges = t0 + np.arange(n_bins + 1) * bin_seconds
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    unique_datasets = sorted(set(datasets.tolist()))
    counts = {ds: np.zeros(n_bins, dtype=int) for ds in unique_datasets}

    for s, e, ds in zip(starts, ends, datasets):
        i0 = max(0, int((s - t0) / bin_seconds))
        i1 = min(n_bins, int(np.ceil((e - t0) / bin_seconds)))
        if i1 > i0:
            counts[ds][i0:i1] += 1

    return pd.DataFrame(counts, index=bin_centers - t0)


def plot_dataset_timeline(
    chunk_metrics: Iterable[dict[str, Any]],
    bin_seconds: float = 1.0,
    ax: Optional[plt.Axes] = None,
    output_manager: Optional[OutputDirectoryManager] = None,
    output_path: Optional[Union[str, Path]] = None,
    filename: str = "dataset_timeline.png",
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

    Returns
    -------
    matplotlib.axes.Axes
        The axes object the plot was drawn on.
    """
    df = build_active_chunks_timeline(chunk_metrics, bin_seconds=bin_seconds)
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
