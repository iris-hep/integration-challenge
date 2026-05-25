"""Save matplotlib figures into a per-session timestamped subdirectory.

The first call to :func:`save_fig` in a Python session caches a timestamp.
Every subsequent call in the same session reuses it, so all plots from one
notebook run land in the same ``<plots_dir>/<timestamp>/`` directory.
Restarting the kernel clears the module state and the next session gets a
fresh timestamp.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from intccms.utils.output import OutputDirectoryManager


_SESSION_TIMESTAMP: Optional[str] = None


def _session_timestamp() -> str:
    """Return the cached session timestamp, generating it on first call."""
    global _SESSION_TIMESTAMP
    if _SESSION_TIMESTAMP is None:
        _SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _SESSION_TIMESTAMP


def save_fig(
    fig: plt.Figure,
    output_manager: OutputDirectoryManager,
    name: str,
    timestamp: Optional[str] = None,
    fmt: str = "pdf",
) -> Path:
    """Save ``fig`` to ``<plots_dir>/<timestamp>/<name>.<fmt>``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    output_manager : OutputDirectoryManager
        Provides ``plots_dir``.
    name : str
        Filename stem (no extension).
    timestamp : str, optional
        Explicit timestamp. Defaults to the cached per-session timestamp.
    fmt : str
        File format. Default ``"pdf"``.

    Returns
    -------
    Path
        The full path the figure was written to.
    """
    ts = timestamp or _session_timestamp()
    out_dir = Path(output_manager.plots_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    return path
