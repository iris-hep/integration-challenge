"""Pipeline components for event skimming.

Core stages for loading, filtering, extracting, and saving events.
"""

from .stages import load_events, apply_selection, extract_columns, save_events

__all__ = [
    "load_events",
    "apply_selection",
    "extract_columns",
    "save_events",
]
