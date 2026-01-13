"""Dataset management for configurable paths, cross-sections, and metadata."""

from .manager import DatasetManager
from .models import Dataset

__all__ = ["Dataset", "DatasetManager"]
