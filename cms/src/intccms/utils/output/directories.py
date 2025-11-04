"""
Centralized output directory management.

This module provides a unified interface for managing all output directories,
using a descriptor pattern to eliminate boilerplate and provide consistent access.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


class DirectoryDescriptor:
    """
    Descriptor that handles both configurable and standard directory paths.

    This descriptor automatically creates directories on access and supports
    user-configurable paths for certain directories (metadata, skimmed, histograms, models).

    Parameters
    ----------
    default_subdir : str
        Default subdirectory name under root_output_dir
    configurable : bool, default=False
        If True, allows user to override with custom path via constructor

    Examples
    --------
    >>> class Manager:
    ...     plots_dir = DirectoryDescriptor("plots", configurable=False)
    ...     metadata_dir = DirectoryDescriptor("metadata", configurable=True)
    """

    def __init__(self, default_subdir: str, configurable: bool = False):
        self.default_subdir = default_subdir
        self.configurable = configurable
        self.attr_name = None

    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to class attribute."""
        self.attr_name = name

    def __get__(self, obj, objtype=None):
        """Return directory path, creating it if needed."""
        if obj is None:
            return self

        # If configurable, check if user provided custom path in constructor
        if self.configurable:
            custom_path = getattr(obj, f"_{self.attr_name}_custom", None)
            if custom_path:
                dir_path = Path(custom_path)
            else:
                dir_path = obj.root_output_dir / self.default_subdir
        else:
            # Standard path - always under root_output_dir
            dir_path = obj.root_output_dir / self.default_subdir

        # Auto-create directory
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path


class OutputDirectoryManager:
    """
    Centralized manager for all output directories in the analysis.

    Provides a single source of truth for output paths with:
    - Descriptor-based directory access (eliminates boilerplate)
    - Configurable paths for metadata, skimmed, histograms, and models
    - Unified .get() interface with validation
    - Support for custom directories

    Parameters
    ----------
    root_output_dir : str or Path
        Root directory for all analysis outputs
    cache_dir : str or Path, optional
        Cache directory for temporary files. If None, uses system temp with 'intccms_cache' subdirectory
    metadata_dir : str or Path, optional
        Directory for metadata JSON files. If None, uses root_output_dir/metadata/
    skimmed_dir : str or Path, optional
        Directory for skimmed files. If None, uses root_output_dir/skimmed/
    histograms_dir : str or Path, optional
        Directory for histogram files. If None, uses root_output_dir/histograms/
    models_dir : str or Path, optional
        Directory for model files. If None, uses root_output_dir/models/

    Examples
    --------
    >>> # Simple case - all defaults
    >>> manager = OutputDirectoryManager(root_output_dir="outputs")
    >>> manager.plots_dir  # outputs/plots (auto-created)
    >>> manager.get("histograms")  # outputs/histograms

    >>> # Custom paths for certain directories
    >>> manager = OutputDirectoryManager(
    ...     root_output_dir="outputs",
    ...     metadata_dir="/shared/metadata",
    ...     histograms_dir="../old_run/histograms"
    ... )
    >>> manager.metadata_dir  # /shared/metadata (auto-created)
    >>> manager.plots_dir  # outputs/plots

    >>> # Subdirectories
    >>> manager.get("plots", "correlations")  # outputs/plots/correlations

    >>> # Custom directories
    >>> manager.get_custom("datasets", "signal")  # outputs/datasets/signal
    """

    # Configurable directories (user can override paths)
    metadata_dir = DirectoryDescriptor("metadata", configurable=True)
    skimmed_dir = DirectoryDescriptor("skimmed", configurable=True)
    histograms_dir = DirectoryDescriptor("histograms", configurable=True)
    models_dir = DirectoryDescriptor("models", configurable=True)

    # Standard directories (always under root_output_dir)
    plots_dir = DirectoryDescriptor("plots", configurable=False)
    statistics_dir = DirectoryDescriptor("statistics", configurable=False)

    # Known directory names mapping (for .get() validation)
    _known_dirs = {
        "plots": "plots_dir",
        "models": "models_dir",
        "histograms": "histograms_dir",
        "statistics": "statistics_dir",
        "metadata": "metadata_dir",
        "skimmed": "skimmed_dir",
    }

    def __init__(
        self,
        root_output_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        metadata_dir: Optional[Union[str, Path]] = None,
        skimmed_dir: Optional[Union[str, Path]] = None,
        histograms_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
    ):
        # Normalize root output directory path
        self.root_output_dir = Path(root_output_dir).expanduser().resolve()

        # Normalize cache directory path or use system temp directory
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "intccms_cache"

        # Store custom paths for configurable directories
        self._metadata_dir_custom = Path(metadata_dir).expanduser().resolve() if metadata_dir else None
        self._skimmed_dir_custom = Path(skimmed_dir).expanduser().resolve() if skimmed_dir else None
        self._histograms_dir_custom = Path(histograms_dir).expanduser().resolve() if histograms_dir else None
        self._models_dir_custom = Path(models_dir).expanduser().resolve() if models_dir else None

        # Create root and cache directories
        self.root_output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory manager initialized with root: {self.root_output_dir}")
        if self._metadata_dir_custom:
            logger.info(f"Using custom metadata directory: {self._metadata_dir_custom}")
        if self._skimmed_dir_custom:
            logger.info(f"Using custom skimmed directory: {self._skimmed_dir_custom}")
        if self._histograms_dir_custom:
            logger.info(f"Using custom histograms directory: {self._histograms_dir_custom}")
        if self._models_dir_custom:
            logger.info(f"Using custom models directory: {self._models_dir_custom}")

    def get(self, name: str, subdir: Optional[str] = None) -> Path:
        """
        Get a known output directory, optionally with subdirectory.

        Parameters
        ----------
        name : str
            Directory name. Must be one of: plots, models, histograms,
            statistics, metadata, skimmed
        subdir : str, optional
            Optional subdirectory within the named directory

        Returns
        -------
        Path
            Path to requested directory (auto-created)

        Raises
        ------
        ValueError
            If name is not a known directory

        Examples
        --------
        >>> manager.get("plots")  # Base plots directory
        >>> manager.get("plots", "correlation_matrices")  # Subdirectory
        """
        if name not in self._known_dirs:
            known = ", ".join(self._known_dirs.keys())
            raise ValueError(
                f"Unknown directory '{name}'. Known directories: {known}. "
                f"Use get_custom('{name}') for custom directories."
            )

        attr_name = self._known_dirs[name]
        base_dir = getattr(self, attr_name)

        if subdir:
            result = base_dir / subdir
            result.mkdir(parents=True, exist_ok=True)
            return result

        return base_dir

    def get_custom(self, name: str, subdir: Optional[str] = None) -> Path:
        """
        Create a custom directory under root_output_dir.

        Parameters
        ----------
        name : str
            Custom directory name (e.g., "datasets", "my_analysis")
        subdir : str, optional
            Optional subdirectory within the custom directory

        Returns
        -------
        Path
            Path to custom directory (auto-created)

        Examples
        --------
        >>> manager.get_custom("datasets", "signal")  # root_output_dir/datasets/signal
        >>> manager.get_custom("my_analysis")  # root_output_dir/my_analysis
        """
        base_dir = self.root_output_dir / name
        base_dir.mkdir(parents=True, exist_ok=True)

        if subdir:
            result = base_dir / subdir
            result.mkdir(parents=True, exist_ok=True)
            return result

        return base_dir

    def list_known_directories(self) -> list[str]:
        """
        Return list of known directory names.

        Returns
        -------
        list[str]
            List of valid directory names for use with .get()
        """
        return list(self._known_dirs.keys())

    def list_structure(self) -> Dict[str, Any]:
        """
        Get a summary of the current directory structure.

        Returns
        -------
        dict
            Dictionary describing the directory structure
        """
        structure = {
            "root_output_dir": str(self.root_output_dir),
            "cache_dir": str(self.cache_dir),
            "metadata_dir": str(self.metadata_dir),
            "skimmed_dir": str(self.skimmed_dir),
            "histograms_dir": str(self.histograms_dir),
            "models_dir": str(self.models_dir),
            "plots_dir": str(self.plots_dir),
            "statistics_dir": str(self.statistics_dir),
        }
        return structure
