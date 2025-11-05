"""Save and load benchmark measurements for later reanalysis.

Measurements are timestamped directories containing all data needed to reproduce
benchmark analysis without re-executing the workflow. This enables:
- Comparing multiple benchmark runs
- Reanalyzing with different parameters
- Archiving performance baselines
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from intccms.utils.output import OutputDirectoryManager


def save_measurement(
    metrics: Dict[str, Any],
    t0: float,
    t1: float,
    output_manager: OutputDirectoryManager,
    measurement_name: Optional[str] = None,
    fileset: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save benchmark measurement to disk for later reanalysis.

    Creates a timestamped directory with all data needed to reproduce the
    benchmark analysis without re-executing the workflow.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Performance metrics dictionary (must be JSON-serializable).
        Should contain simple types: numbers, strings, lists, dicts.
        NOT awkward arrays or histograms - those are analysis outputs.
    t0 : float
        Start timestamp (from time.perf_counter())
    t1 : float
        End timestamp (from time.perf_counter())
    output_manager : OutputDirectoryManager
        Output directory manager for accessing benchmarks directory
    measurement_name : Optional[str]
        Custom name for measurement directory. If None, uses timestamp YYYY-MM-DD_HH-MM-SS.
    fileset : Optional[Dict[str, Any]]
        Input fileset configuration (saved for reproducibility)
    config : Optional[Dict[str, Any]]
        Analysis configuration (saved for reproducibility)

    Returns
    -------
    Path
        Path to the created measurement directory

    Notes
    -----
    Creates the following structure in output_manager.benchmarks_dir/<measurement_name>/:
    - metrics/<timestamp>.json: Timestamped metrics (preserves history)
    - start_end_time.txt: Wall clock timing (t0, t1)
    - fileset.json: Input fileset configuration
    - config.json: Analysis configuration
    - metadata.json: Measurement metadata (timestamp, metrics version, etc.)

    Examples
    --------
    >>> from intccms.utils.output import OutputDirectoryManager
    >>> import time
    >>>
    >>> output_manager = OutputDirectoryManager(root_output_dir="output")
    >>> t0 = time.perf_counter()
    >>> output, report = runner(...)
    >>> t1 = time.perf_counter()
    >>>
    >>> metrics = {
    ...     "wall_time": t1 - t0,
    ...     "events_processed": 1000000,
    ...     "throughput_hz": 50000.0,
    ... }
    >>> measurement_path = save_measurement(
    ...     metrics, t0, t1, output_manager, measurement_name="run_001"
    ... )
    """
    # Create timestamped measurement directory name if not provided
    if measurement_name is None:
        measurement_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get benchmarks directory (auto-created by DirectoryDescriptor)
    # and create measurement subdirectory
    measurement_path = output_manager.benchmarks_dir / measurement_name
    measurement_path.mkdir(parents=True, exist_ok=True)

    # Create metrics subdirectory for timestamped metric files
    metrics_dir = measurement_path / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Save metrics with timestamp to preserve history without overwriting
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # milliseconds precision
    metrics_file = metrics_dir / f"{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save timing information
    with open(measurement_path / "start_end_time.txt", "w") as f:
        f.write(f"{t0},{t1}\n")

    # Save fileset if provided
    if fileset is not None:
        with open(measurement_path / "fileset.json", "w") as f:
            json.dump(fileset, f, indent=2, default=str)

    # Save config if provided
    if config is not None:
        with open(measurement_path / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    # Save measurement metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "wall_time": t1 - t0,
        "metrics_version": "1.0",
        "format": "intccms_measurement_v1",
    }
    with open(measurement_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return measurement_path


def load_measurement(measurement_path: Path) -> Tuple[Dict[str, Any], float, float]:
    """Load saved measurement for reanalysis.

    Parameters
    ----------
    measurement_path : Path
        Directory containing saved measurement

    Returns
    -------
    metrics : Dict[str, Any]
        Performance metrics dictionary
    t0 : float
        Start timestamp
    t1 : float
        End timestamp

    Raises
    ------
    FileNotFoundError
        If measurement directory or required files don't exist
    ValueError
        If measurement format is invalid or corrupted

    Examples
    --------
    >>> from pathlib import Path
    >>>
    >>> measurement_path = Path("measurements/2025-01-15_14-30-00")
    >>> metrics, t0, t1 = load_measurement(measurement_path)
    >>>
    >>> # Analyze metrics
    >>> wall_time = t1 - t0
    >>> throughput = metrics["events_processed"] / wall_time
    """
    measurement_path = Path(measurement_path)

    if not measurement_path.exists():
        raise FileNotFoundError(f"Measurement directory not found: {measurement_path}")

    # Load all metrics from metrics directory
    metrics_dir = measurement_path / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    # Get all JSON files sorted by timestamp (oldest first)
    metrics_files = sorted(metrics_dir.glob("*.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found in: {metrics_dir}")

    # Merge all metrics files (later files override earlier ones)
    metrics = {}
    for metrics_file in metrics_files:
        with open(metrics_file, "r") as f:
            file_metrics = json.load(f)
            metrics.update(file_metrics)

    # Load timing
    timing_file = measurement_path / "start_end_time.txt"
    if not timing_file.exists():
        raise FileNotFoundError(f"Timing file not found: {timing_file}")

    with open(timing_file, "r") as f:
        timing_line = f.readline().strip()
        try:
            t0_str, t1_str = timing_line.split(",")
            t0 = float(t0_str)
            t1 = float(t1_str)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid timing format in {timing_file}: {e}")

    return metrics, t0, t1


def list_measurements(measurements_dir: Path) -> list[Path]:
    """List all measurements in a directory.

    Parameters
    ----------
    measurements_dir : Path
        Root directory containing measurements

    Returns
    -------
    list[Path]
        List of measurement directories, sorted by timestamp (newest first)

    Examples
    --------
    >>> from pathlib import Path
    >>>
    >>> measurements = list_measurements(Path("output/measurements"))
    >>> for m in measurements:
    ...     print(f"{m.name}: {m / 'metadata.json'}")
    """
    measurements_dir = Path(measurements_dir)
    if not measurements_dir.exists():
        return []

    # Find all subdirectories with metadata.json
    measurements = [
        d for d in measurements_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ]

    # Sort by name (which is timestamp) in reverse (newest first)
    return sorted(measurements, reverse=True)
