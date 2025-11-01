"""FilesetManager for building coffea filesets from skimming manifests.

This module provides functionality to read skimming manifests (created by
SkimmingManager) and construct coffea-compatible fileset dictionaries for
use with the processor-based analysis workflow.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class FilesetManager:
    """Manager for building coffea filesets from skimming manifests.

    This class reads manifest.json files created during the skimming phase
    and constructs coffea-compatible fileset dictionaries that can be used
    with coffea's Runner and processor infrastructure.

    Attributes
    ----------
    skimmed_dir : Path
        Base directory containing skimmed files and manifests
    format : str
        Output format from skimming ("parquet" or "root_ttree")

    Examples
    --------
    >>> manager = FilesetManager(
    ...     skimmed_dir=Path("outputs/skimmed"),
    ...     format="parquet"
    ... )
    >>> fileset = manager.build_fileset(["signal__nominal", "background__nominal"])
    >>> # Use fileset with coffea Runner
    >>> runner = Runner(...)
    >>> output = runner(fileset, "Events", processor_instance=processor)
    """

    def __init__(self, skimmed_dir: Path | str, format: str):
        """Initialize FilesetManager.

        Parameters
        ----------
        skimmed_dir : Path or str
            Base directory containing skimmed output files and manifests
        format : str
            Format of skimmed files ("parquet" or "root_ttree")
        """
        self.skimmed_dir = Path(skimmed_dir)
        self.format = format
        logger.info(
            f"Initialized FilesetManager: dir={self.skimmed_dir}, format={self.format}"
        )

    def read_manifest(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Read manifest.json for a specific dataset.

        Parameters
        ----------
        dataset_name : str
            Dataset/fileset key (e.g., "signal__nominal")

        Returns
        -------
        list of dict
            Manifest entries containing file metadata

        Raises
        ------
        FileNotFoundError
            If manifest file doesn't exist for the dataset
        """
        manifest_path = self.skimmed_dir / dataset_name / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found for dataset '{dataset_name}' at {manifest_path}"
            )

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        logger.debug(f"Read manifest for {dataset_name}: {len(manifest)} entries")
        return manifest

    def build_fileset(self, dataset_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Build coffea-compatible fileset from manifests.

        Reads manifests for the specified datasets and constructs a fileset
        dictionary compatible with coffea's Runner.

        Parameters
        ----------
        dataset_names : list of str
            List of dataset names to include in fileset

        Returns
        -------
        dict
            Coffea-compatible fileset with structure:
            {
                "dataset_name": {
                    "files": [list of file paths],
                    "metadata": {
                        "dataset": str,
                        "format": str,
                        "treename": str (for ROOT),
                        "total_processed_events": int,
                    }
                }
            }
        """
        fileset = {}

        for dataset_name in dataset_names:
            try:
                manifest = self.read_manifest(dataset_name)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {dataset_name}: {e}")
                continue

            # Extract file paths from manifest
            files = [entry["output_file"] for entry in manifest]

            # Calculate total processed events
            total_processed = sum(
                entry.get("processed_events", 0) for entry in manifest
            )

            # Build metadata
            metadata = {
                "dataset": dataset_name,
                "format": self.format,
                "total_processed_events": total_processed,
            }

            # Add treename for ROOT format
            if self.format in ("root", "root_ttree"):
                # Get treename from first manifest entry
                if manifest:
                    metadata["treename"] = manifest[0].get("treename", "Events")
                else:
                    metadata["treename"] = "Events"

            fileset[dataset_name] = {
                "files": files,
                "metadata": metadata,
            }

            logger.info(
                f"Added {dataset_name} to fileset: {len(files)} files, "
                f"{total_processed} events"
            )

        return fileset

    def build_fileset_from_datasets(
        self, datasets: List[Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Build fileset from Dataset objects.

        Convenience method that extracts fileset_keys from Dataset objects
        and builds the fileset.

        Parameters
        ----------
        datasets : list of Dataset
            Dataset objects with fileset_keys attribute

        Returns
        -------
        dict
            Coffea-compatible fileset
        """
        # Extract all fileset keys from datasets
        fileset_keys = []
        for dataset in datasets:
            fileset_keys.extend(dataset.fileset_keys)

        logger.info(f"Building fileset from {len(datasets)} datasets")
        return self.build_fileset(fileset_keys)
