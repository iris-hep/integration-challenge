"""Coffea integration for metadata extraction.

This module provides integration with coffea's preprocessing functionality
to extract WorkItems from ROOT files.
"""

import logging
from typing import Any, Dict, List
from lzma import LZMAError

from coffea.processor.executor import WorkItem
from coffea.processor.executor import UprootMissTreeError

logger = logging.getLogger(__name__)


class CoffeaMetadataExtractor:
    """
    Extracts metadata from ROOT files using coffea preprocessing.

    This class uses coffea's preprocessing to generate WorkItems containing
    file paths, entry ranges, and UUIDs for chunked parallel processing.

    Attributes
    ----------
    runner : coffea.processor.Runner
        The coffea processor runner configured for preprocessing
    """

    def __init__(self, executor: Any = None, schema: Any = None, chunksize: int = 100_000):
        """
        Initialize CoffeaMetadataExtractor with configurable executor.

        Parameters
        ----------
        executor : coffea.processor executor, optional
            Executor for preprocessing (FuturesExecutor, DaskExecutor, etc.).
            If None, uses FuturesExecutor by default.
        schema : coffea schema, optional
            Schema for parsing ROOT files (e.g., NanoAODSchema).
            If None, uses NanoAODSchema by default.
        chunksize : int, optional
            Number of events per chunk for WorkItem splitting, default 100_000
        """
        from coffea import processor
        from coffea.nanoevents import NanoAODSchema

        # Use defaults if not provided
        if executor is None:
            executor = processor.FuturesExecutor()

        if schema is None:
            schema = NanoAODSchema

        self.runner = processor.Runner(
            executor=executor,
            schema=schema,
            savemetrics=True,
            chunksize=chunksize,
            skipbadfiles=(OSError, LZMAError, UprootMissTreeError, Exception),
        )

        logger.debug(
            f"Initialized CoffeaMetadataExtractor with {type(executor).__name__} "
            f"and chunksize={chunksize}"
        )

    def extract_metadata(self, fileset: Dict[str, Dict[str, str]]) -> List[WorkItem]:
        """
        Extract WorkItems from fileset using coffea preprocessing.

        WorkItems are file chunks for parallel processing. Coffea automatically splits
        large ROOT files into chunks based on entry count (controlled by chunksize).
        Each WorkItem contains: filename, tree name, entry range (start/stop), file UUID,
        and dataset key. These are later processed independently for skimming.

        Parameters
        ----------
        fileset : Dict[str, Dict[str, str]]
            Coffea-compatible fileset mapping dataset keys to file paths and tree names.

        Returns
        -------
        List[WorkItem]
            WorkItem objects with file metadata and entry ranges for chunked processing.

        Raises
        ------
        Exception
            If coffea preprocessing fails
        """
        logger.info("Extracting metadata using coffea.dataset_tools.preprocess")
        try:
            # Run the coffea preprocess function on the provided fileset
            workitems = self.runner.preprocess(fileset)
            # Convert the generator returned by preprocess to a list of WorkItems
            workitems_list = list(workitems)
            logger.info(f"Extracted {len(workitems_list)} WorkItems from {len(fileset)} datasets")
            return workitems_list
        except Exception as e:
            logger.error(f"Error during coffea preprocessing: {e}")
            raise
