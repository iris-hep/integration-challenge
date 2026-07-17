"""Coffea integration for metadata extraction.

This module provides integration with coffea's preprocessing functionality
to extract WorkItems from ROOT files.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from coffea.processor.executor import WorkItem
from coffea import processor
from coffea.nanoevents import NanoAODSchema

from intccms.metadata_extractor.manager import DEFAULT_PREPROCESS_SKIPBADFILES


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

    def __init__(
        self,
        executor: Any = None,
        schema: Any = None,
        chunksize: int = 500_000,
        skipbadfiles: Union[bool, Tuple[Type[BaseException], ...]] = DEFAULT_PREPROCESS_SKIPBADFILES,
    ):
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
            Number of events per chunk for WorkItem splitting, default 500_000
        skipbadfiles : bool or tuple of exception types, optional
            Forwarded to coffea's Runner. Defaults to DEFAULT_PREPROCESS_SKIPBADFILES.
            Pass False to hard-fail on any bad file, True for coffea's built-in
            OSError-only behavior, or a custom tuple of exception types.
        """


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
            skipbadfiles=skipbadfiles,
        )

        logger.debug(
            f"Initialized CoffeaMetadataExtractor with {type(executor).__name__} "
            f"and chunksize={chunksize}"
        )

    def extract_metadata(
        self,
        fileset: Dict[str, Dict[str, str]],
        uproot_options: Optional[Dict[str, Any]] = None,
    ) -> List[WorkItem]:
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
        uproot_options : dict, optional
            Passed to coffea ``Runner.preprocess`` (``uproot.open`` options). Support for
            forwarding these options requires **coffea v2026.4.0 or later**.

        Returns
        -------
        List[WorkItem]
            WorkItem objects with file metadata and entry ranges for chunked processing.

        Raises
        ------
        Exception
            If coffea preprocessing fails
        """
        logger.info(
            "Extracting metadata using coffea Runner.preprocess "
            f"(chunksize={self.runner.chunksize})"
        )
        try:
            # Run the coffea preprocess function on the provided fileset
            workitems = self.runner.preprocess(
                fileset,
                uproot_options=uproot_options,
            )
            # Convert the generator returned by preprocess to a list of WorkItems
            workitems_list = list(workitems)
            logger.info(f"Extracted {len(workitems_list)} WorkItems from {len(fileset)} datasets")
            return workitems_list
        except Exception as e:
            logger.error(f"Error during coffea preprocessing: {e}")
            raise 
