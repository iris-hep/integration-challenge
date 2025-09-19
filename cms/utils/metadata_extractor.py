"""
NanoAOD dataset metadata extraction and management.

This module builds filesets from ROOT file listings, extracts metadata using
coffea preprocessing tools, and creates WorkItem objects that are later processed
as chunks during the skimming phase.

Outputs three main JSON files:
- fileset.json: Maps dataset names to ROOT file paths and tree names
- workitems.json: Contains WorkItem objects with file chunks and entry ranges
- nanoaods.json: Summary of event counts per dataset and process

"""

# Standard library imports
import base64
import dataclasses
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
from coffea.processor.executor import WorkItem
from rich.pretty import pretty_repr

# Local application imports
from utils.datasets import ConfigurableDatasetManager


# Configure module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _parse_dataset(dataset_key: str) -> Tuple[str, str]:
    """
    Splits a dataset key like 'process__variation' into ('process', 'variation').

    If no '__' is present, 'nominal' is used as the variation.
    """
    if "__" in dataset_key:
        proc, var = dataset_key.split("__", 1)
    else:
        proc, var = dataset_key, "nominal"
    return proc, var


def get_root_file_paths(
    directory: Union[str, Path],
    identifiers: Optional[Union[int, List[int]]] = None,
) -> List[str]:
    """
    Collects ROOT file paths from `.txt` listing files in a directory.

    Searches for `*.txt` files in the specified directory (or specific
    `<id>.txt` files if `identifiers` is given) and reads each line as a
    ROOT file path.

    Parameters
    ----------
    directory : str or Path
        Path to the folder containing text listing files.
    identifiers : int or list of ints, optional
        Specific listing file IDs (without `.txt`) to process. If `None`, all
        `.txt` files in the folder are used.

    Returns
    -------
    List[str]
        A list of ROOT file paths as strings.

    Raises
    ------
    FileNotFoundError
        If no listing files are found or a specified file is missing.
    """
    dir_path = Path(directory)
    # Determine which text files to parse
    if identifiers is None:
        # If no specific identifiers, glob for all .txt files
        listing_files = list(dir_path.glob("*.txt"))
    else:
        # If identifiers are provided, construct specific file paths
        ids = [identifiers] if isinstance(identifiers, int) else identifiers
        listing_files = [dir_path / f"{i}.txt" for i in ids]

    # Raise error if no listing files are found
    if not listing_files:
        raise FileNotFoundError(f"No listing files found in {dir_path}")

    root_paths: List[str] = []
    # Iterate through each listing file
    for txt_file in listing_files:
        # Ensure the listing file exists
        if not txt_file.is_file():
            raise FileNotFoundError(f"Missing listing file: {txt_file}")
        # Read each non-empty line as a file path
        for line in txt_file.read_text().splitlines():
            path_str = line.strip()
            if path_str:
                root_paths.append(path_str)

    return root_paths


class FilesetBuilder:
    """
    Builds and saves a coffea-compatible fileset from dataset configurations.

    This class reads dataset listings and constructs a fileset dictionary
    suitable for `coffea` processors.

    Attributes
    ----------
    dataset_manager : ConfigurableDatasetManager
        Manages dataset configurations, including paths and tree names.
    """

    def __init__(self, dataset_manager: ConfigurableDatasetManager):
        """
        Initializes the FilesetBuilder.

        Parameters
        ----------
        dataset_manager : ConfigurableDatasetManager
            A dataset manager instance (required).
        """
        self.dataset_manager = dataset_manager

    def build_fileset(
        self, identifiers: Optional[Union[int, List[int]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Builds a coffea-compatible fileset mapping.

        Iterates through configured processes, collects ROOT file paths, and
        and constructs a dictionary where keys are dataset names (process__variation)
        and values contain a mapping of file paths to tree names.

        Parameters
        ----------
        identifiers : Optional[Union[int, List[int]]], optional
            Specific listing file IDs (without `.txt`) to process. If `None`, all
            `.txt` files in the process's listing directory are used.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The constructed fileset in the format:
            `{dataset_name: {"files": {file_path: tree_name}}}`
        """
        fileset: Dict[str, Dict[str, Any]] = {}

        max_files = self.dataset_manager.config.max_files

        if max_files and max_files <= 0:
            raise ValueError("max_files must be None or a positive integer.")

        # Iterate over each process configured in the dataset manager
        for process_name in self.dataset_manager.list_processes():
            logger.info(f"Building fileset for process: {process_name}")

            # Get the directory where listing files are located for this process
            listing_dir = self.dataset_manager.get_dataset_directory(process_name)
            # Get the tree name (e.g., "Events") for ROOT files of this process
            tree_name = self.dataset_manager.get_tree_name(process_name)

            try:
                # Collect all ROOT file paths from the listing files
                file_paths = get_root_file_paths(listing_dir, identifiers)[:max_files]

                # Define the dataset key for coffea (process__variation)
                # For now, assuming a "nominal" variation if not explicitly specified
                variation_label = "nominal"
                if process_name != "data":
                    dataset_key = f"{process_name}__{variation_label}"
                else:
                    dataset_key = process_name
                # Create the fileset entry: map each file path to its tree name
                fileset[dataset_key] = {
                    "files": {file_path: tree_name for file_path in file_paths},
                    "metadata": {
                                "process": process_name,
                                "variation": variation_label,
                                "xsec": self.dataset_manager.get_cross_section(process_name)
                            }
                }

                logger.debug(f"Added {len(file_paths)} files for {dataset_key}")

            except FileNotFoundError as fnf:
                # Log an error if listing files are not found and continue to next process
                logger.error(f"Could not build fileset for {process_name}: {fnf}")
                continue

        return fileset

    def save_fileset(
        self, fileset: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Saves the built fileset to a JSON file.

        The output path is determined by the `metadata_output_dir` configured
        in the `dataset_manager`.

        Parameters
        ----------
        fileset : Dict[str, Dict[str, Any]]
            The fileset mapping to save.
        """
        # Construct the full path for the fileset JSON file
        output_dir = Path(self.dataset_manager.config.metadata_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fileset_path = output_dir / "fileset.json"

        # Ensure the parent directory exists
        fileset_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the fileset to the JSON file with pretty-printing
        with fileset_path.open("w") as f:
            json.dump(fileset, f, indent=4)

        logger.info(f"Fileset JSON saved to {fileset_path}")


class CoffeaMetadataExtractor:
    """
    Extracts metadata from ROOT files using `coffea.dataset_tools.preprocess`.

    This class generates a list of `WorkItem` objects containing metadata like
    objects containing metadata like file paths, entry ranges, and UUIDs.

    Attributes
    ----------
    runner : coffea.processor.Runner
        The coffea processor runner configured for preprocessing.
    """

    def __init__(self) -> None:
        """
        Initializes the CoffeaMetadataExtractor.

        Configures a `coffea.processor.Runner` instance with an iterative executor
        and NanoAOD schema for metadata extraction.
        """
        # Import coffea processor and NanoAODSchema here to avoid circular imports
        # or unnecessary imports if this class is not used.
        from coffea import processor
        from coffea.nanoevents import NanoAODSchema

        # Initialize the coffea processor Runner with an iterative executor
        # and the NanoAODSchema for parsing NanoAOD files.
        self.runner = processor.Runner(
            executor=processor.IterativeExecutor(),
            schema=NanoAODSchema,
            savemetrics=True,
            # Use a small chunksize for demonstration/testing to simulate multiple chunks
            chunksize=100_000,
        )

    def extract_metadata(
        self, fileset: Dict[str, Dict[str, str]]
    ) -> List[WorkItem]:
        """
        Extracts metadata from the given fileset using coffea.preprocess.

        Parameters
        ----------
        fileset : Dict[str, Dict[str, str]]
            A coffea-compatible fileset mapping dataset names to file paths and tree names.

        Returns
        -------
        List[WorkItem]
            A list of `coffea.processor.WorkItem` objects with extracted metadata.
        """
        logger.info("Extracting metadata using coffea.dataset_tools.preprocess")
        try:
            # Run the coffea preprocess function on the provided fileset
            workitems = self.runner.preprocess(fileset)
            # Convert the generator returned by preprocess to a list of WorkItems
            return list(workitems)
        except Exception as e:
            # Log any errors encountered during preprocessing
            logger.error(f"Error during coffea preprocessing: {e}")
            # Return an empty list to indicate failure or no metadata extracted
            return []


class NanoAODMetadataGenerator:
    """
    Orchestrates the generation, reading, and summarization of NanoAOD metadata.

    This class combines `FilesetBuilder` and `CoffeaMetadataExtractor` to provide
    a complete metadata management workflow. It can either generate new metadata
    or read existing metadata from disk, storing the results as instance
    attributes for easy access.

    Attributes
    ----------
    dataset_manager : ConfigurableDatasetManager
        Manages dataset configurations and output directories.
    output_directory : Path
        The base directory for all metadata JSON files.
    fileset : Optional[Dict[str, Dict[str, Any]]]
        The generated or read coffea-compatible fileset.
    workitems : Optional[List[WorkItem]]
        The generated or read list of `WorkItem` objects.
    nanoaods_summary : Optional[Dict[str, Dict[str, Any]]]
        The generated or read summarized NanoAOD metadata.
    """

    def __init__(
        self,
        dataset_manager: ConfigurableDatasetManager,
        output_manager
    ):
        """
        Initializes the NanoAODMetadataGenerator.

        Parameters
        ----------
        dataset_manager : ConfigurableDatasetManager
            A dataset manager instance (required).
        output_manager : OutputDirectoryManager
            Centralized output directory manager (required).
        """
        self.dataset_manager = dataset_manager
        self.output_manager = output_manager

        # Use output manager to get metadata directory
        self.output_directory = self.output_manager.get_metadata_dir()

        # Initialize modularized components for fileset building and metadata extraction
        self.fileset_builder = FilesetBuilder(self.dataset_manager)
        self.metadata_extractor = CoffeaMetadataExtractor()

        # Attributes to store generated/read metadata.
        # These will be populated by the run() method.
        self.fileset: Optional[Dict[str, Dict[str, Any]]] = None
        self.workitems: Optional[List[WorkItem]] = None
        self.nanoaods_summary: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_metadata_paths(self) -> Dict[str, Path]:
        """
        Generates and returns the full paths for all metadata JSON files.

        These paths are consistently derived from the `self.output_directory`
        attribute, which is set from the output manager's metadata directory.
        This ensures all read/write operations target the same locations.

        Returns
        -------
        Dict[str, Path]
            A dictionary containing the paths for:
            - 'fileset_path': Path to the fileset JSON (e.g., fileset.json).
            - 'workitems_path': Path to the WorkItems JSON (e.g., workitems.json).
            - 'nanoaods_summary_path': Path to the main NanoAODs summary JSON (e.g., nanoaods.json).
            - 'process_summary_dir': Path to the directory where per-process JSONs are saved.
        """
        # Get the base output directory from the instance attribute.
        # This directory is created during __init__.
        output_dir = self.output_directory

        # Construct and return the full paths for each metadata file
        return {
            "fileset_path": output_dir / "fileset.json",
            "workitems_path": output_dir / "workitems.json",
            "nanoaods_summary_path": output_dir / "nanoaods.json",
            "process_summary_dir": output_dir, # Per-process files are saved directly in this directory
        }

    def run(
        self,
        identifiers: Optional[Union[int, List[int]]] = None,
        generate_metadata: bool = True
    ) -> None:
        """
        Generates or reads all metadata.

        This is the main orchestration method. If `generate_metadata` is True, it
        performs a full generation workflow. Otherwise, it attempts to read
        existing metadata from the expected paths.

        Parameters
        ----------
        identifiers : Optional[Union[int, List[int]]], optional
            Specific listing file IDs to process. Only used if `generate_metadata` is True.
        generate_metadata : bool, optional
            If True, generate new metadata. If False, read existing metadata.
            Defaults to True.

        Raises
        ------
        SystemExit
            If `generate_metadata` is False and any required metadata file is not found.
        """
        if generate_metadata:
            logger.info("Starting metadata generation workflow...")
            # Step 1: Build and save the fileset
            self.fileset = self.fileset_builder.build_fileset(identifiers)
            self.fileset_builder.save_fileset(self.fileset)

            # Step 2: Extract and save WorkItem metadata
            self.workitems = self.metadata_extractor.extract_metadata(self.fileset)
            self.write_metadata()

            # Step 3: Summarize and save NanoAODs metadata
            self.summarise_nanoaods()
            self.write_nanoaods_summary()
            logger.info("Metadata generation complete.")
        else:
            logger.info(f"Skipping metadata generation - using existing metadata from \n %s",
                        pretty_repr(self._get_metadata_paths()))
            try:
                self.read_fileset()
                self.read_metadata()
                self.read_nanoaods_summary()
                logger.info("All metadata successfully loaded from disk.")
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load existing metadata: {e}")
                logger.error("Please ensure metadata files exist or enable generation.")
                sys.exit(1)


    def write_nanoaods_summary(self) -> None:
        """
        Writes the summarized NanoAOD metadata to JSON files.

        This method writes individual JSON files for each process/variation and a
        master `nanoaods.json` file.

        Raises
        ------
        ValueError
            If `self.nanoaods_summary` has not been populated.
        """
        # Check if the summary data is available
        if self.nanoaods_summary is None:
            raise ValueError("NanoAODs summary is not available to write. Please generate or load it first.")

        # Get all necessary output paths from the helper method
        paths = self._get_metadata_paths()
        process_summary_dir = paths["process_summary_dir"]
        nanoaods_summary_path = paths["nanoaods_summary_path"]

        # Write per-process JSON files for detailed breakdown
        for process_name, variations in self.nanoaods_summary.items():
            for variation_label, data in variations.items():
                # Construct filename for per-process summary
                per_process_summary_path = (
                    process_summary_dir
                    / f"nanoaods_{process_name}_{variation_label}.json"
                )
                # Ensure the directory for the output file exists
                per_process_summary_path.parent.mkdir(parents=True, exist_ok=True)

                # Write the specific process/variation data to its JSON file
                with per_process_summary_path.open("w") as f:
                    json.dump(
                        {process_name: {variation_label: data}}, # Wrap in a dict for consistent structure
                        f,
                        indent=4,
                    )
                logger.debug(f"Wrote NanoAODs summary file: {per_process_summary_path}")

        # Write the master metadata index file containing the full aggregated summary
        # This file is the primary input for analysis fileset construction
        with nanoaods_summary_path.open("w") as f:
            json.dump(self.nanoaods_summary, f, indent=4)
        logger.info(f"NanoAODs summary written to {nanoaods_summary_path}")

    def summarise_nanoaods(self) -> None:
        """
        Summarizes the extracted `WorkItem` metadata into a structured NanoAODs summary.

        This method processes `self.workitems` to aggregate event counts per
        file, process, and variation, storing the result in `self.nanoaods_summary`
        with the schema:
        `{process_name: {variation_label: {"files": [...], "nevts_total": int}}}`.

        Raises
        ------
        ValueError
            If `self.workitems` has not been populated.
        """
        # Ensure sample chunks are available for summarization
        if self.workitems is None:
            raise ValueError("Sample chunks (WorkItems) are not available to summarize. Please extract or load them first.")

        # Use self.sample_chunks directly as the source of WorkItems
        workitems = self.workitems

        # Initialize a nested defaultdict to store aggregated event counts:
        # structure: process -> variation -> filename -> event count
        counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Iterate through each WorkItem to extract relevant information
        for wi in workitems:
            # Convert WorkItem dataclass to a dictionary for easier access
            wi_dict = dataclasses.asdict(wi)

            dataset = wi_dict["dataset"] # type: ignore
            filename = wi_dict["filename"]
            # Extract entry start and stop, default to 0 if not present
            start = int(wi_dict.get("entrystart", 0))
            stop = int(wi_dict.get("entrystop", 0))

            # Calculate number of events in this chunk, ensuring it's non-negative
            nevts = max(0, stop - start)

            # Parse the dataset key to get process and variation names
            proc, var = _parse_dataset(dataset)
            logger.debug(f"Processing WorkItem: {proc}, {var}, {filename}, {nevts} events")

            # Aggregate event counts for the specific process, variation, and filename
            counts[proc][var][filename] += nevts

        # Build the final output schema (self.nanoaods_summary)
        out: Dict[str, Dict[str, Any]] = {}
        for proc, per_var in counts.items():
            out[proc] = {}
            for var, per_file in per_var.items():
                # Create a list of files with their event counts, sorted by path for reproducibility
                files_list = [ # type: ignore
                    {"path": str(path), "nevts": nevts}
                    for path, nevts in sorted(per_file.items())
                ] # type: ignore
                # Calculate the total number of events for this process and variation
                nevts_total = sum(f["nevts"] for f in files_list) # type: ignore

                # Store the aggregated data in the output dictionary
                out[proc][var] = {
                    "files": files_list,
                    "nevts_total": int(nevts_total), # Ensure total events is an integer
                }
        # Assign the generated summary to the instance attribute
        self.nanoaods_summary = out
        logger.info("NanoAODs summary generated.")

    def read_fileset(self) -> None:
        """
        Reads the fileset from `fileset.json` and stores it.

        Raises
        ------
        FileNotFoundError
            If the `fileset.json` file does not exist at the expected path.
        """
        # Get the canonical path for the fileset JSON file
        paths = self._get_metadata_paths()
        fileset_path = paths["fileset_path"]

        logger.info(f"Attempting to read fileset from {fileset_path}")
        try:
            # Open and load the JSON file
            with fileset_path.open("r") as f:
                # If max_files is set in dataset_manager, we might want to filter the fileset here
                self.fileset = json.load(f)
                if (max_files := self.dataset_manager.config.max_files):
                    for dataset, data in self.fileset.items():
                        files = list(data["files"].items())[:max_files]
                        self.fileset[dataset]["files"] = dict(files)

            logger.info("Fileset successfully loaded.")
        except FileNotFoundError as e:
            # Log error and re-raise if file is not found
            logger.error(f"Fileset JSON not found at {fileset_path}. {e}")
            raise
        except json.JSONDecodeError as e:
            # Log error and re-raise if JSON decoding fails
            logger.error(f"Error decoding fileset JSON from {fileset_path}. {e}")
            raise
        except KeyError as e:
            # Log error and re-raise if expected keys are missing (less common for fileset)
            logger.error(f"Missing expected key in fileset JSON from {fileset_path}. {e}")
            raise



    def read_metadata(self) -> None:
        """
        Reads `WorkItem` metadata from `workitems.json` and stores it.

        This method deserializes `WorkItem` objects, decoding the base64-encoded
        `fileuuid` field back to its binary format.

        Raises
        ------
        FileNotFoundError
            If the `workitems.json` file does not exist.
        """
        # Get the canonical path for the workitems JSON file
        paths = self._get_metadata_paths()
        workitems_path = paths["workitems_path"]

        # Load JSON data from file
        logger.info(f"Attempting to read WorkItems metadata from {workitems_path}")
        try:
            with workitems_path.open("r") as f:
                workitems_data = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"WorkItems JSON not found at {workitems_path}. {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding WorkItems JSON from {workitems_path}. {e}")
            raise

        # Reconstruct WorkItem objects from dictionaries
        reconstructed_items = []
        for i, item_dict in enumerate(workitems_data):
            try:
                # Decode base64-encoded file UUID back to binary format
                # This reverses the encoding done in write_metadata()
                item_dict["fileuuid"] = base64.b64decode(item_dict["fileuuid"])

                # Reconstruct WorkItem object from dictionary
                # WorkItem is a dataclass that represents file metadata in coffea
                work_item = WorkItem(**item_dict)
                reconstructed_items.append(work_item)
            except KeyError as e:
                logger.error(f"Missing expected key '{e}' in WorkItem entry {i} from {workitems_path}.")
                raise
            except Exception as e:
                logger.error(f"Error reconstructing WorkItem entry {i} from {workitems_path}: {e}")
                raise

        # Assign the reconstructed WorkItems to the instance attribute
        self.workitems = reconstructed_items
        logger.info("WorkItems metadata successfully loaded.")

    def read_nanoaods_summary(self) -> None:
        """
        Reads the NanoAODs summary from `nanoaods.json` and stores it.

        Raises
        ------
        FileNotFoundError
            If the `nanoaods.json` file does not exist.
        """
        # Get the canonical path for the nanoaods summary JSON file
        paths = self._get_metadata_paths()
        nanoaods_summary_path = paths["nanoaods_summary_path"]

        logger.info(f"Attempting to read NanoAODs summary from {nanoaods_summary_path}")
        try:
            # Open and load the JSON file
            with nanoaods_summary_path.open("r") as f:
                self.nanoaods_summary = json.load(f)
            logger.info("NanoAODs summary successfully loaded.")
        except FileNotFoundError as e:
            logger.error(f"NanoAODs summary JSON not found at {nanoaods_summary_path}. {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding NanoAODs summary JSON from {nanoaods_summary_path}. {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing expected key in NanoAODs summary JSON from {nanoaods_summary_path}. {e}")
            raise

    def write_metadata(self) -> None:
        """
        Writes the `WorkItem` metadata to `workitems.json`.

        It serializes the `coffea.processor.WorkItem` objects to a JSON file,
        base64-encoding the binary `fileuuid` field for JSON compatibility.

        Raises
        ------
        ValueError
            If `self.workitems` has not been populated.
        """
        # Ensure sample chunks are available for writing
        if self.workitems is None:
            raise ValueError("Sample chunks (WorkItems) are not available to write. Please extract or load them first.")

        # Get the canonical path for the workitems JSON file
        paths = self._get_metadata_paths()
        workitems_path = paths["workitems_path"]

        # Ensure the parent directory exists
        workitems_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert WorkItem objects to serializable dictionaries
        serializable = []
        for workitem in self.workitems:
            # Convert dataclass to a dictionary
            workitem_dict = dataclasses.asdict(workitem)

            # Encode binary file UUID as base64 string for JSON compatibility
            # This is necessary because JSON cannot handle raw bytes
            workitem_dict["fileuuid"] = base64.b64encode(workitem_dict["fileuuid"]).decode("ascii")

            serializable.append(workitem_dict) # type: ignore

        # Write serialized metadata to JSON file with pretty-printing
        with workitems_path.open("w") as f:
            json.dump(serializable, f, indent=4)

        logger.info(f"WorkItems metadata saved to {workitems_path}")