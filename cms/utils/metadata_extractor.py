"""
NanoAOD dataset metadata extraction and management.

Three-stage workflow for preparing NanoAOD datasets before analysis:

1. **Fileset Building**: Reads ROOT file paths from .txt listings, creates
   coffea-compatible fileset dicts, and generates Dataset objects.
   Output: fileset.json

2. **Metadata Extraction**: Uses coffea preprocessing to chunk files into
   WorkItems (file segments for parallel processing with entry ranges).
   Output: workitems.json

3. **Event Count Aggregation**: Sums total events per process/variation for
   MC normalization. Counts pre-skimming NanoAOD statistics.
   Output: nanoaods.json

Key Classes:
    FilesetBuilder: Constructs filesets from dataset configs
    CoffeaMetadataExtractor: Extracts WorkItems via coffea preprocessing
    NanoAODMetadataGenerator: Orchestrates full workflow

Dataset Key Format: "process__variation" (MC) or "process" (data)
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
from utils.datasets import ConfigurableDatasetManager, Dataset


# Configure module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Dataset key format constants
DATASET_DELIMITER = "__"  # Delimiter for "process__variation" keys in fileset
DEFAULT_VARIATION = "nominal"  # Default variation when none specified


def parse_dataset_key(dataset_key: str) -> Tuple[str, str]:
    """
    Parse a dataset key string into process and variation components.

    Dataset keys encode both the physics process and systematic variation in a
    single string using the DATASET_DELIMITER ("__"). This function splits them
    for separate access. If no delimiter is found, assumes nominal variation.

    Parameters
    ----------
    dataset_key : str
        Dataset key in format "process__variation" (e.g., "ttbar__nominal")
        or just "process" for data without variations.

    Returns
    -------
    Tuple[str, str]
        (process_name, variation_name) where variation defaults to "nominal"
        if not explicitly specified in the key.

    Examples
    --------
    >>> parse_dataset_key("signal__nominal")
    ("signal", "nominal")
    >>> parse_dataset_key("data")
    ("data", "nominal")
    """
    if DATASET_DELIMITER in dataset_key:
        proc, var = dataset_key.split(DATASET_DELIMITER, 1)
    else:
        proc, var = dataset_key, DEFAULT_VARIATION
    return proc, var


def get_root_file_paths(
    directory: Union[str, Path],
    identifiers: Optional[Union[int, List[int]]] = None,
    redirector: str = None,
) -> List[str]:
    """
    Read ROOT file paths from .txt listing files.

    Reads .txt files where each line contains one ROOT file path. This approach
    separates file lists from code, enabling version control and easy updates.
    The `identifiers` parameter allows processing subsets for testing.
    The `redirector` parameter prepends protocol prefixes for remote access.

    Parameters
    ----------
    directory : str or Path
        Directory containing .txt listing files
    identifiers : int or list of ints, optional
        Process only specific listing files by ID (e.g., [0, 1] reads 0.txt, 1.txt).
        If None, reads all .txt files in directory.
    redirector : str, optional
        URL prefix to prepend to paths (e.g., "root://xrootd.server.com//").
        If None, paths used as-is.

    Returns
    -------
    List[str]
        ROOT file paths with optional redirector prefix applied.

    Raises
    ------
    FileNotFoundError
        If directory contains no .txt files or specified identifier file missing.
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
                if redirector:
                    path_str = f"{redirector}{path_str}"
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

    def __init__(self, dataset_manager: ConfigurableDatasetManager, output_manager) -> None:
        """
        Initializes the FilesetBuilder.

        Parameters
        ----------
        dataset_manager : ConfigurableDatasetManager
            A dataset manager instance (required).
        """
        self.dataset_manager = dataset_manager
        self.output_manager = output_manager

    def build_fileset(
        self, identifiers: Optional[Union[int, List[int]]] = None,
        processes_filter: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dataset]]:
        """
        Build coffea-compatible fileset and Dataset objects from configurations.

        Iterates through configured processes, collects ROOT file paths from listing
        files, and constructs both a fileset dict for coffea preprocessing and Dataset
        objects for the analysis pipeline. Handles multi-directory datasets by creating
        separate fileset entries with index suffixes (e.g., "signal_0__nominal").

        Dataset key format:
        - MC: "process__variation" or "process_N__variation" for multi-directory
        - Data: "process" or "process_N" for multi-directory

        Parameters
        ----------
        identifiers : Optional[Union[int, List[int]]], optional
            Specific listing file IDs to process. If None, uses all .txt files.
        processes_filter : Optional[List[str]], optional
            Only build fileset for these processes. If None, builds all.

        Returns
        -------
        Tuple[Dict[str, Dict[str, Any]], List[Dataset]]
            (fileset_dict, datasets_list) where fileset_dict maps dataset keys to
            {"files": {path: treename}, "metadata": {...}} and datasets_list contains
            Dataset objects with cross-sections and process metadata.
        """
        fileset: Dict[str, Dict[str, Any]] = {}
        datasets: List[Dataset] = []

        max_files = self.dataset_manager.config.max_files

        if max_files and max_files <= 0:
            raise ValueError("max_files must be None or a positive integer.")

        # Iterate over each process configured in the dataset manager
        for process_name in self.dataset_manager.list_processes():
            # Check if processes filter is configured
            if processes_filter and process_name not in processes_filter:
                logger.info(f"Skipping {process_name} (not in processes filter)")
                continue

            logger.info(f"Building fileset for process: {process_name}")

            # Get the directory where listing files are located for this process
            listing_dirs = self.dataset_manager.get_dataset_directories(process_name)
            # Get the tree name (e.g., "Events") for ROOT files of this process
            tree_name = self.dataset_manager.get_tree_name(process_name)

            # Validate that we have matching numbers of directories and cross-sections
            if len(listing_dirs) != len(cross_sections):
                logger.error(f"Mismatch between number of directories ({len(listing_dirs)}) and cross-sections ({len(cross_sections)}) for process {process_name}")
                continue

            try:
                redirector  = self.dataset_manager.get_redirector(process_name)
                # Collect all ROOT file paths from the listing files
                file_paths = get_root_file_paths(listing_dirs, identifiers, redirector)[:max_files]

                # Collect fileset keys for this Dataset object
                fileset_keys = []
                variation_label = "nominal"

                # Process each directory-cross-section pair
                # Always create separate entries when multiple directories exist
                is_data = self.dataset_manager.is_data_dataset(process_name)

                for idx, (listing_dir, xsec) in enumerate(zip(listing_dirs, cross_sections)):
                    # Collect ROOT file paths from this directory
                    file_paths = get_root_file_paths(listing_dir, identifiers, redirector)[:max_files]

                    # Construct dataset key for coffea fileset
                    # Multi-directory datasets (e.g., different run periods) get index suffix
                    # to create distinct keys: signal_0__nominal, signal_1__nominal
                    if not is_data:
                        # If multiple directories, append index to distinguish them
                        if len(listing_dirs) > 1:
                            dataset_key = f"{process_name}_{idx}__{variation_label}"
                        else:
                            dataset_key = f"{process_name}__{variation_label}"
                    else:
                        # For data, append index if multiple directories
                        if len(listing_dirs) > 1:
                            dataset_key = f"{process_name}_{idx}"
                        else:
                            dataset_key = process_name

                    # Add to fileset keys list for Dataset object
                    fileset_keys.append(dataset_key)

                    # Create the fileset entry: map each file path to its tree name
                    fileset[dataset_key] = {
                        "files": {file_path: tree_name for file_path in file_paths},
                        "metadata": {
                            "process": process_name,
                            "variation": variation_label,
                            "xsec": xsec,
                            "is_data": is_data,
                        }
                    }

                    logger.debug(f"Added {len(file_paths)} files for {dataset_key} with xsec={xsec}")

                # Create Dataset object for this process
                dataset = Dataset(
                    name=process_name,
                    fileset_keys=fileset_keys,
                    process=process_name,
                    variation=variation_label,
                    cross_sections=cross_sections,
                    is_data=is_data,
                    events=None  # Will be populated during skimming
                )
                datasets.append(dataset)
                logger.debug(f"Created Dataset: {dataset}")

            except FileNotFoundError as fnf:
                # Log an error if listing files are not found and continue to next process
                logger.error(f"Could not build fileset for {process_name}: {fnf}")
                continue

        return fileset, datasets

    def save_fileset(
        self, fileset: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Saves the built fileset to a JSON file.

        The output path is determined by the `self.output_manager.get_metadata_dir()`
        configured in the `output_manager`.

        Parameters
        ----------
        fileset : Dict[str, Dict[str, Any]]
            The fileset mapping to save.
        """
        # Construct the full path for the fileset JSON file
        output_dir = Path(self.output_manager.get_metadata_dir())
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

    def __init__(self, dask=(None, None)) -> None:
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
        if not dask[0]:
            self.runner = processor.Runner(
                executor=processor.FuturesExecutor(),
                schema=NanoAODSchema,
                savemetrics=True,
                # Use a small chunksize for demonstration/testing to simulate multiple chunks
                chunksize=100_000,
            )
        else:
            print("Running dask")
            self.runner = processor.Runner(
                executor=processor.DaskExecutor(client=dask[0]),
                schema=NanoAODSchema,
                savemetrics=True,
                # Use a small chunksize for demonstration/testing to simulate multiple chunks
                chunksize=100_000,
            )
            

    def extract_metadata(
        self, fileset: Dict[str, Dict[str, str]],
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

        Returns
        -------
        List[WorkItem]
            WorkItem objects with file metadata and entry ranges for chunked processing.
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
    datasets : Optional[List[Dataset]]
        The generated Dataset objects for the analysis pipeline.
    workitems : Optional[List[WorkItem]]
        The generated or read list of `WorkItem` objects.
    nanoaods_summary : Optional[Dict[str, Dict[str, Any]]]
        The generated or read summarized NanoAOD metadata.
    """

    def __init__(
        self,
        dataset_manager: ConfigurableDatasetManager,
        output_manager,
        dask=(None,None)
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
        self.fileset_builder = FilesetBuilder(self.dataset_manager, self.output_manager)
        print(dask)
        self.metadata_extractor = CoffeaMetadataExtractor(dask=dask)

        # Attributes to store generated/read metadata.
        # These will be populated by the run() method.
        self.fileset: Optional[Dict[str, Dict[str, Any]]] = None
        self.datasets: Optional[List[Dataset]] = None
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
        generate_metadata: bool = True,
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
        processes_filter : Optional[List[str]], optional
            If provided, only generate metadata for processes in this list.

        Raises
        ------
        SystemExit
            If `generate_metadata` is False and any required metadata file is not found.
        """
        if generate_metadata:
            logger.info("Starting metadata generation workflow...")
            # Step 1: Build and save the fileset and Dataset objects
            self.fileset, self.datasets = self.fileset_builder.build_fileset(identifiers, processes_filter)
            self.fileset_builder.save_fileset(self.fileset)

            # Step 2: Extract and save WorkItem metadata
            self.workitems = self.metadata_extractor.extract_metadata(self.fileset)
            self.write_metadata()

            # Step 3: Aggregate event counts and save summary
            self.summarize_event_counts()
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

    def summarize_event_counts(self) -> None:
        """
        Aggregate event counts from WorkItems into a process/variation summary.

        Processes extracted WorkItems to count total events per file, process, and
        systematic variation. These counts represent pre-skimming NanoAOD statistics
        and are used for MC normalization in the analysis phase. Each file's events
        are summed across all its WorkItem chunks (since coffea splits large files).

        The aggregated summary is stored in `self.nanoaods_summary` with schema:
        ```python
        {
            "process": {
                "variation": {
                    "files": [{"path": str, "nevts": int}, ...],
                    "nevts_total": int
                }
            }
        }
        ```

        Raises
        ------
        ValueError
            If `self.workitems` has not been populated via extraction or loading.
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
            proc, var = parse_dataset_key(dataset)
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
        Reads the fileset from `fileset.json` and reconstructs Dataset objects.

        This method reads the fileset and reconstructs the Dataset objects by:
        1. Loading the fileset JSON
        2. Grouping fileset keys by process name from metadata
        3. Creating Dataset objects with the keep_split flag from config

        Raises
        ------
        FileNotFoundError
            If the `fileset.json` file does not exist at the expected path.
        ValueError
            If required metadata fields are missing or invalid.
        KeyError
            If process is not found in dataset configuration.
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

            # Reconstruct Dataset objects from fileset
            self._reconstruct_datasets_from_fileset()

        except FileNotFoundError as e:
            # Log error and re-raise if file is not found
            logger.error(f"Fileset JSON not found at {fileset_path}. {e}")
            raise
        except json.JSONDecodeError as e:
            # Log error and re-raise if JSON decoding fails
            logger.error(f"Error decoding fileset JSON from {fileset_path}. {e}")
            raise
        except KeyError as e:
            # Log error and re-raise if expected keys are missing
            logger.error(f"Missing expected key in fileset JSON from {fileset_path}. {e}")
            raise

    def _reconstruct_datasets_from_fileset(self) -> None:
        """
        Reconstructs Dataset objects from the loaded fileset.

        Groups fileset keys by process name and creates Dataset objects
        with metadata from the fileset and keep_split flag from config.

        Raises
        ------
        ValueError
            If required metadata fields (process, variation, xsec) are missing.
        KeyError
            If process is not found in dataset configuration.
        """
        from collections import defaultdict

        if not self.fileset:
            raise ValueError("No fileset loaded - cannot reconstruct datasets")

        # Group fileset keys by process name
        process_groups = defaultdict(list)

        for fileset_key, fileset_data in self.fileset.items():
            # Require metadata to exist
            if "metadata" not in fileset_data:
                raise ValueError(f"Fileset key '{fileset_key}' is missing 'metadata' field")

            metadata = fileset_data["metadata"]

            # Require process field to exist
            if "process" not in metadata:
                raise ValueError(f"Fileset key '{fileset_key}' is missing 'metadata.process' field")

            process = metadata["process"]
            process_groups[process].append(fileset_key)

        # Create Dataset objects
        self.datasets = []
        for process, fileset_keys in process_groups.items():
            # Get metadata from first fileset entry
            first_key = fileset_keys[0]
            first_metadata = self.fileset[first_key]["metadata"]

            # Require variation field
            if "variation" not in first_metadata:
                raise ValueError(f"Fileset key '{first_key}' is missing 'metadata.variation' field")
            variation = first_metadata["variation"]

            # Collect cross-sections for each fileset key
            cross_sections = []
            for key in fileset_keys:
                metadata = self.fileset[key]["metadata"]
                # Require xsec field
                if "xsec" not in metadata:
                    raise ValueError(f"Fileset key '{key}' is missing 'metadata.xsec' field")
                cross_sections.append(metadata["xsec"])

            # Determine is_data flag
            is_data = first_metadata.get("is_data")
            if is_data is None:
                try:
                    is_data = self.dataset_manager.is_data_dataset(process)
                except KeyError:
                    logger.warning(
                        f"Process '{process}' not found in dataset manager when reconstructing datasets"
                    )
                    is_data = False

            dataset = Dataset(
                name=process,
                fileset_keys=fileset_keys,
                process=process,
                variation=variation,
                cross_sections=cross_sections,
                is_data=is_data,
                events=None  # Will be populated during skimming
            )
            self.datasets.append(dataset)
            logger.debug(f"Reconstructed Dataset: {dataset}")

        logger.info(f"Reconstructed {len(self.datasets)} Dataset objects from fileset")



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
            # WorkItem.fileuuid is binary (bytes), which JSON cannot serialize.
            # Base64 encoding converts it to ASCII string for JSON storage.
            # Decoded back to binary when reading (see read_metadata).
            workitem_dict["fileuuid"] = base64.b64encode(workitem_dict["fileuuid"]).decode("ascii")

            serializable.append(workitem_dict) # type: ignore

        # Write serialized metadata to JSON file with pretty-printing
        with workitems_path.open("w") as f:
            json.dump(serializable, f, indent=4)

        logger.info(f"WorkItems metadata saved to {workitems_path}")
