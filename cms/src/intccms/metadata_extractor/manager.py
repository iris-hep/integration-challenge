"""High-level workflow coordination for metadata extraction.

This module provides the DatasetMetadataManager class that orchestrates the complete
metadata generation workflow, composing FilesetBuilder, CoffeaMetadataExtractor,
and core functions.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

from servicex import ProgressBarFormat
from servicex import query, dataset, deliver
import os
import yaml
import copy
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.pretty import pretty_repr
from coffea.processor.executor import WorkItem
from coffea.nanoevents import NanoAODSchema


from intccms.datasets import DatasetManager, Dataset
from intccms.metadata_extractor.builders import FilesetBuilder
from intccms.metadata_extractor.extractor import CoffeaMetadataExtractor
from intccms.metadata_extractor.core import (
    aggregate_workitem_events,
    format_event_summary,
    extract_nevts_from_summary,
)
from intccms.metadata_extractor.io import (
    save_json,
    load_json,
    serialize_workitems,
    deserialize_workitems,
)

logger = logging.getLogger(__name__)


# Type definitions for metadata structures
class MetadataEntry(TypedDict):
    """Metadata for a single dataset/process/variation.

    This structure contains all information needed for physics analysis normalization
    and event processing.
    """
    process: str  # Process name (e.g., "signal", "ttbar_semilep")
    variation: str  # Systematic variation (e.g., "nominal", "JES_up")
    xsec: float  # Cross-section in picobarns
    nevts: int  # Total number of events
    is_data: bool  # True if real data, False if Monte Carlo
    lumi_mask_config: Optional[object]  # FunctorConfig for luminosity mask, or None
    dataset: str  # Dataset key/identifier
    year: Optional[str]  # Correction year (e.g., "2016preVFP", "2017", "2018"), or None


# Type alias for the metadata lookup dictionary
MetadataLookup = Dict[str, MetadataEntry]


class DatasetMetadataManager:
    """
    Orchestrates dataset metadata generation and management workflow.

    This class combines FilesetBuilder and CoffeaMetadataExtractor to provide
    a complete metadata management workflow. It can either generate new metadata
    or read existing metadata from disk.

    Attributes
    ----------
    dataset_manager : DatasetManager
        Manages dataset configurations
    output_manager : OutputDirectoryManager
        Manages output directory paths
    output_directory : Path
        Base directory for all metadata JSON files
    fileset : Dict[str, Dict[str, Any]], optional
        Generated or loaded coffea-compatible fileset
    datasets : List[Dataset], optional
        Generated or loaded Dataset objects
    workitems : List[WorkItem], optional
        Generated or loaded WorkItem objects
    nanoaods_summary : Dict[str, Dict[str, Any]], optional
        Generated or loaded event count summary
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        output_manager: Any,
        config: Optional[Any] = None,
    ):
        """
        Initialize DatasetMetadataManager.

        Parameters
        ----------
        dataset_manager : DatasetManager
            Dataset manager instance
        output_manager : OutputDirectoryManager
            Output directory manager
        config : Config, optional
            Configuration object. Used to extract run_metadata_generation,
            processes filter, and chunksize settings.
        """
        self.dataset_manager = dataset_manager
        self.output_manager = output_manager
        self.output_directory = self.output_manager.metadata_dir
        self.config = config

        # Extract config-derived attributes
        if config:
            self.generate_metadata = config.general.run_metadata_generation
            self.processes_filter = getattr(config.general, 'processes', None)
            self.servicex_preskim = config.general.run_servicex_preskim #Prayag:WIP
            # Extract chunksize from config
            if hasattr(config, 'preprocess') and hasattr(config.preprocess, 'skimming'):
                self.chunksize = config.preprocess.skimming.chunk_size
            else:
                self.chunksize = 100_000
        else:
            self.generate_metadata = True
            self.servicex_preskim = False #Prayag:WIP
            self.processes_filter = None
            self.chunksize = 100_000

        # Initialize fileset builder (doesn't need executor)
        self.fileset_builder = FilesetBuilder(dataset_manager, output_manager)

        # Attributes to store generated/read metadata
        self.fileset: Optional[Dict[str, Dict[str, Any]]] = None
        self.datasets: Optional[List[Dataset]] = None
        self.workitems: Optional[List[WorkItem]] = None
        self.nanoaods_summary: Optional[Dict[str, Dict[str, Any]]] = None

        logger.info(f"Initialized DatasetMetadataManager with output dir: {self.output_directory}")

    def _get_metadata_paths(self) -> Dict[str, Path]:
        """
        Get paths for all metadata JSON files.

        Returns
        -------
        Dict[str, Path]
            Dictionary with keys: fileset_path, workitems_path, nanoaods_summary_path
        """
        output_dir = self.output_directory
        return {
            "fileset_path": output_dir / "fileset.json",
            "workitems_path": output_dir / "workitems.json",
            "nanoaods_summary_path": output_dir / "nanoaods.json",
        }

    def run(
        self,
        executor: Any = None,
        schema: Any = None,
        identifiers: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Generate or load metadata based on config settings.

        Uses `config.general.run_metadata_generation` to decide whether to generate
        new metadata or load existing files.

        Parameters
        ----------
        executor : coffea executor, optional
            Required when generating metadata (e.g., DaskExecutor, FuturesExecutor).
            Not needed when loading existing metadata.
        schema : coffea schema, optional
            Schema for parsing ROOT files. Defaults to NanoAODSchema.
        identifiers : int or list of ints, optional
            Specific listing file IDs to process. Only used when generating metadata.

        Raises
        ------
        ValueError
            If generating metadata but no executor provided
        SystemExit
            If loading metadata and required files are missing
        """
        # logger.info(f"Prayag WIP: self.service_preskim : {self.servicex_preskim}")
        if self.servicex_preskim: #Prayag:WIP
            
            self._generate_servicex_preskim(identifiers) #Prayag:WIP
        if self.generate_metadata:
            if executor is None:
                raise ValueError(
                    "executor is required when run_metadata_generation=True. "
                    "Pass a DaskExecutor or FuturesExecutor."
                )
            self._generate_metadata(executor, schema, identifiers)
        else:
            self._load_existing_metadata()

    def _servicex_key(self, path):#Prayag:WIP
        return path.replace("://", "___").replace("/", "_")
        
    def _reorder_servicex_output(self, og_filelist, servicex_files):#Prayag:WIP
        '''Service applies sort on input list : this function ondoes that'''
    
        if len(servicex_files) < 2:
            return servicex_files
        
        # map encoded original path -> servicex file(s)
        sx_map = defaultdict(deque)
        for f in servicex_files:
            encoded = f.split("/")[-1]
            sx_map[encoded].append(f)
        
        ordered_servicex = []
        for f in og_filelist:
            key = self._servicex_key(f)
            if key in sx_map:
                ordered_servicex.append(sx_map[key].popleft())
            else:
                ordered_servicex.append(None)
        return ordered_servicex

    def _servicex_query(self, treename, cuts, branch_list):
        '''Query Builder'''
        the_query = [
         {
          'treename': treename,
          'filter_name': branch_list,
          'cut': cuts
         }
        ]
        return the_query

    def _filter_these_branches(self, config, is_data):
        '''Using the preprocess config construct a list of branches to be filtered'''
        branches = copy.copy(config["branches"])
        mc_branches = copy.copy(config["mc_branches"])
    
        if is_data :
            # Remove MC-specific branches
            out_branches = copy.copy(branches)
            for key, mc_values in mc_branches.items():
                out_branches[key] = list(set(branches[key]) - set(mc_values))
        else:
            out_branches = branches
            
        # Optional: remove keys that become empty
        need_these_branches = {k: v for k, v in out_branches.items() if len(v) > 0}
        expanded_list = []
        for branch, subbranch_list in need_these_branches.items():
            for subbranch in subbranch_list:
                if branch == 'event':
                    to_append = subbranch
                else:
                    to_append = branch+'_'+subbranch
                expanded_list.append(to_append)
        
        return expanded_list
        
    def _add_tree_info_to_files(self, fileset, treenames):
        output = copy.copy(fileset)
        for name in fileset.keys():
            tree = treenames[name]
            output[name]["files"] = {file:tree for file in fileset[name]["files"]}
        return output
        
    def _batch_deliver(self, fileset, treenames, is_data_list, servicex_config, preprocess_config):#Prayag:WIP
        '''Deliver each dataset by concurrently submitting to servicex'''

        def process_key(key, fileset, treename, cuts, config, is_data):
            files_to_process = list(fileset[key]["files"].keys())

            expanded_branch_list = self._filter_these_branches(config, is_data)
            
            spec = {
                "Sample": [
                    {
                        "Name": key,
                        "Dataset": dataset.FileList(files_to_process),
                        "Query": query.UprootRaw(self._servicex_query(treename, cuts, expanded_branch_list)),
                    }
                ]
            }
        
            result = self._reorder_servicex_output(
                files_to_process,
                list(deliver(spec)[key])
            )
        
            return key, result

        
        output = copy.deepcopy(fileset)
        max_workers = servicex_config.get("futures_max_workers",8)
        cuts = servicex_config.get("selections", None)
        cut_string = f"(({') & ('.join(cuts)}))"
        if (not isinstance(cuts, list)) or (len(cuts) == 0):
            raise ValueError("No Pre-Skim selections provided")
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_key, key, fileset, treenames[key], cut_string, preprocess_config, is_data_list[key])
                for key in fileset.keys()
            ]
    
            # for future in as_completed(futures):
            #     key, result = future.result()
            #     output[key]["files"] = list(result)
            for future in as_completed(futures):
                key, result = future.result()
            
                cleaned = [f for f in result if f is not None]
            
                if len(cleaned) == 0:
                    logger.warning(f"No ServiceX output files for dataset {key}")
            
                output[key]["files"] = cleaned
        
        return self._add_tree_info_to_files(output, treenames)

    def _extract_treenames(self,datasets): #Prayag:WIP
        keys = self.fileset.keys()
        out = {}
        for name in datasets:
            treename = self.dataset_manager.get_tree_name(name)
            for key in keys:
                if key.startswith(name):
                    out[key] = treename
        return out

    def _extract_isdata(self,datasets): #Prayag:WIP
        keys = self.fileset.keys()
        out = {}
        for name in datasets:
            isdata = self.dataset_manager.is_data_dataset(name)
            for key in keys:
                if key.startswith(name):
                    out[key] = isdata
        return out
        
    def _generate_servicex_preskim(
        self,
        identifiers: Optional[Union[int, List[int]]],
    ) -> None: #Prayag:WIP
        """Use servicex to do the pre-skimming"""
        logger.info("Performing the ServiceX Pre-skim ...")
        self.fileset, self.datasets = self.fileset_builder.build_fileset(
            identifiers, self.processes_filter
        )
        datasets = list(self.dataset_manager.datasets.keys())
        treenames = self._extract_treenames(datasets)
        is_data_dataset = self._extract_isdata(datasets)
        # logger.info(f"is_data_dataset : {is_data_dataset}")
        
        self._servicex_output = self._batch_deliver(
            fileset = self.fileset,
            treenames = treenames,
            is_data_list = is_data_dataset,
            servicex_config = self.config.servicex_preskim,
            preprocess_config = self.config.preprocess
        )

        self._use_servicex_as_input = True
        # logger.info(f"self.fileset\n {self._servicex_output} \n \n")
        logger.info("ServiceX Pre-skim complete.")
    
    def _generate_metadata(
        self,
        executor: Any,
        schema: Any,
        identifiers: Optional[Union[int, List[int]]],
    ) -> None:
        """
        Generate metadata workflow.

        Parameters
        ----------
        executor : coffea executor
            Executor for coffea preprocessing
        schema : coffea schema or None
            Schema for parsing ROOT files. If None, uses NanoAODSchema.
        identifiers : int or list of ints, optional
            Listing file IDs to process
        """
        logger.info("Starting metadata generation workflow...")

        # Create extractor on-demand with provided executor
        if schema is None:
            schema = NanoAODSchema
        metadata_extractor = CoffeaMetadataExtractor(executor, schema, self.chunksize)

        # Step 1: Build and save fileset and Dataset objects
        # If servicex preskimming was active, use the output from that
        if self._use_servicex_as_input:
            self.fileset = self._servicex_output
            # logger.info(f"self.fileset : {self.fileset}")
        else:
            self.fileset, self.datasets = self.fileset_builder.build_fileset(
                identifiers, self.processes_filter
            )
        self.fileset_builder.save_fileset(self.fileset)

        # Step 2: Extract and save WorkItem metadata
        self.workitems = metadata_extractor.extract_metadata(self.fileset)
        self._save_workitems()

        # Step 3: Aggregate event counts and save summary
        self._summarize_event_counts()
        self._save_nanoaods_summary()

        logger.info("Metadata generation complete.")

    def _load_existing_metadata(self) -> None:
        """
        Load existing metadata from disk.

        Raises
        ------
        SystemExit
            If metadata files are missing or corrupted
        """
        logger.info(
            f"Loading existing metadata from:\n{pretty_repr(self._get_metadata_paths())}"
        )

        try:
            self._load_fileset()
            self._load_workitems()
            self._load_nanoaods_summary()
            logger.info("All metadata successfully loaded from disk.")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load existing metadata: {e}")
            logger.error("Please ensure metadata files exist or enable generation.")
            sys.exit(1)

    def _summarize_event_counts(self) -> None:
        """Aggregate event counts from WorkItems."""
        if self.workitems is None:
            raise ValueError(
                "WorkItems are not available to summarize. "
                "Call run(generate_metadata=True) first."
            )

        logger.info("Aggregating event counts from WorkItems...")

        # Use core functions for aggregation
        event_counts = aggregate_workitem_events(self.workitems)
        self.nanoaods_summary = format_event_summary(event_counts)

        logger.info("Event count summary generated.")

    def _save_workitems(self) -> None:
        """Save WorkItems to JSON."""
        if self.workitems is None:
            raise ValueError("No workitems to save.")

        paths = self._get_metadata_paths()
        serialized = serialize_workitems(self.workitems)
        save_json(serialized, paths["workitems_path"])

    def _save_nanoaods_summary(self) -> None:
        """Save event count summary to JSON."""
        if self.nanoaods_summary is None:
            raise ValueError("No summary to save.")

        paths = self._get_metadata_paths()

        # Save main summary file
        save_json(self.nanoaods_summary, paths["nanoaods_summary_path"])

        # Save per-process summary files
        for process_name, variations in self.nanoaods_summary.items():
            for variation_label, data in variations.items():
                per_process_path = (
                    self.output_directory / f"nanoaods_{process_name}_{variation_label}.json"
                )
                save_json(
                    {process_name: {variation_label: data}},
                    per_process_path
                )
                logger.debug(f"Saved per-process summary: {per_process_path}")

    def _load_fileset(self) -> None:
        """Load fileset and reconstruct Dataset objects from disk."""
        paths = self._get_metadata_paths()
        self.fileset = load_json(paths["fileset_path"])

        # Reconstruct Dataset objects from fileset
        # Group fileset keys by process name
        from collections import defaultdict
        process_groups = defaultdict(lambda: {"keys": [], "xsecs": [], "years": []})

        for dataset_key, entry in self.fileset.items():
            metadata = entry.get("metadata", {})
            process = metadata.get("process")
            xsec = metadata.get("xsec", 1.0)
            is_data = metadata.get("is_data", False)
            variation = metadata.get("variation", "nominal")
            year = metadata.get("year")

            if process:
                process_groups[process]["keys"].append(dataset_key)
                process_groups[process]["xsecs"].append(xsec)
                process_groups[process]["years"].append(year)
                process_groups[process]["is_data"] = is_data
                process_groups[process]["variation"] = variation

        # Create Dataset objects
        self.datasets = []
        for process, data in process_groups.items():
            # Build list of lumi_mask_configs (one per fileset_key)
            lumi_mask_configs = []
            for idx in range(len(data["keys"])):
                lumi_mask_config = self.dataset_manager.get_lumi_mask_config(process, directory_index=idx)
                lumi_mask_configs.append(lumi_mask_config)

            dataset = Dataset(
                name=process,
                fileset_keys=data["keys"],
                process=process,
                variation=data["variation"],
                cross_sections=data["xsecs"],
                is_data=data["is_data"],
                lumi_mask_configs=lumi_mask_configs,
                years=data["years"],
                events=None,
            )
            self.datasets.append(dataset)

        logger.info(f"Loaded {len(self.datasets)} Dataset objects from fileset")

    def _load_workitems(self) -> None:
        """Load WorkItems from disk."""
        paths = self._get_metadata_paths()
        serialized_data = load_json(paths["workitems_path"])
        self.workitems = deserialize_workitems(serialized_data)
        logger.info(f"Loaded {len(self.workitems)} WorkItems")

    def _load_nanoaods_summary(self) -> None:
        """Load event count summary from disk."""
        paths = self._get_metadata_paths()
        self.nanoaods_summary = load_json(paths["nanoaods_summary_path"])
        logger.info("Loaded event count summary")

    def get_coffea_fileset(self) -> Dict[str, Dict[str, Any]]:
        """
        Get coffea-compatible fileset from generated/loaded metadata.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Coffea-compatible fileset dictionary

        Raises
        ------
        ValueError
            If fileset hasn't been generated or loaded yet
        """
        if self.fileset is None:
            raise ValueError(
                "Fileset has not been generated yet. "
                "Call run(generate_metadata=True) first."
            )

        logger.info(f"Returning coffea fileset with {len(self.fileset)} datasets")
        return self.fileset

    def build_metadata_lookup(self) -> Dict[str, Dict[str, Any]]:
        """
        Build metadata lookup dictionary from Dataset objects and event summary.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping of fileset_key -> {process, variation, xsec, nevts, is_data, lumi_mask_config, dataset}

        Raises
        ------
        ValueError
            If datasets or summary haven't been generated/loaded
        """
        if self.datasets is None:
            raise ValueError(
                "Datasets have not been generated yet. "
                "Call run(generate_metadata=True) first."
            )

        if self.nanoaods_summary is None:
            logger.warning(
                "nanoaods_summary is None. Event counts (nevts) will be set to 0. "
                "This may affect MC normalization."
            )

        lookup = {}

        for dataset in self.datasets:
            for fileset_key in dataset.fileset_keys:
                # Get index for this fileset_key
                try:
                    idx = dataset.fileset_keys.index(fileset_key)
                except ValueError as e:
                    logger.error(f"Failed to find index for {fileset_key}: {e}")
                    idx = 0

                # Get cross-section for this fileset_key
                try:
                    xsec = dataset.cross_sections[idx]
                except IndexError as e:
                    logger.error(f"Failed to get cross-section for {fileset_key}: {e}")
                    xsec = 1.0

                # Get lumi_mask_config for this fileset_key
                try:
                    lumi_mask_config = dataset.lumi_mask_configs[idx] if dataset.lumi_mask_configs else None
                except IndexError as e:
                    logger.warning(f"Failed to get lumi_mask_config for {fileset_key}: {e}")
                    lumi_mask_config = None

                # Get year for this fileset_key
                try:
                    year = dataset.years[idx] if dataset.years else None
                except IndexError as e:
                    logger.warning(f"Failed to get year for {fileset_key}: {e}")
                    year = None

                # Extract nevts from summary
                nevts = extract_nevts_from_summary(
                    fileset_key,
                    dataset.variation,
                    self.nanoaods_summary,
                )

                lookup[fileset_key] = {
                    "process": dataset.process,
                    "variation": dataset.variation,
                    "xsec": xsec,
                    "nevts": nevts,
                    "is_data": dataset.is_data,
                    "lumi_mask_config": lumi_mask_config,
                    "dataset": fileset_key,
                    "year": year,
                }

        logger.info(f"Built metadata lookup for {len(lookup)} fileset keys")
        return lookup
