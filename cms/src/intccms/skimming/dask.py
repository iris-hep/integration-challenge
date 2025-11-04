"""Dask-based orchestration for parallel workitem skimming.

This module provides the orchestration layer that coordinates parallel processing
of workitems using dask.bag, with retry logic and comprehensive failure handling.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, TypedDict

import dask
import dask.bag
from coffea.processor.executor import WorkItem
from tabulate import tabulate

from intccms.skimming.workitem import process_workitem, resolve_output_path, ManifestEntry
from intccms.utils.schema import SkimmingConfig, default_histogram

logger = logging.getLogger(__name__)


# Type definitions for dask result structures
class FailureInfo(TypedDict):
    """Information about a failed workitem for error reporting."""
    dataset: str  # Dataset name from workitem
    error_type: str  # Exception class name
    error_msg: str  # Exception message
    filename: str  # Source file that failed


class MergedResult(TypedDict):
    """Aggregated results from multiple workitem processings.

    This structure is the output of merge_results() and represents the
    combined state from all parallel workers.
    """
    hist: object  # Merged histogram (dummy for compatibility)
    failed_items: Set[WorkItem]  # Union of all failed workitems
    processed_events: int  # Sum of processed events across all workers
    output_files: List[str]  # Concatenated list of all output files
    manifest_entries: List[ManifestEntry]  # All manifest entries from all workers
    failure_infos: List[FailureInfo]  # Detailed failure information for reporting


def merge_results(result_a: Dict[str, Any], result_b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two result dictionaries from parallel workitem processing.

    Used by dask.bag.fold() to combine results from parallel workers. Adds
    histograms, unions failed item sets, concatenates output file lists, and
    accumulates failure information for detailed error reporting.

    Args:
        result_a: First result dictionary
        result_b: Second result dictionary

    Returns:
        Combined result dictionary with merged statistics
    """
    # Collect failure info from both results
    failure_infos = []
    if result_a.get("failure_info"):
        failure_infos.append(result_a["failure_info"])
    if result_b.get("failure_info"):
        failure_infos.append(result_b["failure_info"])

    # If we have multiple failures, accumulate them as a list
    if "failure_infos" in result_a:
        failure_infos.extend(result_a["failure_infos"])
    if "failure_infos" in result_b:
        failure_infos.extend(result_b["failure_infos"])

    return {
        "hist": result_a["hist"] + result_b["hist"],
        "failed_items": result_a["failed_items"] | result_b["failed_items"],
        "processed_events": result_a["processed_events"] + result_b["processed_events"],
        "output_files": result_a["output_files"] + result_b["output_files"],
        "failure_infos": failure_infos,
        "manifest_entries": result_a.get("manifest_entries", []) + result_b.get("manifest_entries", []),
    }


class WorkitemSkimmingManager:
    """Manager for workitem-based skimming using dask.bag processing.

    This class orchestrates the preprocessing workflow that processes
    workitems directly using dask.bag, providing robust failure handling
    and retry mechanisms.

    Attributes
    ----------
    config : SkimmingConfig
        Skimming configuration with selection functions and output settings
    output_manager : OutputDirectoryManager
        Centralized output directory manager
    """

    def __init__(self, config: SkimmingConfig, output_manager: Any):
        """Initialize the workitem skimming manager.

        Args:
            config: Skimming configuration with selection functions and output settings
            output_manager: Centralized output directory manager
        """
        self.config = config
        self.output_manager = output_manager
        logger.info("Initialized workitem-based skimming manager")

    def process_workitems(
        self,
        workitems: List[WorkItem],
        configuration: Any,
        datasets: List[Any],
        split_every: int = 4,
    ) -> Dict[str, Any]:
        """Process a list of workitems using dask.bag with failure handling.

        This is the main entry point that implements the dask.bag workflow
        with retry logic for failed workitems.

        Args:
            workitems: List of workitems to process
            configuration: Main analysis configuration object
            datasets: Dataset objects with metadata (including is_data flag)
            split_every: Split parameter for dask.bag.fold operation

        Returns:
            Final combined results with histograms and processing statistics
        """
        # Setup phase
        max_retries = self.config.max_retries
        self._log_start(workitems, max_retries)

        dataset_lookup = self._build_dataset_lookup(datasets)
        full_result = self._init_result()

        # Retry loop
        remaining_workitems = workitems.copy()
        retry_count = 0

        while remaining_workitems and retry_count < max_retries:
            self._log_attempt_start(retry_count, remaining_workitems)

            # Execute dask workflow
            result = self._execute_dask_workflow(
                remaining_workitems,
                configuration,
                dataset_lookup,
                split_every,
            )

            # Update state
            remaining_workitems = list(result["failed_items"])
            self._accumulate_results(full_result, result)

            # Report progress
            self._log_attempt_complete(workitems, remaining_workitems, result, retry_count)
            retry_count += 1

        # Finalize
        if remaining_workitems:
            full_result["failed_items"] = set(remaining_workitems)

        self._log_final_summary(workitems, full_result, remaining_workitems, max_retries)

        # Save manifest
        self._save_manifest(full_result["manifest_entries"])

        return full_result

    def discover_workitem_outputs(self, workitems: List[WorkItem]) -> List[str]:
        """Discover existing output files from previous workitem processing.

        This method scans for output files that would be created by the
        workitem processing, allowing for resumption of interrupted workflows.

        Args:
            workitems: List of workitems to check for existing outputs

        Returns:
            List of existing output file paths
        """
        output_files = []
        dataset_counts = {}

        for workitem in workitems:
            resolved_path, like_local, _ = resolve_output_path(
                workitem,
                self.config.output,
                self.output_manager,
            )

            if like_local:
                if Path(resolved_path).exists():
                    output_files.append(resolved_path)
            else:
                # TODO: consistent way to check existence for non-local paths
                output_files.append(resolved_path)

            dataset = workitem.dataset
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        # Log with dataset breakdown
        if dataset_counts:
            dataset_info = ", ".join(
                [f"{dataset}: {count}" for dataset, count in dataset_counts.items()]
            )
            logger.info(f"Found existing skimmed files for {dataset_info}")
        else:
            logger.info("No existing output files found")

        return output_files

    # =========================================================================
    # Helper methods: Computation
    # =========================================================================

    def _build_dataset_lookup(self, datasets: List[Any]) -> Dict[str, Any]:
        """Build mapping from fileset keys to dataset objects.

        Args:
            datasets: List of Dataset objects with fileset_keys

        Returns:
            Dictionary mapping fileset_key -> Dataset object
        """
        dataset_lookup = {}
        for dataset in datasets or []:
            for fileset_key in dataset.fileset_keys:
                dataset_lookup[fileset_key] = dataset
        return dataset_lookup

    def _init_result(self) -> Dict[str, Any]:
        """Initialize empty result accumulator.

        Returns:
            Dictionary with empty histogram, sets, and counters
        """
        return {
            "hist": default_histogram(),
            "failed_items": set(),
            "processed_events": 0,
            "output_files": [],
            "failure_infos": [],
            "manifest_entries": [],
        }

    def _execute_dask_workflow(
        self,
        workitems: List[WorkItem],
        configuration: Any,
        dataset_lookup: Dict[str, Any],
        split_every: int,
    ) -> Dict[str, Any]:
        """Execute the dask.bag workflow for a batch of workitems.

        Args:
            workitems: List of workitems to process
            configuration: Main analysis configuration
            dataset_lookup: Mapping from fileset keys to datasets
            split_every: Split parameter for fold operation

        Returns:
            Combined result dictionary from all workitems
        """
        bag = dask.bag.from_sequence(workitems)

        futures = bag.map(
            lambda wi: process_workitem(
                wi,
                self.config,
                configuration,
                self.output_manager,
                is_mc=not (
                    (dataset := dataset_lookup.get(wi.dataset)) and dataset.is_data
                ),
            )
        )

        task = futures.fold(merge_results, split_every=split_every)
        (result,) = dask.compute(task)
        return result

    def _accumulate_results(
        self, full_result: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Accumulate successful results and failures into full_result.

        Modifies full_result in place.

        Args:
            full_result: Accumulator for all results
            result: Result from current attempt
        """
        if result["processed_events"] > 0:
            full_result["hist"] += result["hist"]
            full_result["processed_events"] += result["processed_events"]
            full_result["output_files"].extend(result["output_files"])

        if result.get("failure_infos"):
            full_result["failure_infos"].extend(result["failure_infos"])

        if result.get("manifest_entries"):
            full_result["manifest_entries"].extend(result["manifest_entries"])

    def _save_manifest(self, manifest_entries: List[Dict[str, Any]]) -> None:
        """Save manifest JSON mapping output files to source files.

        Args:
            manifest_entries: List of manifest metadata dictionaries
        """
        if not manifest_entries:
            logger.info("No manifest entries to save")
            return

        # Group entries by dataset
        by_dataset = defaultdict(list)
        for entry in manifest_entries:
            by_dataset[entry["dataset"]].append(entry)

        # Save one manifest per dataset
        base_dir = Path(self.output_manager.skimmed_dir)
        for dataset, entries in by_dataset.items():
            manifest_path = base_dir / dataset / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(manifest_path, "w") as f:
                json.dump(entries, f, indent=2)

            logger.info(f"Saved manifest for {dataset}: {manifest_path} ({len(entries)} entries)")

    # =========================================================================
    # Helper methods: Logging
    # =========================================================================

    def _log_start(self, workitems: List[WorkItem], max_retries: int) -> None:
        """Log processing start.

        Args:
            workitems: List of workitems to process
            max_retries: Maximum retry attempts
        """
        logger.info(
            f"Processing {len(workitems)} workitems with max {max_retries} retries"
        )

    def _log_attempt_start(
        self, retry_count: int, remaining_workitems: List[WorkItem]
    ) -> None:
        """Log attempt start.

        Args:
            retry_count: Current retry attempt number (0-indexed)
            remaining_workitems: Workitems remaining to process
        """
        logger.info(
            f"Attempt {retry_count + 1}: processing {len(remaining_workitems)} workitems"
        )

    def _log_attempt_complete(
        self,
        all_workitems: List[WorkItem],
        remaining_workitems: List[WorkItem],
        result: Dict[str, Any],
        retry_count: int,
    ) -> None:
        """Log attempt completion with success/failure counts.

        Args:
            all_workitems: Original full list of workitems
            remaining_workitems: Workitems that failed in this attempt
            result: Result dictionary from this attempt
            retry_count: Current retry attempt number (0-indexed)
        """
        failed_count = len(remaining_workitems)
        successful_count = len(all_workitems) - failed_count
        logger.info(
            f"Attempt {retry_count + 1} complete: "
            f"{successful_count} successful, {failed_count} failed"
        )

        # Show detailed failure summary if there are failures
        if remaining_workitems and result.get("failure_infos"):
            logger.warning(f"\n=== Failures in Attempt {retry_count + 1} ===")
            self._log_failure_summary(all_workitems, result["failure_infos"])

    def _log_final_summary(
        self,
        workitems: List[WorkItem],
        full_result: Dict[str, Any],
        remaining_workitems: List[WorkItem],
        max_retries: int,
    ) -> None:
        """Log final processing summary.

        Args:
            workitems: Original full list of workitems
            full_result: Final accumulated result
            remaining_workitems: Workitems that failed after all retries
            max_retries: Maximum retry attempts configured
        """
        if remaining_workitems:
            logger.warning(
                f"Failed to process {len(remaining_workitems)} workitems "
                f"after {max_retries} attempts"
            )
            logger.warning("\n=== Final Failure Summary (All Attempts) ===")
            self._log_failure_summary(workitems, full_result["failure_infos"])
        else:
            logger.info("All workitems processed successfully")

        self._log_processing_summary(
            workitems, full_result["output_files"], full_result["processed_events"]
        )

    def _log_processing_summary(
        self, workitems: List[WorkItem], output_files: List[str], total_events: int
    ) -> None:
        """Log a summary table of processing results by dataset.

        Args:
            workitems: Original list of workitems processed
            output_files: List of output files created
            total_events: Total number of events processed
        """
        # Collect statistics by dataset
        dataset_stats = defaultdict(lambda: {"files_written": 0})

        # Count files written per dataset
        for output_file in output_files:
            try:
                # Extract dataset name from output path structure
                # IMPORTANT: Assumes path format: .../output_dir/{dataset}/file_{N}/part_{M}.ext
                # path_parts[-3] gets dataset from this fixed structure
                # If build_output_path() changes, this logic must be updated
                path_parts = Path(output_file).parts
                if len(path_parts) >= 3:
                    dataset = path_parts[-3]
                    dataset_stats[dataset]["files_written"] += 1
            except Exception:
                pass

        # Create summary table
        if dataset_stats:
            table_data = []
            total_files = 0

            for dataset, stats in sorted(dataset_stats.items()):
                files = stats["files_written"]
                table_data.append([dataset, files])
                total_files += files

            # Add totals row
            table_data.append(["TOTAL", total_files])

            # Create and log table
            headers = ["Dataset", "Files Written"]
            table = tabulate(table_data, headers=headers, tablefmt="grid")

            logger.info(f"Processing Summary: {total_events:,} total events saved")
            logger.info(f"\n{table}")
        else:
            logger.info("No output files were created during processing")

    def _log_failure_summary(
        self, workitems: List[WorkItem], failure_infos: List[Dict[str, Any]]
    ) -> None:
        """Log a detailed summary of failures by dataset, including percentages and error types.

        Args:
            workitems: Original list of workitems processed
            failure_infos: List of failure information dictionaries
        """
        if not failure_infos:
            return

        # Count total workitems per dataset
        total_per_dataset = defaultdict(int)
        for workitem in workitems:
            total_per_dataset[workitem.dataset] += 1

        # Organize failures by dataset
        failures_by_dataset = defaultdict(list)
        for failure in failure_infos:
            dataset = failure["dataset"]
            failures_by_dataset[dataset].append(failure)

        # Count error types per dataset
        error_types_by_dataset = defaultdict(lambda: defaultdict(int))
        for failure in failure_infos:
            dataset = failure["dataset"]
            error_type = failure["error_type"]
            error_types_by_dataset[dataset][error_type] += 1

        # Count global error types for overall statistics
        global_error_counts = defaultdict(int)
        for failure in failure_infos:
            global_error_counts[failure["error_type"]] += 1

        # Create summary table
        table_data = []
        total_failed = 0
        total_workitems = sum(total_per_dataset.values())

        for dataset, failures in sorted(failures_by_dataset.items()):
            failed_count = len(failures)
            total_count = total_per_dataset[dataset]
            percentage = (failed_count / total_count * 100) if total_count > 0 else 0

            # Build error type breakdown string
            error_counts = error_types_by_dataset[dataset]
            error_breakdown = ", ".join(
                [
                    f"{error_type}: {count} ({count/failed_count*100:.0f}%)"
                    for error_type, count in sorted(
                        error_counts.items(), key=lambda x: -x[1]
                    )
                ]
            )

            table_data.append(
                [
                    dataset,
                    failed_count,
                    total_count,
                    f"{percentage:.1f}%",
                    error_breakdown,
                ]
            )
            total_failed += failed_count

        # Add totals row with global error breakdown
        total_percentage = (
            (total_failed / total_workitems * 100) if total_workitems > 0 else 0
        )
        global_error_breakdown = ", ".join(
            [
                f"{error_type}: {count} ({count/total_failed*100:.0f}%)"
                for error_type, count in sorted(
                    global_error_counts.items(), key=lambda x: -x[1]
                )
            ]
        )
        table_data.append(
            [
                "TOTAL",
                total_failed,
                total_workitems,
                f"{total_percentage:.1f}%",
                global_error_breakdown,
            ]
        )

        # Create and log table
        headers = ["Dataset", "Failed", "Total", "Failure %", "Error Types (count, %)"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")

        logger.warning("Failure Summary:")
        logger.warning(f"\n{table}")

        # Log sample failures with file names for debugging
        logger.warning("\nSample failures (first 5):")
        for i, failure in enumerate(failure_infos[:5]):
            logger.warning(
                f"  {i+1}. [{failure['dataset']}] {failure['error_type']}: "
                f"{failure['error_msg'][:100]}... (file: {Path(failure['filename']).name})"
            )

        if len(failure_infos) > 5:
            logger.warning(f"  ... and {len(failure_infos) - 5} more failures")
