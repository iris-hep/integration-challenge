# =========================
# Minimal FuncADL-Uproot Framework
# =========================

from dataclasses import dataclass
from servicex import query, deliver
from servicex_analysis_utils import ds_type_resolver, to_awk
import awkward as ak
import logging
import os


# -------------------------
# Run configuration
# -------------------------
@dataclass
class RunConfig:
    tree_name: str
    dataset: str | list[str]
    request_name: str | list[str]
    output_folder: str = "./data/"
    files_per_sample: int = None
    join_result_parquet: bool = True

    def __post_init__(self):
        if isinstance(self.dataset, str):
            self.dataset = [self.dataset]

        if isinstance(self.request_name, str):
            self.request_name = [self.request_name]

        if not self.output_folder.endswith("/"):
            self.output_folder += "/"

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # If no request names provided, use dataset names
        if self.request_name is None:
            self.request_name = self.dataset

        # Ensure request names and dataset have same length
        if len(self.request_name) != len(self.dataset):
            raise ValueError("Length of request_name must match length of dataset")


# -------------------------
# Query builder
# -------------------------
class ServiceXQuery:
    def __init__(self, cuts=[], selection=None, config=None):
        self.cuts = cuts
        self.selection = selection
        self.config = config
        self.func_adl_query = self.build()

    def build(self):
        if self.config.tree_name is None:
            raise RuntimeError("No tree name defined in configuration")
        q = query.FuncADL_Uproot().FromTree(self.config.tree_name)

        for cut in self.cuts:
            q = q.Where(cut)

        if self.selection is None:
            raise RuntimeError("No Select() defined")

        return q.Select(self.selection)

    def write_to_parquet(self, files: dict, **kwargs):
        arrays = to_awk(files, **kwargs)

        for key, arr in arrays.items():
            logging.info("Key: %s, Entries: %d", key, len(arr))
            file_path = self.config.output_folder + f"{key}.parquet"

            ak.to_parquet(
                arr,
                file_path,
                compression="GZIP",
                compression_level=9,
            )

    def deliver(self, **kwargs):
        queries = []

        for sample, name in zip(self.config.dataset, self.config.request_name):
            if not isinstance(sample, str):
                raise ValueError("Input sample must be a string or list of strings")

            queries.append(
                {
                    "NFiles": self.config.files_per_sample,
                    "Name": name,
                    "Dataset": ds_type_resolver(sample),
                    "Query": self.func_adl_query,
                }
            )

        spec = {"General": {"Delivery": "LocalCache"}, "Sample": queries}

        files = deliver(spec, **kwargs)
        if self.config.join_result_parquet:
            self.write_to_parquet(files, **kwargs)
            # Logging
            logging.warning("Fetched data written to %s", self.config.output_folder)
        else:
            return files
