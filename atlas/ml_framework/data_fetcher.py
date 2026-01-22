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
    request_name: str = "ML_Training_Data"
    output_folder: str = "./data/"
    files_per_sample: int = None

    def __post_init__(self):
        if isinstance(self.dataset, str):
            self.dataset = [self.dataset]

        if not self.output_folder.endswith("/"):
            self.output_folder += "/"

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)


# -------------------------
# Query builder
# -------------------------


class ServiceXQuery:
    def __init__(self, cuts=[None], selection=None):
        self._cuts = cuts  # list of cut functions
        self._select_fn = selection  # selection function

    def build(self, tree_name: str):
        q = query.FuncADL_Uproot().FromTree(tree_name)

        for cut in self._cuts:
            q = q.Where(cut)

        if self._select_fn is None:
            raise RuntimeError("No Select() defined")

        return q.Select(self._select_fn)


# -------------------------
# Executor (ServiceX bridge)
# -------------------------


class ServiceXExecutor:
    def __init__(self, query_builder: ServiceXQuery, config: RunConfig):
        self._query_builder = query_builder
        self._cfg = config

    def write_to_parquet(self, files: dict, **kwargs):
        arrays = to_awk(files, **kwargs)

        for key, arr in arrays.items():
            logging.info("Key: %s, Entries: %d", key, len(arr))
            file_path = self._cfg.output_folder + f"{key}.parquet"

            ak.to_parquet(
                arr,
                file_path,
                compression="GZIP",
                compression_level=9,
            )

    def deliver(self, **kwargs):
        funcadl_query = self._query_builder.build(self._cfg.tree_name)

        queries = []
        for sample in self._cfg.dataset:
            if not isinstance(sample, str):
                raise ValueError("Input sample must be a string or list of strings")

            queries.append(
                {
                    "NFiles": self._cfg.files_per_sample,
                    "Name": sample,
                    "Dataset": ds_type_resolver(sample),
                    "Query": funcadl_query,
                }
            )

        spec = {"General": {"Delivery": "LocalCache"}, "Sample": queries}

        files = deliver(spec, **kwargs)
        self.write_to_parquet(files, **kwargs)
        # Logging
        logging.warning("Fetched data written to %s", self._cfg.output_folder)
