# =========================
# Minimal FuncADL-Uproot Framework
# =========================

from dataclasses import dataclass
from servicex import query, deliver
from servicex_analysis_utils import ds_type_resolver, to_awk
import awkward as ak
import logging


# -------------------------
# Run configuration
# -------------------------
@dataclass
class RunConfig:
    tree_name: str
    dataset: str
    request_name: str = "ML_Training_Data"
    output_path: str = "./data/training_samples.parquet"
    ignore_cache: bool = False


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

    def write_to_file(self, files: dict, **kwargs):
        arrays = to_awk(files, **kwargs)[self._cfg.request_name]

        ak.to_parquet(
            arrays, self._cfg.output_path, compression="GZIP", compression_level=9
        )

    def deliver(self, **kwargs):
        funcadl_query = self._query_builder.build(self._cfg.tree_name)

        spec = {
            "Sample": [
                {
                    "Name": self._cfg.request_name,
                    "Dataset": ds_type_resolver(self._cfg.dataset),
                    "Query": funcadl_query,
                }
            ],
        }
        files = deliver(spec, ignore_local_cache=self._cfg.ignore_cache, **kwargs)
        self.write_to_file(files, **kwargs)
        # Logging
        logging.info("Data written to %s", self._cfg.output_path)
