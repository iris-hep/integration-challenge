# =========================
# Minimal FuncADL-Uproot Framework
# =========================

from dataclasses import dataclass
from servicex import query, deliver
from servicex_analysis_utils import ds_type_resolver, to_awk
import awkward as ak


# -------------------------
# Run configuration
# -------------------------
@dataclass
class RunConfig:
    tree_name: str
    request_name: str = "ML_Training_Data"
    output_path: str = "training_data.parquet"
    ignore_cache: bool = False


# -------------------------
# Pure cut functions
# One cut = one branch
# -------------------------


def make_jet_pt_cut(pt_min: float):
    def cut(evt) -> bool:
        return evt["jet_pt_NOSYS"].Where(lambda pt: pt > pt_min).Count() > 0

    return cut


def make_jet_eta_cut(eta_max: float):
    def cut(evt) -> bool:
        return evt["truth_alp_eta"].Where(lambda eta: abs(eta) < eta_max).Count() > 0

    return cut


def make_jet_select():
    def select(evt):
        return {
            "jet_pt": evt["jet_pt_NOSYS"],
            "jet_eta": evt["truth_alp_eta"],
        }

    return select


# -------------------------
# Query builder
# -------------------------


class JetQuery:
    def __init__(self):
        self._cuts = []
        self._select_fn = None

    def add_cut(self, cut_fn):
        # cut_fn MUST be: f(evt) -> bool
        self._cuts.append(cut_fn)
        return self

    def select(self, select_fn):
        # select_fn must be f(evt) -> dict
        self._select_fn = select_fn
        return self

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
    def __init__(self, ds):
        self._dataset = ds_type_resolver(ds)

    def run(self, query_builder: JetQuery, config: RunConfig):
        funcadl_query = query_builder.build(config.tree_name)

        spec = {
            "Sample": [
                {
                    "Name": config.request_name,
                    "Dataset": self._dataset,
                    "Query": funcadl_query,
                }
            ]
        }

        return spec


# -------------------------
# User code
# -------------------------

"""
def fetch_training_data_to_file(ds_name: str, config: RunConfig):
    result_list = fetch_training_data(ds_name, config)

    # Finally, write it out into a training file.
    ak.to_parquet(
        result_list, config.output_path, compression="GZIP", compression_level=9
    )
"""

files = [
    "root://eospublic.cern.ch//eos/opendata/atlas/rucio/data16_13TeV/"
    "DAOD_PHYSLITE.37019878._000001.pool.root.1"
]

my_DS = "user.acordeir:michigan-tutorial.displaced-signal.root"

query_builder = (
    JetQuery()
    .add_cut(make_jet_pt_cut(pt_min=30_000))
    .add_cut(make_jet_eta_cut(eta_max=2.5))
    .select(make_jet_select())
)

config = RunConfig(tree_name="reco", request_name="testin2")

executor = ServiceXExecutor(my_DS)
result = executor.run(query_builder, config)
print(result)


f = deliver(result, ignore_local_cache=config.ignore_cache)
print(f)

result_list = to_awk(f)
print(ak.mean(result_list["testin2"]["jet_pt"]))  # just to show we have data

# NEXT STEP
# USER CODE OUTSIDE THE FRAMEWORK:
# ak.to_parquet as well
# modular choices for cuts and selections
