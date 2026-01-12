import data_fetcher as fetcher
import os

# Example user code

# Define run configuration daata class
my_DS = "user.acordeir:michigan-tutorial.displaced-signal.root"
config = fetcher.RunConfig(
    tree_name="reco",
    request_name="testin2",
    dataset=my_DS,
    output_path="testin2.parquet",
)


# Define cuts functions with FuncADL
def jet_pt_cut(evt):
    return evt["jet_pt_NOSYS"].Where(lambda pt: pt > 31_000).Count() > 0


def jet_eta_cut(evt):
    return evt["truth_alp_eta"].Where(lambda eta: abs(eta) < 2.5).Count() > 0


cuts = [jet_pt_cut, jet_eta_cut]


# Define selection function
def branch_select(evt):
    return {
        "jet_pt": evt["jet_pt_NOSYS"].Select(lambda pt: pt / 1000.0),
        "jet_eta": evt["truth_alp_eta"],
    }


query = fetcher.ServiceXQuery(cuts, branch_select)

# Execute the data fetch
executor = fetcher.ServiceXExecutor(query, config)
executor.deliver()
