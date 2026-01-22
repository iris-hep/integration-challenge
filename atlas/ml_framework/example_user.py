import data_fetcher as fetcher

# Example user code

# Define run configuration daata class
my_DS = "user.acordeir:michigan-tutorial.displaced-signal.root"
my_DS2 = "user.alheld:user.alheld.410470.PhPy8EG.DAOD_PHYSLITE.e6337_s3681_r13144_r13146_p6697.IC-v1_output/"
config = fetcher.RunConfig(
    tree_name="reco",
    dataset=my_DS2,
    output_folder="./ml_framework/data/",
    files_per_sample=2,
)


# Define cuts functions with FuncADL
def jet_pt_cut(evt):
    return evt["jet_pt_NOSYS"].Where(lambda pt: pt > 33_000).Count() > 0


cuts = [jet_pt_cut]


# Define selection function
def branch_select(evt):
    return {
        "jet_pt": evt["jet_pt_NOSYS"].Select(lambda pt: pt / 1000.0),
    }


query = fetcher.ServiceXQuery(cuts, branch_select)

# Execute the data fetch
executor = fetcher.ServiceXExecutor(query, config)
executor.deliver()
