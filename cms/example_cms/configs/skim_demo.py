"""
Skimming configuration and selection functions for the Z-prime ttbar analysis.

This module contains all skimming-related configuration including:
- Dataset definitions with cross-sections and paths
- Skimming selection functions
- Skimming configuration parameters
"""
from coffea.analysis_tools import PackedSelection


# ==============================================================================
#  Dataset Configuration
# ==============================================================================

REDIRECTOR="root://xcache/"

datasets_config = [
    {
        "name": "signal",
        "directories": ("example-demo/cms_datasets/2016/ZPrimeToTT_M2000_W200/", 
                      "example-demo/cms_datasets/2017/ZPrimeToTT_M2000_W200/", 
                      "example-demo/cms_datasets/2018/ZPrimeToTT_M2000_W200/"),
        "cross_sections": (0.01895, 0.01895, 0.01895),
        "keep_split": False,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },

    {
        "name": "ttbar_semilep",
        "directories": "example-demo/cms_datasets/2016/TTToSemiLeptonic/",
        "cross_sections": 364.31,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    {
        "name": "ttbar_had",
        "directories": "example-demo/cms_datasets/2016/TTToHadronic/",
        "cross_sections": 380.11,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    {
        "name": "ttbar_lep",
        "directories": "example-demo/cms_datasets/2016/TTToHadronic/",
        "cross_sections": 87.33,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    {
        "name": "wjets",
        "directories": "example-demo/cms_datasets/2016/WJetsToLNu_HT-*/",
        "cross_sections": 61526.7, # for now
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    {
        "name": "data",
        "directories": "example-demo/cms_datasets/2016/SingleMuonRun*/",
        "cross_sections": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": None,
        "redirector": REDIRECTOR,
    }
]

# ==============================================================================
#  Dataset Manager Configuration
# ==============================================================================

dataset_manager_config = {
    "datasets": datasets_config,
    "max_files": None  # No limit by default
}

# ==============================================================================
#  Skimming Configuration
# ==============================================================================


def default_skim_selection(puppimet, hlt):
    """
    Default skimming selection function.

    Applies basic trigger, muon, and MET requirements for skimming.
    This matches the hardcoded behavior from the original preprocessing.
    """

    selection = PackedSelection()

    # Individual cuts
    selection.add("trigger", hlt.TkMu50)
    selection.add("met_cut", puppimet.pt > 50)

    # Combined skimming selection
    selection.add("skim", selection.all("trigger", "met_cut"))

    return selection


skimming_config = {
    "selection_function": default_skim_selection,
    "selection_use": [("PuppiMET", None), ("HLT", None)],
    "chunk_size": 100_000,
    "tree_name": "Events",
}
