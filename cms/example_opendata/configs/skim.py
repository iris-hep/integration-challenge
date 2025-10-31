"""
Skimming configuration and selection functions for the Z-prime ttbar analysis.

This module contains all skimming-related configuration including:
- Dataset definitions with cross-sections and paths
- Skimming selection functions
- Skimming configuration parameters
"""

from coffea.analysis_tools import PackedSelection
from .cuts import lumi_mask


# ==============================================================================
#  Dataset Configuration
# ==============================================================================

datasets_config = [
    {
        "name": "signal",
        "directories": "example_opendata/datasets/signal/m2000_w20/",
        "cross_sections": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "ttbar_semilep",
        "directories": "example_opendata/datasets/ttbar_semilep/",
        "cross_sections": 831.76 * 0.438,  # 364.35
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "ttbar_had",
        "directories": "example_opendata/datasets/ttbar_had/",
        "cross_sections": 831.76 * 0.457,  # 380.11
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "ttbar_lep",
        "directories": "example_opendata/datasets/ttbar_lep/",
        "cross_sections": 831.76 * 0.105,  # 87.33
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "wjets",
        "directories": "example_opendata/datasets/wjets/",
        "cross_sections": 61526.7,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "data",
        "directories": "example_opendata/datasets/data/",
        "cross_sections": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": None,
        "redirector": "",
        "is_data": True,
        "lumi_mask": {
            "function": lumi_mask,
            "use": [("event", "run"), ("event", "luminosityBlock")],
            "static_kwargs": {"lumifile": "./example_opendata/corrections/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"},
        },
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
    "function": default_skim_selection,
    "use": [("PuppiMET", None), ("HLT", None)],
    "chunk_size": 100_000,
    "tree_name": "Events",
}
