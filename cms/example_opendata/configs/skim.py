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

datasets_config = [
    {
        "name": "signal",
        "directories": "example-demo/datasets/signal/m2000_w20/",
        "cross_sections": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "ttbar_semilep",
        "directories": "example-demo/datasets/ttbar_semilep/",
        "cross_sections": 831.76 * 0.438,  # 364.35
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "ttbar_had",
        "directories": "example-demo/datasets/ttbar_had/",
        "cross_sections": 831.76 * 0.457,  # 380.11
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "ttbar_lep",
        "directories": "example-demo/datasets/ttbar_lep/",
        "cross_sections": 831.76 * 0.105,  # 87.33
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "wjets",
        "directories": "example-demo/datasets/wjets/",
        "cross_sections": 61526.7,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": "",
    },
    {
        "name": "data",
        "directories": "example-demo/datasets/data/",
        "cross_sections": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": None,
        "redirector": "",
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
