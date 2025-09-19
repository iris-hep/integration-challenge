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
        "directory": "example/datasets/signal/m2000_w20/",
        "cross_section": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "ttbar_semilep",
        "directory": "example/datasets/ttbar_semilep/",
        "cross_section": 831.76 * 0.438,  # 364.35
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "ttbar_had",
        "directory": "example/datasets/ttbar_had/",
        "cross_section": 831.76 * 0.457,  # 380.11
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "ttbar_lep",
        "directory": "example/datasets/ttbar_lep/",
        "cross_section": 831.76 * 0.105,  # 87.33
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "wjets",
        "directory": "example/datasets/wjets/",
        "cross_section": 61526.7,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "data",
        "directory": "example/datasets/data/",
        "cross_section": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
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


def default_skim_selection(muons, puppimet, hlt):
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
