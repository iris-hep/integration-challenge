"""
Skimming configuration and selection functions for the Z-prime ttbar analysis.

This module contains all skimming-related configuration including:
- Dataset definitions with cross-sections and paths
- Skimming selection functions
- Skimming configuration parameters
"""
import json
import os
from pathlib import Path
from typing import List, Tuple

from coffea.analysis_tools import PackedSelection
from cuts import lumi_mask
from utils.schema import WorkerEval


def get_cross_sections_for_datasets(
    years: List[str],
    dataset_names: List[str],
    base_path: str = "example_cms/cms_datasets"
) -> Tuple[str, ...]:
    """
    Extract cross-sections for datasets across multiple years.

    Parameters
    ----------
    years : List[str]
        List of years (e.g., ["2016", "2017", "2018"])
    dataset_names : List[str]
        List of dataset names to look up (e.g., ["WJetsToLNu_HT-70To100"])
    base_path : str
        Base path to the cms_datasets directory

    Returns
    -------
    Tuple[float, ...]
        Tuple of cross-sections matching the order of years * dataset_names

    Raises
    ------
    ValueError
        If a dataset is not found in the xsecs.json file
    """
    cross_sections = []

    for year in years:
        xsecs_file = Path(base_path) / year / "xsecs.json"
        with open(xsecs_file, "r") as f:
            xsecs = json.load(f)

        for dataset_name in dataset_names:
            if dataset_name not in xsecs:
                raise ValueError(f"Dataset '{dataset_name}' not found in {xsecs_file}")
            cross_sections.append(xsecs[dataset_name])

    return tuple(cross_sections)


# ==============================================================================
#  Dataset Configuration
# ==============================================================================

REDIRECTOR = "root://xcache/"
YEARS = ["2016", "2017", "2018"]

# WJets HT-binned samples
WJETS_DATASETS = [
    "WJetsToLNu_HT-70To100",
    "WJetsToLNu_HT-100To200",
    "WJetsToLNu_HT-200To400",
    "WJetsToLNu_HT-400To600",
    "WJetsToLNu_HT-600To800",
    "WJetsToLNu_HT-800To1200",
    "WJetsToLNu_HT-1200To2500",
    "WJetsToLNu_HT-2500ToInf",
]

# DYJets HT-binned samples
DYJETS_DATASETS = [
    "DYJetsToLL_M-50_HT-70to100",
    "DYJetsToLL_M-50_HT-100to200",
    "DYJetsToLL_M-50_HT-200to400",
    "DYJetsToLL_M-50_HT-400to600",
    "DYJetsToLL_M-50_HT-600to800",
    "DYJetsToLL_M-50_HT-800to1200",
    "DYJetsToLL_M-50_HT-1200to2500",
    "DYJetsToLL_M-50_HT-2500toInf",
]

# Single top samples
SINGLETOP_DATASETS = [
    "ST_s-channel_4f",
    "ST_t-channel_top_4f",
    "ST_t-channel_antitop_4f",
    "ST_tW_top_5f",
    "ST_tW_antitop_5f",
]

# QCD HT-binned samples
QCD_DATASETS = [
    "QCD_HT50to100",
    "QCD_HT100to200",
    "QCD_HT200to300",
    "QCD_HT300to500",
    "QCD_HT500to700",
    "QCD_HT700to1000",
    "QCD_HT1000to1500",
    "QCD_HT1500to2000",
    "QCD_HT2000toInf",
]

# Diboson samples
DIBOSON_DATASETS = ["WW", "WZ", "ZZ"]

datasets_config = [
    # Signal: Z' -> ttbar (M=2000 GeV, W=200 GeV)
    {
        "name": "signal",
        "directories": tuple(f"example_cms/cms_datasets/{year}/ZPrimeToTT_M2000_W200/" for year in YEARS),
        "cross_sections": get_cross_sections_for_datasets(YEARS, ["ZPrimeToTT_M2000_W200"]),
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    # TTbar semileptonic
    {
        "name": "ttbar_semilep",
        "directories": tuple(f"example_cms/cms_datasets/{year}/TTToSemiLeptonic/" for year in YEARS),
        "cross_sections": get_cross_sections_for_datasets(YEARS, ["TTToSemiLeptonic"]),
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    # TTbar hadronic
    {
        "name": "ttbar_had",
        "directories": tuple(f"example_cms/cms_datasets/{year}/TTToHadronic/" for year in YEARS),
        "cross_sections": get_cross_sections_for_datasets(YEARS, ["TTToHadronic"]),
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    # TTbar dileptonic
    {
        "name": "ttbar_lep",
        "directories": tuple(f"example_cms/cms_datasets/{year}/TTTo2L2Nu/" for year in YEARS),
        "cross_sections": get_cross_sections_for_datasets(YEARS, ["TTTo2L2Nu"]),
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    # W+jets (HT-binned, combined across years)
    {
        "name": "wjets",
        "directories": tuple(f"example_cms/cms_datasets/{year}/{ds}/" for year in YEARS for ds in WJETS_DATASETS),
        "cross_sections": get_cross_sections_for_datasets(YEARS, WJETS_DATASETS),
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    # DY+jets (HT-binned, combined across years)
    {
        "name": "dyjets",
        "directories": tuple(f"example_cms/cms_datasets/{year}/{ds}/" for year in YEARS for ds in DYJETS_DATASETS),
        "cross_sections": get_cross_sections_for_datasets(YEARS, DYJETS_DATASETS),
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    # Single top (all channels combined)
    {
        "name": "single_top",
        "directories": tuple(f"example_cms/cms_datasets/{year}/{ds}/" for year in YEARS for ds in SINGLETOP_DATASETS),
        "cross_sections": get_cross_sections_for_datasets(YEARS, SINGLETOP_DATASETS),
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    # QCD multijet (HT-binned, combined across years)
    {
        "name": "qcd",
        "directories": tuple(f"example_cms/cms_datasets/{year}/{ds}/" for year in YEARS for ds in QCD_DATASETS),
        "cross_sections": get_cross_sections_for_datasets(YEARS, QCD_DATASETS),
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    # Diboson (WW, WZ, ZZ combined)
    {
        "name": "diboson",
        "directories": tuple(f"example_cms/cms_datasets/{year}/{ds}/" for year in YEARS for ds in DIBOSON_DATASETS),
        "cross_sections": get_cross_sections_for_datasets(YEARS, DIBOSON_DATASETS),
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "redirector": REDIRECTOR,
    },
    # Data: Single muon (different run periods per year)
    {
        "name": "data",
        "directories": tuple(
            f"example_cms/cms_datasets/{year}/SingleMuonRun{run}/"
            for year, runs in [("2016", ["B", "C", "D", "E", "F"]),
                              ("2017", ["B", "C", "D", "E", "F"]),
                              ("2018", ["A", "B", "C", "D"])]
            for run in runs
        ),
        "cross_sections": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": None,
        "redirector": REDIRECTOR,
        "is_data": True,
        "lumi_mask": {
            "function": lumi_mask,
            "use": [("event", "run"), ("event", "luminosityBlock")],
            "static_kwargs": {"lumifile": "./example_cms/corrections/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"},
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
    selection.add("trigger", hlt.Mu50)
    selection.add("met_cut", puppimet.pt > 50)

    # Combined skimming selection
    selection.add("skim", selection.all("trigger", "met_cut"))

    return selection


skimming_config = {
    "function": default_skim_selection,
    "use": [("PuppiMET", None), ("HLT", None)],
    "chunk_size": 100_000,
    "tree_name": "Events",
    "output": {
        "format": "parquet",
        "protocol": "s3",  # Change to "local" for local filesystem
        "base_uri": "s3://",  # S3 endpoint
        # To switch to local Ceph: change endpoint_url to
        # "http://rook-ceph-rgw-my-store.rook-ceph.svc/triton-116ed3e4-b173-48c1-aea0-affee451feda"
        "to_kwargs": {
            "compression": "zstd",
            "storage_options": {
                "key": WorkerEval(lambda: os.environ['AWS_ACCESS_KEY_ID']),
                "secret": WorkerEval(lambda: os.environ['AWS_SECRET_ACCESS_KEY']),
                "client_kwargs": {
                    "endpoint_url": 'https://red-s3.unl.edu/cmsaf-test-oshadura'
                }
            }
        },
        "from_kwargs": {
            "storage_options": {
                "key": WorkerEval(lambda: os.environ['AWS_ACCESS_KEY_ID']),
                "secret": WorkerEval(lambda: os.environ['AWS_SECRET_ACCESS_KEY']),
                "client_kwargs": {
                    "endpoint_url": 'https://red-s3.unl.edu/cmsaf-test-oshadura'
                }
            }
        }
    }
}
