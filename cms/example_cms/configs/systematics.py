"""
Systematic uncertainties and corrections configuration for the Z-prime ttbar analysis.

This module contains:
- Year-aware correction file paths
- Corrections configuration (scale factors, pileup weights) keyed by year
- Systematic uncertainties configuration

Correction sources:
- Scale factors: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun2LegacyAnalysis
- POG JSON files: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/
- Common JSON SFs: https://cms-xpog.docs.cern.ch/commonJSONSFs/

Systematics naming convention:
- Correlated across years: simple name (e.g., "muon_id_sf", "btag_hf")
- Decorrelated by year: name_YEAR (e.g., "pileup_2017", "btag_hfstats1_2017")
"""

import awkward as ak
import numpy as np

from intccms.schema.base import ObjVar, Sys

# Marker for systematic string position in correctionlib args
SYS = Sys()

# ==============================================================================
#  Year Configuration
# ==============================================================================

YEARS = ["2016preVFP", "2017", "2018"]

# Base path for correction files
CORRECTIONS_BASE = "./example_cms/corrections/DONT_EXPOSE_CMS_INTERNAL"


def get_correction_file(year: str, correction_type: str) -> str:
    """
    Get the path to a correction file for a given year.

    Parameters
    ----------
    year : str
        Year identifier (2016preVFP, 2017, 2018)
    correction_type : str
        Type of correction (muon_Z, electron, btagging, pileup, JEC, etc.)

    Returns
    -------
    str
        Path to the correction file
    """
    year_dir = "2016" if year.startswith("2016") else year
    suffix = "_preVFP" if year == "2016preVFP" else "_postVFP" if year == "2016postVFP" else ""

    return f"{CORRECTIONS_BASE}/{year_dir}/{correction_type}{suffix}.json.gz"


# ==============================================================================
#  Pileup Correction Keys by Year
# ==============================================================================

PILEUP_KEYS = {
    "2016preVFP": "Collisions16_UltraLegacy_goldenJSON",
    "2016postVFP": "Collisions16_UltraLegacy_goldenJSON",
    "2017": "Collisions17_UltraLegacy_goldenJSON",
    "2018": "Collisions18_UltraLegacy_goldenJSON",
}


# ==============================================================================
#  B-tagging Configuration
# ==============================================================================

# DeepCSV working point thresholds (for reference, shape SF uses full discriminant)
# From: https://btv-wiki.docs.cern.ch/ScaleFactors/
DEEPCSV_WP_THRESHOLDS = {
    "2016preVFP": {"loose": 0.2027, "medium": 0.6001, "tight": 0.8819},
    "2016postVFP": {"loose": 0.1918, "medium": 0.5847, "tight": 0.8767},
    "2017": {"loose": 0.1355, "medium": 0.4506, "tight": 0.7738},
    "2018": {"loose": 0.1208, "medium": 0.4168, "tight": 0.7665},
}



# ==============================================================================
#  Transform Functions
# ==============================================================================

def muon_sf_transform(eta, pt):
    """
    Transform muon inputs for scale factor evaluation.
    Takes leading muon only (analysis requires exactly 1 muon).
    """
    return (np.abs(eta)[:, 0], pt[:, 0])


def _btag_valid_mask(eta, pt, disc):
    """Valid jets for b-tagging: pt > 30, |eta| < 2.5, disc > 0."""
    return (pt > 30) & (np.abs(eta) < 2.5) & (disc > 0)


def btag_hf_transform_in(hadronFlavour, eta, pt, disc):
    """Fake c-jets and invalid jets as light for correctionlib eval."""
    valid = _btag_valid_mask(eta, pt, disc)
    skip = (hadronFlavour == 4) | ~valid
    fake_flavor = ak.where(skip, 0, hadronFlavour)
    return (fake_flavor, np.abs(eta), pt, disc)


def btag_hf_transform_out(sf, hadronFlavour, eta, pt, disc):
    """C-jets and invalid jets get SF=1.0."""
    valid = _btag_valid_mask(eta, pt, disc)
    skip = (hadronFlavour == 4) | ~valid
    return ak.where(skip, 1.0, sf)


def btag_cferr_transform_in(hadronFlavour, eta, pt, disc):
    """Fake non-c jets and invalid jets as c for correctionlib eval."""
    valid = _btag_valid_mask(eta, pt, disc)
    skip = (hadronFlavour != 4) | ~valid
    fake_flavor = ak.where(skip, 4, hadronFlavour)
    return (fake_flavor, np.abs(eta), pt, disc)


def btag_cferr_transform_out(sf, hadronFlavour, eta, pt, disc):
    """Non-c jets and invalid jets get SF=1.0."""
    valid = _btag_valid_mask(eta, pt, disc)
    skip = (hadronFlavour != 4) | ~valid
    return ak.where(skip, 1.0, sf)


# ==============================================================================
#  Corrections Configuration (per year)
# ==============================================================================

def _get_corrections_for_year(year: str) -> list:
    """
    Get corrections configuration for a specific year.

    Parameters
    ----------
    year : str
        Year identifier (2016preVFP, 2017, 2018)

    Returns
    -------
    list
        List of correction configuration dictionaries
    """
    corrections = [
        # ------------------------------------------------------------------
        # Pileup reweighting (decorrelated by year)
        # Signature: (nTrueInt, systematic)
        # ------------------------------------------------------------------
        {
            "name": f"pileup_{year}",
            "file": get_correction_file(year, "pileup"),
            "type": "event",
            "args": [ObjVar("Pileup", "nTrueInt"), SYS],
            "op": "mult",
            "key": PILEUP_KEYS[year],
            "use_correctionlib": True,
            "nominal_idx": "nominal",
            "up_and_down_idx": ["up", "down"],
        },
        # ------------------------------------------------------------------
        # Muon ID scale factor (Medium ID) - correlated across years
        # Signature: (abseta, pt, systematic)
        # ------------------------------------------------------------------
        {
            "name": "muon_id_sf",
            "file": get_correction_file(year, "muon_Z"),
            "type": "event",
            "args": [ObjVar("Muon", "eta"), ObjVar("Muon", "pt"), SYS],
            "transform_in": muon_sf_transform,
            "key": "NUM_MediumID_DEN_TrackerMuons",
            "use_correctionlib": True,
            "op": "mult",
            "nominal_idx": "sf",
            "up_and_down_idx": ["systup", "systdown"],
        },
        # ------------------------------------------------------------------
        # Muon ISO scale factor (Tight relative ISO) - correlated across years
        # Signature: (abseta, pt, systematic)
        # ------------------------------------------------------------------
        {
            "name": "muon_iso_sf",
            "file": get_correction_file(year, "muon_Z"),
            "type": "event",
            "args": [ObjVar("Muon", "eta"), ObjVar("Muon", "pt"), SYS],
            "transform_in": muon_sf_transform,
            "key": "NUM_TightRelIso_DEN_MediumID",
            "use_correctionlib": True,
            "op": "mult",
            "nominal_idx": "sf",
            "up_and_down_idx": ["systup", "systdown"],
        },
        # ------------------------------------------------------------------
        # Muon trigger scale factor (Mu50) - correlated across years
        # Signature: (abseta, pt, systematic)
        # ------------------------------------------------------------------
        {
            "name": "muon_trigger_sf",
            "file": get_correction_file(year, "muon_Z"),
            "type": "event",
            "args": [ObjVar("Muon", "eta"), ObjVar("Muon", "pt"), SYS],
            "transform_in": muon_sf_transform,
            "key": (
                "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose" if "2016" not in year 
                else "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose"
            ),
            "use_correctionlib": True,
            "op": "mult",
            "nominal_idx": "nominal",
            "up_and_down_idx": ["systup", "systdown"],
        },
    ]

    # ------------------------------------------------------------------
    # B-tagging scale factors (DeepCSV shape)
    # Signature: (systematic, flavor, abseta, pt, discriminant)
    # ------------------------------------------------------------------
    btag_args = [
        SYS,
        ObjVar("Jet", "hadronFlavour"),
        ObjVar("Jet", "eta"),
        ObjVar("Jet", "pt"),
        ObjVar("Jet", "btagDeepB"),
    ]

    # hf/lf systematics - apply to b and light jets (c-jets get SF=1)
    for syst in ["hf", "lf"]:
        corrections.append({
            "name": f"btag_{syst}",
            "file": get_correction_file(year, "btagging"),
            "type": "event",
            "args": btag_args,
            "transform_in": btag_hf_transform_in,
            "transform_out": btag_hf_transform_out,
            "reduce": "prod",
            "key": "deepJet_shape",
            "use_correctionlib": True,
            "op": "mult",
            "nominal_idx": "central",
            "up_and_down_idx": [f"up_{syst}", f"down_{syst}"],
        })

    # cferr systematics - apply to c jets only (b/light jets get SF=1)
    for syst in ["cferr1", "cferr2"]:
        corrections.append({
            "name": f"btag_{syst}",
            "file": get_correction_file(year, "btagging"),
            "type": "event",
            "args": btag_args,
            "transform_in": btag_cferr_transform_in,
            "transform_out": btag_cferr_transform_out,
            "reduce": "prod",
            "key": "deepJet_shape",
            "use_correctionlib": True,
            "op": "mult",
            "nominal_idx": "central",
            "up_and_down_idx": [f"up_{syst}", f"down_{syst}"],
        })

    # hfstats/lfstats systematics - decorrelated by year, apply to b and light jets
    for syst in ["hfstats1", "hfstats2", "lfstats1", "lfstats2"]:
        corrections.append({
            "name": f"btag_{syst}_{year}",
            "file": get_correction_file(year, "btagging"),
            "type": "event",
            "args": btag_args,
            "transform_in": btag_hf_transform_in,
            "transform_out": btag_hf_transform_out,
            "reduce": "prod",
            "key": "deepJet_shape",
            "use_correctionlib": True,
            "op": "mult",
            "nominal_idx": "central",
            "up_and_down_idx": [f"up_{syst}", f"down_{syst}"],
        })

    return corrections


# ==============================================================================
#  Systematics Configuration
# ==============================================================================

def _get_systematics_for_year(year: str) -> list:
    """
    Get systematics configuration for a specific year.

    Currently empty as all systematics are handled via corrections.
    JEC/JER systematics will be added here in the future.

    Parameters
    ----------
    year : str
        Year identifier (2016preVFP, 2017, 2018)

    Returns
    -------
    list
        List of systematic configuration dictionaries
    """
    return []


# ==============================================================================
#  Build Year-Keyed Configuration Dicts
# ==============================================================================

def build_corrections_config() -> dict:
    """
    Build corrections configuration dictionary keyed by year.

    Returns
    -------
    dict
        Dictionary with years as keys and correction lists as values
    """
    return {year: _get_corrections_for_year(year) for year in YEARS}


def build_systematics_config() -> dict:
    """
    Build systematics configuration dictionary keyed by year.

    Returns
    -------
    dict
        Dictionary with years as keys and systematics lists as values
    """
    return {year: _get_systematics_for_year(year) for year in YEARS}


# Pre-build the configs for import
corrections_config = build_corrections_config()
systematics_config = build_systematics_config()
