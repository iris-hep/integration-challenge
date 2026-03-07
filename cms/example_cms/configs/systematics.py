"""
Systematic uncertainties and corrections configuration for the Z-prime ttbar analysis.

This module contains:
- Year-aware correction file paths
- Corrections configuration (scale factors, pileup weights) keyed by year
- Systematic uncertainties configuration

Correction sources:
- Scale factors: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun2LegacyAnalysis
- POG JSON files: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/
- POG Corrections: https://cms-analysis-corrections.docs.cern.ch/

Systematics naming convention:
- Correlated across years: simple name (e.g., "muon_id_sf", "btag_hf")
- Decorrelated by year: name_YEAR (e.g., "pileup_2017", "btag_hfstats1_2017")
"""

import awkward as ak
import correctionlib
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

# JES sys-string templates accepted by the deepJet_shape btag evaluator.
# Used to build btag JES uncertainty sources with varies_with linkage.
# Convention: {direction}_<name> where <name> matches the JEC source name.
# Only individual sources (not regrouped or total) — matches JecConfigAK4.json Full set.
# Year-correlated sources (17 individual, same across all years):
_BTAG_JES_CORRELATED = [
    "{direction}_jesAbsoluteMPFBias",
    "{direction}_jesAbsoluteScale",
    "{direction}_jesFlavorQCD",
    "{direction}_jesFragmentation",
    "{direction}_jesPileUpDataMC",
    "{direction}_jesPileUpPtBB",
    "{direction}_jesPileUpPtEC1",
    "{direction}_jesPileUpPtEC2",
    "{direction}_jesPileUpPtHF",
    "{direction}_jesPileUpPtRef",
    "{direction}_jesRelativeBal",
    "{direction}_jesRelativeFSR",
    "{direction}_jesRelativeJERHF",
    "{direction}_jesRelativePtBB",
    "{direction}_jesRelativePtHF",
    "{direction}_jesSinglePionECAL",
    "{direction}_jesSinglePionHCAL",
]

# Year-decorrelated sources (10 individual, suffixed with year)
_BTAG_JES_DECORRELATED_BASES = [
    "jesAbsoluteStat",
    "jesRelativeJEREC1",
    "jesRelativeJEREC2",
    "jesRelativePtEC1",
    "jesRelativePtEC2",
    "jesRelativeSample",
    "jesRelativeStatEC",
    "jesRelativeStatFSR",
    "jesRelativeStatHF",
    "jesTimePtEta",
]


# ==============================================================================
#  JEC Configuration
# ==============================================================================

# MC campaign names per year (from JecConfigAK4.json tag names)
JEC_CAMPAIGNS = {
    "2016preVFP": "Summer19UL16APV_V7_MC",
    "2016postVFP": "Summer19UL16_V7_MC",
    "2017": "Summer19UL17_V5_MC",
    "2018": "Summer19UL18_V5_MC",
}

# Individual JEC uncertainty sources from the Full set (JecConfigAK4.json).
# Evaluator names: {Campaign}_{SourceName}_AK4PFchs
# Correlated across years (17 sources — no year suffix in CMS name):
_JEC_UNC_CORRELATED = [
    "AbsoluteMPFBias", "AbsoluteScale", "FlavorQCD", "Fragmentation",
    "PileUpDataMC", "PileUpPtBB", "PileUpPtEC1", "PileUpPtEC2",
    "PileUpPtHF", "PileUpPtRef", "RelativeBal", "RelativeFSR",
    "RelativeJERHF", "RelativePtBB", "RelativePtHF",
    "SinglePionECAL", "SinglePionHCAL",
]

# Decorrelated by year (10 sources — CMS name has _YEAR suffix):
_JEC_UNC_DECORRELATED = [
    "AbsoluteStat", "RelativeJEREC1", "RelativeJEREC2",
    "RelativePtEC1", "RelativePtEC2", "RelativeSample",
    "RelativeStatEC", "RelativeStatFSR", "RelativeStatHF",
    "TimePtEta",
]

# Module-level cache for correctionlib CorrectionSets
_jec_correction_sets = {}


def _get_correctionset(file_path):
    if file_path not in _jec_correction_sets:
        _jec_correction_sets[file_path] = correctionlib.CorrectionSet.from_file(
            file_path
        )
    return _jec_correction_sets[file_path]
    
    
def _get_jec_evaluator(ject_corrset, evaluator_key):
    """Lazily load and cache correctionlib CorrectionSets."""
    return ject_corrset[evaluator_key]



def _make_jec_nominal_func(ject_corrset, l1_key, l2_key):
    """Sequential L1FastJet + L2Relative JEC: (area, eta, pt, rho) -> total factor.

    Applies L1FastJet(area, eta, pt, rho) then L2Relative(eta, corrected_pt)
    and returns the combined multiplicative factor C_L1 * C_L2.

    rho is event-level (one value per event) while area/eta/pt are per-jet
    (jagged). Broadcasting expands rho to match the per-jet structure before
    flattening for correctionlib evaluation.
    """
    def func(area, eta, pt, rho):
        l1_eval = _get_jec_evaluator(ject_corrset, l1_key)
        l2_eval = _get_jec_evaluator(ject_corrset, l2_key)
        counts = ak.num(eta)
        rho_broadcast, _ = ak.broadcast_arrays(rho, eta)

        flat_area = np.asarray(ak.flatten(area), dtype=np.float64)
        flat_eta = np.asarray(ak.flatten(eta), dtype=np.float64)
        flat_pt = np.asarray(ak.flatten(pt), dtype=np.float64)
        flat_rho = np.asarray(ak.flatten(rho_broadcast), dtype=np.float64)

        # L1FastJet: remove pileup contribution
        c_l1 = l1_eval.evaluate(flat_area, flat_eta, flat_pt, flat_rho)
        pt_l1 = flat_pt * c_l1

        # L2Relative: flatten eta response (uses L1-corrected pt)
        c_l2 = l2_eval.evaluate(flat_eta, pt_l1)

        return ak.unflatten(c_l1 * c_l2, counts)
    return func


def _make_jec_unc_func(ject_corrset, evaluator_key, sign):
    """JEC uncertainty: (eta, pt) -> (1 +/- delta). sign: +1 for up, -1 for down."""
    def func(eta, pt):
        evaluator = _get_jec_evaluator(ject_corrset, evaluator_key)
        counts = ak.num(eta)
        flat_eta = np.asarray(ak.flatten(eta), dtype=np.float64)
        flat_pt = np.asarray(ak.flatten(pt), dtype=np.float64)
        delta = evaluator.evaluate(flat_eta, flat_pt)
        return ak.unflatten(1.0 + sign * delta, counts)
    return func


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


def btag_transform_in(hadronFlavour, eta, pt, disc):
    """Nominal btag transform: abs(eta), safe flavor for invalid jets.

    Used for the "central" evaluation where all flavors (0, 4, 5) are valid.
    Invalid jets get flavor 0 to avoid correctionlib errors; transform_out
    masks them to SF=1.0.
    """
    valid = _btag_valid_mask(eta, pt, disc)
    safe_flavor = ak.where(valid, hadronFlavour, 0)
    return (safe_flavor, np.abs(eta), pt, disc)


def btag_transform_out(sf, hadronFlavour, eta, pt, disc):
    """Nominal btag output: invalid jets get SF=1.0."""
    valid = _btag_valid_mask(eta, pt, disc)
    return ak.where(valid, sf, 1.0)


def btag_hf_transform_in(hadronFlavour, eta, pt, disc):
    """hf/lf/jes btag transform: fake c-jets as light for correctionlib eval.

    The evaluator only accepts flavors 0 and 5 for hf/lf/jes systematics.
    C-jets (flavor 4) and invalid jets are faked as light (flavor 0);
    transform_out masks them back to SF=1.0.
    """
    valid = _btag_valid_mask(eta, pt, disc)
    skip = (hadronFlavour == 4) | ~valid
    fake_flavor = ak.where(skip, 0, hadronFlavour)
    return (fake_flavor, np.abs(eta), pt, disc)


def btag_hf_transform_out(sf, hadronFlavour, eta, pt, disc):
    """hf/lf/jes btag output: c-jets and invalid jets get SF=1.0."""
    valid = _btag_valid_mask(eta, pt, disc)
    skip = (hadronFlavour == 4) | ~valid
    return ak.where(skip, 1.0, sf)


def btag_cferr_transform_in(hadronFlavour, eta, pt, disc):
    """cferr btag transform: fake non-c jets as c for correctionlib eval.

    The evaluator only accepts flavor 4 for cferr systematics.
    Non-c jets (flavors 0, 5) and invalid jets are faked as c (flavor 4);
    transform_out masks them back to SF=1.0.
    """
    valid = _btag_valid_mask(eta, pt, disc)
    skip = (hadronFlavour != 4) | ~valid
    fake_flavor = ak.where(skip, 4, hadronFlavour)
    return (fake_flavor, np.abs(eta), pt, disc)


def btag_cferr_transform_out(sf, hadronFlavour, eta, pt, disc):
    """cferr btag output: non-c jets and invalid jets get SF=1.0."""
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
    # JEC file and campaign for this year
    jec_file = get_correction_file(year, "JEC")
    ject_corrset = _get_correctionset(jec_file)
    
    campaign = JEC_CAMPAIGNS[year]
    year_suffix = "2016" if year.startswith("2016") else year

    # ------------------------------------------------------------------
    # JEC: one correction with nominal + 27 uncertainty sources
    # Nominal: L1FastJet + L2Relative, (area, eta, pt, rho) -> factor
    # Sources: (eta, pt) -> (1 +/- delta), applied as delta on nominal
    # ------------------------------------------------------------------
    jec_sources = []

    # Correlated across years (17 sources)
    for source in _JEC_UNC_CORRELATED:
        evaluator_key = f"{campaign}_{source}_AK4PFchs"
        jec_sources.append({
            "name": f"jes{source}",
            "args": [ObjVar("Jet", "eta"), ObjVar("Jet", "pt")],
            "up_function": _make_jec_unc_func(ject_corrset, evaluator_key, +1),
            "down_function": _make_jec_unc_func(ject_corrset, evaluator_key, -1),
            "is_delta": True,
        })

    # Decorrelated by year (10 sources)
    for source in _JEC_UNC_DECORRELATED:
        evaluator_key = f"{campaign}_{source}_AK4PFchs"
        jec_sources.append({
            "name": f"jes{source}_{year_suffix}",
            "args": [ObjVar("Jet", "eta"), ObjVar("Jet", "pt")],
            "up_function": _make_jec_unc_func(ject_corrset, evaluator_key, +1),
            "down_function": _make_jec_unc_func(ject_corrset, evaluator_key, -1),
            "is_delta": True,
        })

    corrections = [
        {
            "name": "jec",
            "type": "object",
            "use_correctionlib": False,
            "target": ObjVar("Jet", "pt"),
            "args": [
                ObjVar("Jet", "area"), ObjVar("Jet", "eta"),
                ObjVar("Jet", "pt"), ObjVar("event", "fixedGridRhoFastjetAll"),
            ],
            "op": "mult",
            "nominal_function": _make_jec_nominal_func(
                ject_corrset,
                f"{campaign}_L1FastJet_AK4PFchs",
                f"{campaign}_L2Relative_AK4PFchs",
            ),
            "uncertainty_sources": jec_sources,
        },
    ]

    corrections += [
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
            "uncertainty_sources": [
                {"name": f"pileup_{year}", "up_and_down_idx": ["up", "down"]},
            ],
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
            "nominal_idx": "nominal",
            "uncertainty_sources": [
                {"name": "muon_id_sf", "up_and_down_idx": ["systup", "systdown"]},
            ],
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
            "nominal_idx": "nominal",
            "uncertainty_sources": [
                {"name": "muon_iso_sf", "up_and_down_idx": ["systup", "systdown"]},
            ],
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
            "uncertainty_sources": [
                {"name": "muon_trigger_sf", "up_and_down_idx": ["systup", "systdown"]},
            ],
        },
    ]

    # ------------------------------------------------------------------
    # B-tagging scale factors (deepJet_shape)
    # One correction with multiple uncertainty sources.
    # Real hadronFlavour passed to evaluator for all sources — no flavor
    # remapping. The evaluator returns SF(central) for irrelevant flavors.
    # Signature: (systematic, flavor, abseta, pt, discriminant)
    # ------------------------------------------------------------------
    btag_args = [
        SYS,
        ObjVar("Jet", "hadronFlavour"),
        ObjVar("Jet", "eta"),
        ObjVar("Jet", "pt"),
        ObjVar("Jet", "btagDeepB"),
    ]

    # hf/lf sources: evaluator accepts flavors 0 (light) and 5 (bottom) only
    _hf_transforms = {
        "transform_in": btag_hf_transform_in,
        "transform_out": btag_hf_transform_out,
    }
    # cferr sources: evaluator accepts flavor 4 (charm) only
    _cferr_transforms = {
        "transform_in": btag_cferr_transform_in,
        "transform_out": btag_cferr_transform_out,
    }

    btag_sources = [
        {"name": "btag_hf", "up_and_down_idx": ["up_hf", "down_hf"],
         **_hf_transforms},
        {"name": "btag_lf", "up_and_down_idx": ["up_lf", "down_lf"],
         **_hf_transforms},
        {"name": "btag_cferr1", "up_and_down_idx": ["up_cferr1", "down_cferr1"],
         **_cferr_transforms},
        {"name": "btag_cferr2", "up_and_down_idx": ["up_cferr2", "down_cferr2"],
         **_cferr_transforms},
        {"name": f"btag_hfstats1_{year}", "up_and_down_idx": ["up_hfstats1", "down_hfstats1"],
         **_hf_transforms},
        {"name": f"btag_hfstats2_{year}", "up_and_down_idx": ["up_hfstats2", "down_hfstats2"],
         **_hf_transforms},
        {"name": f"btag_lfstats1_{year}", "up_and_down_idx": ["up_lfstats1", "down_lfstats1"],
         **_hf_transforms},
        {"name": f"btag_lfstats2_{year}", "up_and_down_idx": ["up_lfstats2", "down_lfstats2"],
         **_hf_transforms},
    ]

    # JES-linked btag sources: varied together with JEC object sources
    # JES sources also only accept flavors 0 and 5
    for template in _BTAG_JES_CORRELATED:
        jec_name = template.replace("{direction}_", "")
        btag_sources.append({
            "name": f"btag_{jec_name}",
            "up_and_down_idx": [template.format(direction="up"), template.format(direction="down")],
            "varies_with": [jec_name],
            **_hf_transforms,
        })
    # 2016preVFP btag file uses bare evaluator strings for individual
    # decorrelated JES sources; 2016postVFP/2017/2018 include the year suffix.
    btag_jes_eval_suffix = "" if year == "2016preVFP" else f"_{year_suffix}"

    for base in _BTAG_JES_DECORRELATED_BASES:
        jec_name = f"{base}_{year_suffix}"
        eval_name = f"{base}{btag_jes_eval_suffix}"
        btag_sources.append({
            "name": f"btag_{jec_name}",
            "up_and_down_idx": [f"up_{eval_name}", f"down_{eval_name}"],
            "varies_with": [jec_name],
            **_hf_transforms,
        })

    corrections.append({
        "name": "btag",
        "file": get_correction_file(year, "btagging"),
        "type": "event",
        "args": btag_args,
        "transform_in": btag_transform_in,
        "transform_out": btag_transform_out,
        "reduce": "prod",
        "key": "deepJet_shape",
        "use_correctionlib": True,
        "op": "mult",
        "nominal_idx": "central",
        "uncertainty_sources": btag_sources,
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
