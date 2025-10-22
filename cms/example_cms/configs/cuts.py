import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection


# ===================
# Select good run data
# ===================
def lumi_mask(
    lumifile: str, run: ak.Array, lumiBlock: ak.Array) -> ak.Array:
    """
    Create a boolean mask selecting events that pass the good run/lumi criteria.
    https://github.com/cms-opendata-workshop/workshop2024-lesson-event-selection/blob/main/instructors/dpoa_workshop_utilities.py

    This function compares the `(run, lumiBlock)` pairs in the dataset to the
    certified good luminosity sections provided in a JSON file (e.g., from
    CMS Golden JSON).

    Parameters
    ----------
    lumifile : str
        Path to the JSON file defining the certified good lumi sections.
    run : ak.Array
        Run numbers for each event in the dataset.
    lumiBlock : ak.Array
        Luminosity block numbers for each event in the dataset.
    verbose : bool, optional
        If True, prints additional debug information.

    Returns
    -------
    ak.Array
        A boolean array (same length as `run`) indicating good events.
    """
    # -----------------------------
    # Load good lumi sections JSON
    # -----------------------------
    good_lumi_sections = ak.from_json(open(lumifile, "rb"))

    # Extract good run numbers (as integers)
    good_runs = np.array(good_lumi_sections.fields).astype(int)

    # -----------------------------
    # Build array of lumi blocks
    # -----------------------------
    good_blocks = [
        good_lumi_sections[field] for field in good_lumi_sections.fields
    ]
    all_good_blocks = ak.Array(good_blocks)

    # -----------------------------
    # Match run numbers to good runs
    # -----------------------------
    def find_indices(arr1: np.ndarray, arr2: ak.Array) -> ak.Array:
        arr1_np = np.asarray(ak.to_numpy(arr1))
        arr2_np = np.asarray(ak.to_numpy(arr2))

        # Sort arr1 and track indices
        sorter = np.argsort(arr1_np)
        sorted_arr1 = arr1_np[sorter]

        # Find insertion positions of arr2 elements into arr1
        pos = np.searchsorted(sorted_arr1, arr2_np)

        # Validate matches
        valid = (pos < len(arr1_np)) & (sorted_arr1[pos] == arr2_np)

        # Build result array
        out = np.full(len(arr2_np), -1, dtype=int)
        out[valid] = sorter[pos[valid]]
        return ak.Array(out)

    good_run_indices = find_indices(good_runs, run)

    # -----------------------------
    # Compute per-event lumi block diffs
    # -----------------------------

    # Calculate (event lumi - good lumi) for matched run
    diff = lumiBlock - all_good_blocks[good_run_indices]

    # -----------------------------
    # Evaluate mask from differences
    # -----------------------------
    # For a lumi to be valid, it must lie within one of the good ranges
    # So the difference must have both positive and negative signs
    prod_diff = ak.prod(diff, axis=2)
    mask = ak.any(prod_diff <= 0, axis=1)

    return mask


# ===================
# Selection which is applied to all regions
# ===================
def Zprime_baseline(
    muons: ak.Array, jets: ak.Array, fatjets: ak.Array, met: ak.Array
) -> PackedSelection:
    """
    Define baseline selection criteria used across all analysis regions.

    Parameters
    ----------
    muons : ak.Array
        Muon collection for the event.
    jets : ak.Array
        Jet collection (not used in baseline, included for interface consistency).
    fatjets : ak.Array
        Fat jet collection (not used here).
    met : ak.Array
        Missing transverse energy (MET), unused here but accepted for API uniformity.

    Returns
    -------
    PackedSelection
        Bitmask selection object with baseline cuts.
    """
    selections = PackedSelection(dtype="uint64")

    # ---------------------
    # Require exactly 1 muon
    # ---------------------
    selections.add("exactly_1mu", ak.num(muons, axis=1) == 1)

    # ---------------------
    # Baseline composite mask
    # ---------------------
    selections.add("baseline", selections.all("exactly_1mu"))

    return selections


# ===================
# Selection which will not be optimised from WS
# ===================
def Zprime_hardcuts(
    muons: ak.Array, jets: ak.Array, fatjets: ak.Array
) -> PackedSelection:
    """
    Define non-optimizable kinematic cuts.
    These hard cuts + baseline cuts ensure observable calculations
    can work without errors.

    Parameters
    ----------
    muons : ak.Array
        Muon collection.
    jets : ak.Array
        Jet collection.
    fatjets : ak.Array
        Fat jet collection.

    Returns
    -------
    PackedSelection
        Bitmask selection object containing hard selection criteria.
    """
    selections = PackedSelection(dtype="uint64")

    # ---------------------
    # Object count requirements
    # ---------------------
    selections.add("exactly_1mu", ak.count(muons.pt, axis=1) == 1)
    selections.add("atleast_2jet", ak.count(jets.pt, axis=1) > 1)
    selections.add("atleast_1fj", ak.count(fatjets.pt, axis=1) > 0)

    # ---------------------
    # Composite region selection
    # ---------------------
    selections.add(
        "Zprime_channel",
        selections.all("exactly_1mu", "atleast_2jet", "atleast_1fj"),
    )

    return selections


def Zprime_hardcuts_no_fj(
    muons: ak.Array,
    jets: ak.Array,
) -> PackedSelection:
    """
    Define non-optimizable kinematic cuts.
    These hard cuts + baseline cuts ensure observable calculations
    can work without errors.

    Parameters
    ----------
    muons : ak.Array
        Muon collection.
    jets : ak.Array
        Jet collection.

    Returns
    -------
    PackedSelection
        Bitmask selection object containing hard selection criteria.
    """
    selections = PackedSelection(dtype="uint64")

    # ---------------------
    # Object count requirements
    # ---------------------
    selections.add("exactly_1mu", ak.count(muons.pt, axis=1) == 1)
    selections.add("atleast_2jet", ak.count(jets.pt, axis=1) > 1)

    # ---------------------
    # Composite region selection
    # ---------------------
    selections.add(
        "Zprime_channel_no_fj", selections.all("exactly_1mu", "atleast_2jet")
    )

    return selections


# ===================
# All selection from workshop
# ===================
# -----------------------------------------------------------------------------
# Zprime Softcuts (non-JAX, Workshop Version) — PackedSelection
# -----------------------------------------------------------------------------
def Zprime_workshop_cuts(
    muons: ak.Array, jets: ak.Array, fatjets: ak.Array, met: ak.Array
) -> PackedSelection:
    """
    Apply all selection cuts for Zprime analysis from CMS workshop 2024.

    These represent a non-optimized, physics-motivated set of cuts used during
    prototyping and initial analysis, and match the OpenData workshop results.

    Parameters
    ----------
    muons : ak.Array
        Muon collection.
    jets : ak.Array
        Jet collection.
    fatjets : ak.Array
        Fat jet collection.
    met : ak.Array
        Missing transverse energy (MET).

    Returns
    -------
    PackedSelection
        Bitmask selection object containing workshop softcut flags.
    """
    selections = PackedSelection(dtype="uint64")

    # ---------------------
    # Basic object and MET cuts
    # ---------------------
    selections.add("exactly_1mu", ak.num(muons, axis=1) == 1)
    selections.add(
        "atleast_1b",
        ak.sum((jets.btagDeepB > 0.5) & (jets.jetId >= 4), axis=1) > 0,
    )
    selections.add("met_cut", met.pt > 50)

    # ---------------------
    # Leptonic HT (muon + MET)
    # ---------------------
    lep_ht = muons.pt + met.pt
    selections.add("muon_ht", ak.sum(lep_ht > 150.0, axis=1) == 1)

    # ---------------------
    # Fatjet topology cut
    # ---------------------
    selections.add(
        "exactly_1fatjet",
        ak.sum(
            (fatjets.particleNet_TvsQCD > 0.5) & (fatjets.pt > 500.0), axis=1
        )
        == 1,
    )

    # ---------------------
    # Composite softcut channel
    # ---------------------
    selections.add(
        "Zprime_channel",
        selections.all(
            "exactly_1mu",
            "atleast_1b",
            "met_cut",
            "muon_ht",
            "exactly_1fatjet",
        ),
    )

    return selections

# ===========================================================
# Zprime Selection Regions Based on Physics Paper Definitions
# ===========================================================
def Zprime_softcuts_nonjax_paper(
    muons: ak.Array, jets: ak.Array, fatjets: ak.Array, met: ak.Array
) -> PackedSelection:
    """
    Paper-based soft selection cuts for the Zprime analysis.

    Implements kinematic selections and object ID cuts as described in
    section 7.2 of https://arxiv.org/pdf/1810.05905, including lepton
    isolation, jet pT thresholds, and top-tagging constraints.

    Parameters
    ----------
    muons : ak.Array
        Muon candidates.
    jets : ak.Array
        Jet candidates.
    fatjets : ak.Array
        Large-R jet candidates.
    met : ak.Array
        Missing transverse energy.

    Returns
    -------
    PackedSelection
        Bitmask selection object with all softcut flags.
    """
    selections = PackedSelection(dtype="uint64")

    lep_ht = muons.pt + met.pt

    # Compute deltaR and pTrel between muon and nearest jet
    muon_jets = ak.cartesian([muons, jets], nested=True)
    muon_in_pair, jet_in_pair = ak.unzip(muon_jets)
    deltaR = muon_in_pair.deltaR(jet_in_pair)
    min_deltaR = ak.min(deltaR, axis=1)

    closest_jet_idx = ak.argmin(deltaR, axis=1, keepdims=True)
    closest_jet = jet_in_pair[closest_jet_idx]
    delta_angle = muons.deltaangle(closest_jet)
    pt_rel = muons.p * ak.sin(delta_angle)

    selections.add("atleast_1b", ak.sum(jets.btagDeepB > 0.5, axis=1) > 0)
    selections.add("met_cut", met.pt > 50)
    selections.add("lep_ht_cut", ak.fill_none(ak.firsts(lep_ht) > 150, False))
    selections.add(
        "lepton_2d",
        ak.fill_none(
            ak.sum((min_deltaR > 0.4) | (pt_rel > 25.0), axis=1) > 0, False
        ),
    )
    selections.add("at_least_1_150gev_jet", ak.sum(jets.pt > 150, axis=1) > 0)
    selections.add("at_least_1_50gev_jet", ak.sum(jets.pt > 50, axis=1) > 0)
    selections.add(
        "nomore_than_1_top_tagged_jet",
        ak.sum(fatjets.particleNet_TvsQCD > 0.5, axis=1) < 2,
    )

    return selections


def Zprime_softcuts_SR_tag(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
    ttbar_reco: ak.Array,
    mva: ak.Array,
) -> PackedSelection:
    """
    Signal Region (1-tag) selection following Sec. 7.2 of the Zprime paper.

    Parameters
    ----------
    muons : ak.Array
    jets : ak.Array
    fatjets : ak.Array
    met : ak.Array
    ttbar_reco : ak.Array
        ttbar reconstruction output including chi2.
    mva : ak.Array
        Neural net discriminant scores.

    Returns
    -------
    PackedSelection
    """
    selections = PackedSelection(dtype="uint64")
    lep_ht = muons.pt + met.pt

    selections.add("atleast_1b", ak.sum(jets.btagDeepB > 0.5, axis=1) > 0)
    selections.add("met_cut", met.pt > 50)
    selections.add("lep_ht_cut", ak.fill_none(ak.firsts(lep_ht) > 150, False))
    selections.add(
        "exactly_1fatjet",
        ak.sum(fatjets.particleNet_TvsQCD > 0.5, axis=1) == 1,
    )
    selections.add("chi2_cut", ttbar_reco.chi2 < 30.0)
    selections.add("nn_score", mva.nn_score >= 0.5)

    return selections


def Zprime_softcuts_SR_notag(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
    ttbar_reco: ak.Array,
    mva: ak.Array,
) -> PackedSelection:
    """
    Signal Region (0-tag) selection (no top-tagged fatjets).

    Parameters
    ----------
    muons : ak.Array
    jets : ak.Array
    fatjets : ak.Array
    met : ak.Array
    ttbar_reco : ak.Array
    mva : ak.Array

    Returns
    -------
    PackedSelection
    """
    selections = PackedSelection(dtype="uint64")
    lep_ht = muons.pt + met.pt

    selections.add("atleast_1b", ak.sum(jets.btagDeepB > 0.5, axis=1) > 0)
    selections.add("met_cut", met.pt > 50)
    selections.add("lep_ht_cut", ak.fill_none(ak.firsts(lep_ht) > 150, False))
    selections.add("exactly_0fatjet", ak.num(fatjets) == 0)
    selections.add("chi2_cut", ttbar_reco.chi2 < 30.0)
    selections.add("nn_score", mva.nn_score >= 0.5)

    return selections


def Zprime_softcuts_CR1(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
    ttbar_reco: ak.Array,
    mva: ak.Array,
) -> PackedSelection:
    """
    Control Region 1 (W+jets enriched) selection as in paper Sec. 7.2.

    Returns
    -------
    PackedSelection
    """
    selections = PackedSelection(dtype="uint64")
    lep_ht = muons.pt + met.pt

    selections.add("atleast_1b", ak.sum(jets.btagDeepB > 0.5, axis=1) > 0)
    selections.add("met_cut", met.pt > 50)
    selections.add("lep_ht_cut", ak.fill_none(ak.firsts(lep_ht) > 150, False))
    selections.add("chi2_cut", ttbar_reco.chi2 < 30.0)
    selections.add("nn_score", mva.nn_score < -0.75)

    return selections


def Zprime_softcuts_CR2(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
    ttbar_reco: ak.Array,
    mva: ak.Array,
) -> PackedSelection:
    """
    Control Region 2 (ttbar enriched) selection.

    Returns
    -------
    PackedSelection
    """
    selections = PackedSelection(dtype="uint64")
    lep_ht = muons.pt + met.pt

    selections.add("atleast_1b", ak.sum(jets.btagDeepB > 0.5, axis=1) > 0)
    selections.add("met_cut", met.pt > 50)
    selections.add("lep_ht_cut", ak.fill_none(ak.firsts(lep_ht) > 150, False))
    selections.add("chi2_cut", ttbar_reco.chi2 < 30.0)
    selections.add(
        "nn_score_range", (mva.nn_score > 0.0) & (mva.nn_score < 0.5)
    )

    return selections
