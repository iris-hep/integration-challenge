import logging

import awkward as ak
import numba
import numpy as np
import vector

logger = logging.getLogger(__name__)

# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()


def get_mtt(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
) -> ak.Array:
    """
    Calculate the invariant mass of the top quark pair (m_tt) from input objects.

    Parameters
    ----------
    muons : ak.Array
        Array of muon candidates.
    jets : ak.Array
        Array of jet candidates.
    fatjets : ak.Array
        Array of fatjet candidates.
    met : ak.Array
        Array of missing transverse energy (MET) information.

    Returns
    -------
    ak.Array
        Flattened array of reconstructed m_tt values per event.
    """

    # jets = jets[(jets.btagDeepB > 0.5) & (jets.jetId > 4)]
    jets = jets[:, 0]  # only the first jet per event
    fatjets = fatjets[:, 0]
    muons = muons[:, 0]  # only the first muon per event
    # p4mu,p4fj,p4j,p4met = ak.unzip(ak.cartesian([muons, fatjets, jets, met]))
    # Convert to 4-vectors
    p4mu, p4fj, p4j = [
        ak.zip(
            {"pt": o.pt, "eta": o.eta, "phi": o.phi, "mass": o.mass},
            with_name="Momentum4D",
        )
        for o in [muons, fatjets, jets]
    ]
    p4met = ak.zip(
        {
            "pt": met.pt,
            "eta": 0 * met.pt,
            "phi": met.phi,
            "mass": 0,
        },
        with_name="Momentum4D",
    )
    p4tot = p4mu + p4fj + p4j + p4met
    return p4tot.mass


def solve_neutrino_pz(lepton, met, mW=80.4):
    """
    Compute neutrino pz solutions from lepton and MET using W mass constraint.

    Parameters
    ----------
    lepton : ak.Array
        Lepton 4-vector (pt, eta, phi, mass).
    met : ak.Array
        MET as a 4-vector (pt, phi); eta=0, mass=0.

    Returns
    -------
    ak.Array
        Array of shape (n_events, 2) with pz solutions (real or complex).
        If no real solution, return real part of complex root.
    """
    px_l = lepton.pt * np.cos(lepton.phi)
    py_l = lepton.pt * np.sin(lepton.phi)
    pz_l = lepton.pz
    e_l = lepton.energy

    px_nu = met.pt * np.cos(met.phi)
    py_nu = met.pt * np.sin(met.phi)

    pt_l_sq = lepton.pt**2
    pt_nu_sq = met.pt**2

    # mu = (mW^2 / 2) + px_l * px_nu + py_l * py_nu
    mu = (mW**2) / 2 + px_l * px_nu + py_l * py_nu

    a = mu * pz_l / pt_l_sq
    A = (mu**2) * (pz_l**2)
    B = (e_l**2) * (pt_l_sq * pt_nu_sq)
    C = (mu**2) * pt_l_sq
    discriminant = A - B + C

    sqrt_discriminant = ak.where(
        discriminant >= 0, np.sqrt(discriminant), np.sqrt(-discriminant) * 1j
    )
    pz_nu_1 = a + sqrt_discriminant / pt_l_sq
    pz_nu_2 = a - sqrt_discriminant / pt_l_sq

    return pz_nu_1, pz_nu_2


def build_leptonic_tops(muon, met, lepjet, pz1, pz2):
    """
    Return list of leptonic top 4-vectors per event based on neutrino pz solutions.
    """

    # ============================================================
    # Solve for neutrino
    # ============================================================
    def make_nu(pz):
        return ak.zip(
            {
                "pt": met.pt,
                "phi": met.phi,
                "eta": ak.zeros_like(met.pt),
                "mass": ak.zeros_like(met.pt),
                "pz": ak.real(pz),
            },
            with_name="Momentum4D",
        )

    def ak_is_real(array):
        return ak.real(array) == array

    pz1 = ak.Array(pz1)
    pz2 = ak.Array(pz2)

    # Count real solutions
    disc = (pz1 - pz2) ** 2
    two_real = ak_is_real(pz1) & (disc != 0)

    # Always build all 3 cases per element — rely on masking
    nu1 = make_nu(pz1)
    nu2 = make_nu(pz2)

    lep_top1 = muon + lepjet + nu1
    lep_top2 = muon + lepjet + nu2

    lep_top1_masked = ak.mask(lep_top1, two_real)
    lep_top2_masked = ak.mask(lep_top2, two_real)

    lep_top_candidates = ak.where(
        ak.firsts(two_real, axis=1),
        ak.concatenate([lep_top1_masked, lep_top2_masked], axis=1),
        lep_top1,
    )
    return lep_top_candidates


@numba.njit
def build_index(index_values, a_lengths, b_offsets):
    """
    Fill `index_values` such that for each sublist in `a`, we repeat the elements
    from the corresponding sublist in `b` cyclically to match its length.

    Parameters
    ----------
    index_values : np.ndarray
        Output array to fill with indices into flattened `b`.
    a_lengths : np.ndarray
        Length of each sublist in `a`.
    b_offsets : np.ndarray
        Offsets defining the start/stop positions of each sublist in `b`.
    """
    pos = 0
    for i in range(len(a_lengths)):
        len_a_i = a_lengths[i]  # number of entries in a[i]
        start_b_i = b_offsets[i]  # start of b[i] in flat_b
        stop_b_i = b_offsets[i + 1]  # end of b[i]
        len_b_i = stop_b_i - start_b_i  # number of entries in b[i]

        if len_b_i == 0:
            if len_a_i != 0:
                raise ValueError(
                    f"Incompatible: a[{i}] has length {len_a_i} "
                    f"but b[{i}] is empty."
                )
            continue  # skip if both are empty

        # Repeat b[i] cyclically to fill a[i]
        for j in range(len_a_i):
            index_values[pos] = start_b_i + (j % len_b_i)
            pos += 1


def map_a_to_b(a: ak.Array, b: ak.Array) -> ak.Array:
    """
    Map the jagged structure of `a` onto `b`, such that each sublist in `a` gets
    a matching sublist from `b`, repeating elements cyclically if needed.

    Parameters
    ----------
    a : ak.Array
        Jagged array whose structure will be imposed.
    b : ak.Array
        Jagged array to be aligned with `a`.

    Returns
    -------
    ak.Array
        Jagged array with the same outer structure as `a` filled with
        elements from `b`.
    """
    # Ensure inputs are Awkward Arrays
    a = ak.Array(a)
    b = ak.Array(b)

    # Flatten b and get its list offsets
    flat_b = ak.flatten(b)
    b_offsets = b.layout.offsets.data

    # Get offsets and lengths of sublists in `a`
    a_offsets = a.layout.offsets.data
    a_lengths = a_offsets[1:] - a_offsets[:-1]

    # Allocate index array that will map a -> b
    index_values = np.empty(a_offsets[-1], dtype=np.int64)

    # Fill the index array using the numba-accelerated function
    build_index(index_values, a_lengths, b_offsets)

    # Build the Awkward IndexedArray from the flat content of b
    index = ak.index.Index64(index_values)
    indexed_array = ak.contents.IndexedArray(index, flat_b.layout)

    # Wrap into a jagged array with the same offsets as a
    list_array = ak.contents.ListOffsetArray(a.layout.offsets, indexed_array)
    return ak.Array(list_array)


def ttbar_reco(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
) -> tuple[ak.Array, ak.Array]:
    """
    Perform semi-leptonic ttbar reconstruction using fatjets if available,
    falling back to combinatoric jet assignment otherwise.

    Parameters
    ----------
    muons : ak.Array
        Array of muon candidates.
    jets : ak.Array
        Array of jet candidates.
    fatjets : ak.Array
        Array of fatjet candidates.
    met : ak.Array
        Array of MET information.

    Returns
    -------
    tuple[ak.Array, ak.Array]
        Tuple of (chi2, mtt) values for the best ttbar reconstruction per event.
    """
    # Define Gaussian means and widths for chi2 computation
    mean_mlep = 175.0
    sigma_mlep = 19.0
    mean_mhad_fj = 173.0
    sigma_mhad_fj = 15.0
    mean_mhad_nofj = 177.0
    sigma_mhad_nofj = 16.0

    # Define which events have at least one fatjet
    has_fatjet = ak.num(fatjets, axis=1) > 0
    no_fatjet = ~has_fatjet

    # ============================================================
    # Fatjet-based reconstruction
    # ============================================================

    # Mask inputs to only fatjet-containing events
    fatjets_masked = ak.mask(fatjets, has_fatjet)
    jets_masked = ak.mask(jets, has_fatjet)
    muons_masked = ak.mask(muons, has_fatjet)
    met_masked = ak.mask(met, has_fatjet)

    # Use leading fatjet as hadronic top candidate
    leading_fatjet = fatjets_masked[:, 0]
    had_top_fj = ak.zip(
        {
            "pt": leading_fatjet.pt,
            "eta": leading_fatjet.eta,
            "phi": leading_fatjet.phi,
            "mass": leading_fatjet.mass,
        },
        with_name="Momentum4D",
    )
    # Use leading muon
    leading_muon = muons_masked[:, 0]

    # Broadcast other objects to jet axis
    fatjet_broadcast = ak.broadcast_arrays(jets_masked, leading_fatjet)[1]
    muon_broadcast = ak.broadcast_arrays(jets_masked, leading_muon)[1]
    met_broadcast = ak.broadcast_arrays(jets_masked, met_masked)[1]

    # Reject jets too close to fatjet (likely overlap with substructure)
    delta_r = jets_masked.deltaR(fatjet_broadcast)
    jet_mask = delta_r > 1.2

    valid_jets = jets_masked[jet_mask]
    valid_muons = muon_broadcast[jet_mask]
    valid_met = met_broadcast[jet_mask]

    # Construct 4-vectors for leptonic top components
    valid_muons_4vec, valid_jets_4vec = [
        ak.zip(
            {"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "mass": obj.mass},
            with_name="Momentum4D",
        )
        for obj in [valid_muons, valid_jets]
    ]
    valid_met_4vec = ak.zip(
        {
            "pt": valid_met.pt,
            "eta": 0 * valid_met.pt,
            "phi": valid_met.phi,
            "mass": 0,
        },
        with_name="Momentum4D",
    )

    # Leptonic top: muon + MET + jet
    # lep_top_fj = valid_muons_4vec + valid_met_4vec + valid_jets_4vec
    # Solve for neutrino
    pz1_fj, pz2_fj = solve_neutrino_pz(valid_muons_4vec, valid_met_4vec)
    lep_top_fj = build_leptonic_tops(
        valid_muons_4vec, valid_met_4vec, valid_jets_4vec, pz1_fj, pz2_fj
    )
    had_mass_broadcast = ak.broadcast_arrays(lep_top_fj, had_top_fj)[1].mass

    # Chi² for each combination
    chi2_fj = ((lep_top_fj.mass - mean_mlep) / sigma_mlep) ** 2 + (
        (had_mass_broadcast - mean_mhad_fj) / sigma_mhad_fj
    ) ** 2

    # ============================================================
    # Combinatoric AK4 reconstruction (no fatjets)
    # ============================================================

    muons_nofj = ak.mask(muons, no_fatjet)[:, 0]
    met_nofj = ak.mask(met, no_fatjet)

    # All jet pairs -> one jet for leptonic top, two for hadronic top
    # combs = ak.combinations(jets_nofj, 2, axis=1, replacement=False)
    # lepjet, hadjets = combs["0"], combs["1"]
    four_jet_combos = ak.combinations(
        jets, 4, fields=["j1", "j2", "j3", "j4"]
    )  # trijet candidates
    lepjet = four_jet_combos["j4"]
    had_top_nofj = (
        four_jet_combos["j1"] + four_jet_combos["j2"] + four_jet_combos["j3"]
    )

    # Broadcast muon and MET to pair structure
    muon_broadcast = ak.broadcast_arrays(lepjet, muons_nofj)[1]
    met_broadcast = ak.broadcast_arrays(lepjet, met_nofj)[1]

    # Leptonic top 4-vectors
    lepjet_4vec = ak.zip(
        {
            "pt": lepjet.pt,
            "eta": lepjet.eta,
            "phi": lepjet.phi,
            "mass": lepjet.mass,
        },
        with_name="Momentum4D",
    )
    muon_4vec = ak.zip(
        {
            "pt": muon_broadcast.pt,
            "eta": muon_broadcast.eta,
            "phi": muon_broadcast.phi,
            "mass": muon_broadcast.mass,
        },
        with_name="Momentum4D",
    )
    met_4vec = ak.zip(
        {
            "pt": met_broadcast.pt,
            "eta": 0 * met_broadcast.pt,
            "phi": met_broadcast.phi,
            "mass": 0,
        },
        with_name="Momentum4D",
    )

    pz1_nofj, pz2_nofj = solve_neutrino_pz(muon_4vec, met_4vec)
    lep_top_nofj = build_leptonic_tops(
        muon_4vec, met_4vec, lepjet_4vec, pz1_nofj, pz2_nofj
    )

    # Hadronic top = dijet system
    had_top_nofj = ak.zip(
        {
            "pt": had_top_nofj.pt,
            "eta": had_top_nofj.eta,
            "phi": had_top_nofj.phi,
            "mass": had_top_nofj.mass,
        },
        with_name="Momentum4D",
    )
    # Map leptonic top jagged structure onto hadronic tops
    had_top_nofj = map_a_to_b(lep_top_nofj, had_top_nofj)

    # Make sure if we don't have a leptonic top, we don't have a hadronic top
    lep_empty = ak.num(lep_top_nofj, axis=1) == 0
    had_top_nofj = ak.where(
        lep_empty, ak.Array([[]] * len(lep_top_nofj)), had_top_nofj
    )

    # Chi² for each combination
    chi2_nofj = ((lep_top_nofj.mass - mean_mlep) / sigma_mlep) ** 2 + (
        (had_top_nofj.mass - mean_mhad_nofj) / sigma_mhad_nofj
    ) ** 2

    # ============================================================
    # Select best candidate and compute mtt
    # ============================================================
    best_idx_fj = ak.argmin(chi2_fj, axis=1, keepdims=True)
    best_idx_nofj = ak.argmin(chi2_nofj, axis=1, keepdims=True)

    # Use chi2 to select best pairing
    lep_top_fj_best = lep_top_fj[best_idx_fj]
    had_top_fj_best = ak.broadcast_arrays(lep_top_fj, had_top_fj)[1][
        best_idx_fj
    ]

    lep_top_nofj_best = lep_top_nofj[best_idx_nofj]
    had_top_nofj_best = ak.broadcast_arrays(lep_top_nofj, had_top_nofj)[1][
        best_idx_nofj
    ]

    # Choose reconstruction based on presence of fatjet
    lep_top = ak.flatten(
        ak.where(has_fatjet, lep_top_fj_best, lep_top_nofj_best)
    )
    had_top = ak.flatten(
        ak.where(has_fatjet, had_top_fj_best, had_top_nofj_best)
    )
    mtt = ak.fill_none((lep_top + had_top).mass, -1.0)

    # Final chi2 score
    chi2 = ak.where(has_fatjet, chi2_fj, chi2_nofj)
    best_idx = ak.argmin(chi2, axis=1, keepdims=True)
    best_chi2 = ak.fill_none(ak.flatten(chi2[best_idx]), 9999.0)

    return best_chi2, mtt


def mtt_from_ttbar_reco(ttbar_reco: ak.Array) -> ak.Array:
    """
    Extract mtt from the ttbar reconstruction result.

    Parameters
    ----------
    ttbar_reco : ak.Array
        Output of `ttbar_reco`, expected to be awkward array with an 'mtt' field.

    Returns
    -------
    ak.Array
        Array of mtt values.
    """
    return ttbar_reco.mtt


def chi2_from_ttbar_reco(ttbar_reco: ak.Array) -> ak.Array:
    """
    Extract chi2 from the ttbar reconstruction result.

    Parameters
    ----------
    ttbar_reco : ak.Array
        Output of `ttbar_reco`, expected to be awkward array with an 'mtt' field.

    Returns
    -------
    ak.Array
        Array of mtt values.
    """
    return ttbar_reco.chi2


def compute_mva_vars(muons: ak.Array, jets: ak.Array) -> dict[str, np.ndarray]:
    """
    Compute input features for MVA from muon and jet collections.

    Assumes exactly one muon and at least two jets per event.

    Parameters
    ----------
    muons : ak.Array
        Array of muon candidates (exactly one per event).
    jets : ak.Array
        Array of jet candidates (at least two per event).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping feature names to NumPy arrays,
        suitable for use in ML model evaluation or training.
    """
    # Sanity checks
    assert ak.all(
        ak.num(jets, axis=1) >= 2
    ), "Each event must have at least 2 jets"
    assert ak.all(
        ak.num(muons, axis=1) == 1
    ), "Each event must have exactly 1 muon"

    features = {}

    # jet-level features 1
    padded_jet_mass = ak.pad_none(jets.mass, target=2, axis=1, clip=True)
    padded_jet_mass = ak.fill_none(padded_jet_mass, 0.0)  # fill None with 0.0
    features["n_jet"] = ak.to_numpy(ak.num(jets, axis=1))
    features["leading_jet_mass"] = ak.to_numpy(jets.mass[:, 0])
    features["subleading_jet_mass"] = ak.to_numpy(padded_jet_mass[:, 1])

    # Scalar sum of transverse momenta (ST)
    features["st"] = ak.to_numpy(
        ak.sum(jets.pt, axis=1) + ak.sum(muons.pt, axis=1)
    )

    # jet-level features 2
    features["leading_jet_btag_score"] = ak.to_numpy(jets.btagDeepB[:, 0])
    features["subleading_jet_btag_score"] = ak.to_numpy(jets.btagDeepB[:, 1])

    # Longitudinal imbalance S_zz = sum(pz^2) / sum(p^2)
    jet_p2 = jets.px**2 + jets.py**2 + jets.pz**2
    with np.errstate(divide="ignore", invalid="ignore"):
        S_zz = ak.sum(jets.pz**2, axis=1) / ak.sum(jet_p2, axis=1)
    features["S_zz"] = ak.to_numpy(S_zz)

    # Minimum deltaR between the muon and any jet
    muon_in_pair, jet_in_pair = ak.unzip(ak.cartesian([muons, jets]))
    deltaR = muon_in_pair.deltaR(jet_in_pair)
    min_deltaR = ak.min(deltaR, axis=1)
    features["deltaR"] = min_deltaR.to_numpy()

    # pt_rel: transverse momentum of muon relative to closest jet
    closest_jet_idx = ak.argmin(deltaR, axis=1, keepdims=True)
    closest_jet = jet_in_pair[closest_jet_idx]
    delta_angle = muons.deltaangle(closest_jet)
    pt_rel = muons.p * np.sin(delta_angle)
    features["pt_rel"] = ak.to_numpy(ak.flatten(pt_rel, axis=None))

    # deltaR * pT of closest jet
    closest_jet_pt = closest_jet.pt
    deltaR_times_pt = min_deltaR * closest_jet_pt
    features["deltaR_times_pt"] = ak.to_numpy(
        ak.flatten(deltaR_times_pt, axis=None)
    )

    return features


def get_mva_vars(muons: ak.Array, jets: ak.Array) -> ak.Array:
    """
    Extracts MVA input variables from given muon and jet collections.

    Parameters
    ----------
    muons : ak.Array
        Array of selected muons per event.
    jets : ak.Array
        Array of selected jets per event.

    Returns
    -------
    ak.Array
        Tuple of MVA input features, as arrays aligned with the event structure.
    """
    d = compute_mva_vars(muons, jets)
    return tuple(d.values())


def get_n_jet(mva):
    return mva.n_jet


def get_leading_jet_mass(mva):
    return mva.leading_jet_mass


def get_subleading_jet_mass(mva):
    return mva.subleading_jet_mass


def get_st(mva):
    return mva.st


def get_leading_jet_btag_score(mva):
    return mva.leading_jet_btag_score


def get_subleading_jet_btag_score(mva):
    return mva.subleading_jet_btag_score


def get_S_zz(mva):
    return mva.S_zz


def get_deltaR(mva):
    return mva.deltaR


def get_pt_rel(mva):
    return mva.pt_rel


def get_deltaR_times_pt(mva):
    return mva.deltaR_times_pt


def compute_mva_scores(
    muons: ak.Array,
    jets: ak.Array,
    model_path: str = "output/models/model_new.keras",
) -> np.ndarray:
    """
    Compute MVA scores using a pre-trained Keras model.

    Parameters
    ----------
    muons : ak.Array
        Array of muon candidates.
    jets : ak.Array
        Array of jet candidates.
    model_path : str, optional
        Path to the Keras model file, by default "output/model.keras".

    Returns
    -------
    np.ndarray
        Flat array of MVA scores per event.
    """
    from tensorflow.keras.models import load_model  # type: ignore


    model = load_model(model_path, compile=False)
    logger.info(model.summary())
    scores = np.full(len(jets), -1.0, dtype=np.float32)

    idx = (
        (ak.num(jets, axis=1) >= 2) & (ak.num(muons, axis=1) == 1)
    ).to_numpy()
    mva_vars = compute_mva_vars(muons[idx], jets[idx])
    X = np.column_stack(list(mva_vars.values())).astype(float)
    scores[idx] = model.predict(X, batch_size=1024).flatten()
    # need to map scores to -1 -> 1 instead of 0 -> 1
    scores = scores * 2 - 1
    return scores


def get_mva_scores(mva):
    return mva.nn_score
