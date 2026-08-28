"""Systematics discovered directly from the fit input file, shared by both frameworks"""

import os

import uproot

NOMINAL = "nominal"
UP_SUFFIX = "_Up"
DOWN_SUFFIX = "_Down"

# name prefix -> category, first match wins
CATEGORIES = [
    ("FT_EFF_Eigen_B",     "b-tagging (b)"),
    ("FT_EFF_Eigen_C",     "b-tagging (c)"),
    ("FT_EFF_Eigen_Light", "b-tagging (light)"),
    ("FT_",                "b-tagging"),
    ("JET_JER",            "Jet resolution"),
    ("JET_",               "Jet energy scale"),
    ("MET_",               "Missing transverse momentum"),
    ("MUON_EFF_",          "Muon efficiency"),
    ("MUON_",              "Muon momentum"),
    ("EL_EFF_",            "Electron efficiency"),
    ("EG_",                "Electron energy"),
]


def category(name):
    for prefix, category_name in CATEGORIES:
        if name.startswith(prefix):
            return category_name
    return "Other"


def input_file(args):
    """path of the ROOT file holding the fit inputs"""
    return os.path.join(args.histo_path, f"{args.region_histo_file}.root")


def _read_directory(path, directory):
    """return {histogram name: tuple of bin contents} for one sample directory"""
    with uproot.open(path) as f:
        contents = f[directory]
        return {key: tuple(contents[key].values()) for key in contents.keys(cycle=False)}


def _relative_effects(nominal, variation):
    """(normalisation, shape) effect of `variation`, both relative to `nominal`"""
    nominal_total = sum(nominal)
    variation_total = sum(variation)
    if nominal_total == 0 or variation_total == 0:
        return 0.0, 0.0

    norm = abs(variation_total / nominal_total - 1)
    # the shape effect is whatever is left once the normalisation difference is scaled
    # away, the convention TRExFitter uses to separate its norm and shape pruning
    scale = nominal_total / variation_total
    shape = max(
        (abs(v * scale / n - 1) for v, n in zip(variation, nominal) if n != 0),
        default=0.0,
    )
    return norm, shape


def _is_negligible(nominal, up, down, threshold):
    """whether neither side of a variation moves `nominal` by at least `threshold`"""
    for variation in (up, down):
        if variation is None:
            continue
        norm, shape = _relative_effects(nominal, variation)
        if norm >= threshold or shape >= threshold:
            return False
    return True


def discover_systematics(args, samples):
    """collect the systematics available in the input file

    `samples` is the list of `Sample`s the systematics can apply to. A variation whose
    normalisation and shape effects are both below `--syst-pruning-threshold` is dropped
    for that sample, the equivalent of TRExFitter's `SystPruningNorm`/`SystPruningShape`
    but applied to both frameworks, so they fit the same set of nuisance parameters; a
    systematic left without any sample disappears entirely.

    Returns a list of (name, names of the samples with that systematic, onesided) tuples
    sorted by name, and the number of sample variations that were pruned away.
    """
    path = input_file(args)
    threshold = args.syst_pruning_threshold

    found = {}  # name -> {sample: (up contents, down contents or None)}
    pruned = 0
    for sample in samples:
        histograms = _read_directory(
            path, f"{args.region_histo_name}{sample.directory}".rstrip("/")
        )
        nominal = histograms[NOMINAL]
        for key, contents in histograms.items():
            if key == NOMINAL or not key.endswith(UP_SUFFIX):
                continue
            syst = key[: -len(UP_SUFFIX)]
            down = histograms.get(f"{syst}{DOWN_SUFFIX}")
            if _is_negligible(nominal, contents, down, threshold):
                pruned += 1
                continue
            found.setdefault(syst, {})[sample.name] = (contents, down)

    order = {sample.name: i for i, sample in enumerate(samples)}
    systematics = []
    for syst in sorted(found):
        variations = found[syst]
        onesided = all(down is None or up == down for up, down in variations.values())
        systematics.append((syst, sorted(variations, key=order.get), onesided))
    return systematics, pruned


def print_summary(systematics, pruned, threshold):
    n_onesided = sum(onesided for _, _, onesided in systematics)
    print(f"  {len(systematics)} systematics ({n_onesided} one-sided, "
          f"{len(systematics) - n_onesided} two-sided)")
    print(f"  {pruned} sample variations pruned below {threshold:g}")
