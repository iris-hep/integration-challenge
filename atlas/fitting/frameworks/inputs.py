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


def discover_systematics(args, samples):
    """collect the systematics available in the input file

    `samples` is the list of `Sample`s the systematics can apply to. Returns a list of
    (name, names of the samples with that systematic, onesided) tuples, sorted by name.
    """
    path = input_file(args)

    found = {}  # name -> {sample: (up contents, down contents or None)}
    for sample in samples:
        histograms = _read_directory(
            path, f"{args.region_histo_name}{sample.directory}".rstrip("/")
        )
        for key, contents in histograms.items():
            if key == NOMINAL or not key.endswith(UP_SUFFIX):
                continue
            syst = key[: -len(UP_SUFFIX)]
            found.setdefault(syst, {})[sample.name] = (
                contents,
                histograms.get(f"{syst}{DOWN_SUFFIX}"),
            )

    order = {sample.name: i for i, sample in enumerate(samples)}
    systematics = []
    for syst in sorted(found):
        variations = found[syst]
        onesided = all(down is None or up == down for up, down in variations.values())
        systematics.append((syst, sorted(variations, key=order.get), onesided))
    return systematics


def print_summary(systematics):
    n_onesided = sum(onesided for _, _, onesided in systematics)
    print(f"  {len(systematics)} systematics ({n_onesided} one-sided, "
          f"{len(systematics) - n_onesided} two-sided)")
