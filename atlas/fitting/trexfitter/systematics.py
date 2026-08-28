"""Systematic blocks, discovered directly from the TRExFitter input file.

Every sample directory in `wsi/trexfitter.root` holds a `nominal` histogram plus one
histogram per variation, named `<syst>_Up` / `<syst>_Down`. The systematics are read back
from there rather than hard-coded, so the config always matches the input.

A systematic is written with `Symmetrisation: ONESIDED` (up variation only) when it has no
down variation at all, or when its up and down histograms are bin-by-bin identical -- which
is how `hist_to_trexfitter.py` ends up storing genuinely one-sided variations. Everything
else is written as a regular two-sided up/down systematic.
"""

import os

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


def _category(name):
    for prefix, category in CATEGORIES:
        if name.startswith(prefix):
            return category
    return "Other"


def _read_directory(path, directory):
    """return {histogram name: tuple of bin contents} for one sample directory"""
    try:
        import uproot

        with uproot.open(path) as f:
            contents = f[directory]
            return {key: tuple(contents[key].values()) for key in contents.keys(cycle=False)}
    except ImportError:
        pass

    import ROOT  # provided by `asetup StatAnalysis`, see setup.sh

    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise RuntimeError(f"could not open {path}")
    contents = f.Get(directory)
    if not contents:
        raise RuntimeError(f"no directory {directory} in {path}")
    histograms = {}
    for key in contents.GetListOfKeys():
        h = key.ReadObj()
        histograms[key.GetName()] = tuple(h.GetBinContent(i) for i in range(1, h.GetNbinsX() + 1))
    f.Close()
    return histograms


def discover_systematics(args, samples):
    """collect the systematics available in the input file

    `samples` is the list of (config name, HistoNameSuff) pairs the systematics can apply to.
    Returns a list of (name, samples with that systematic, onesided) tuples, sorted by name.
    """
    path = os.path.join(args.histo_path, f"{args.region_histo_file}.root")

    found = {}  # name -> {sample: (up contents, down contents or None)}
    for name, directory in samples:
        histograms = _read_directory(path, f"{args.region_histo_name}{directory}".rstrip("/"))
        for key, contents in histograms.items():
            if key == NOMINAL or not key.endswith(UP_SUFFIX):
                continue
            syst = key[: -len(UP_SUFFIX)]
            found.setdefault(syst, {})[name] = (contents, histograms.get(f"{syst}{DOWN_SUFFIX}"))

    order = {name: i for i, (name, _) in enumerate(samples)}
    systematics = []
    for syst in sorted(found):
        variations = found[syst]
        onesided = all(down is None or up == down for up, down in variations.values())
        systematics.append((syst, sorted(variations, key=order.get), onesided))
    return systematics


def _write_histo_syst(f, name, samples, onesided, region):
    f.write(f'Systematic: "{name}"\n')
    f.write(f'  Title: "{name.replace("_", " ")}"\n')
    f.write(f"  Type: HISTO\n")
    f.write(f"  Samples: {','.join(samples)}\n")
    f.write(f"  Regions: {region}\n")
    f.write(f"  HistoNameSufUp: {name}{UP_SUFFIX}\n")
    if onesided:
        f.write(f"  Symmetrisation: ONESIDED\n")
    else:
        f.write(f"  HistoNameSufDown: {name}{DOWN_SUFFIX}\n")
    f.write(f'  Category: "{_category(name)}"\n')
    f.write(f"  NuisanceParameter: {name}\n")
    f.write("\n")


def write_systematic_blocks(f, args, samples):
    systematics = discover_systematics(args, samples)

    f.write("% --------------------- %\n")
    f.write("% ---  Systematics  --- %\n")
    f.write("% --------------------- %\n\n")

    category = None
    for name, syst_samples, onesided in systematics:
        if _category(name) != category:
            category = _category(name)
            f.write(f"% --- {category} --- %\n\n")
        _write_histo_syst(f, name, syst_samples, onesided, args.region_name)

    n_onesided = sum(onesided for _, _, onesided in systematics)
    print(f"  {len(systematics)} systematics ({n_onesided} one-sided, "
          f"{len(systematics) - n_onesided} two-sided)")
