"""TRExFitter config blocks for the H+ -> cb dijet-mass fit.

The input (`wsi/trexfitter.root`, written by `hist_to_trexfitter.py`) holds one
TDirectory per sample, named `<region prefix><sample>` (e.g. `SR_ttbar_nom`), each
containing the `nominal` histogram plus one histogram per systematic variation
(`<syst>_Up` / `<syst>_Down`).

TRExFitter builds the histogram name as

    Region.HistoName + Sample.HistoNameSuff + Systematic.HistoNameSufUp

with the nominal name being Region.HistoName + Sample.HistoNameSuff +
Job.HistoNameNominal, so the region contributes `SR_`, the sample contributes
`ttbar_nom/` and the systematic contributes `JET_BJES_Response_Up`.
"""

# Samples: (config name, HistoNameSuff = TDirectory in the input file, title, fill colour).
#
# The input file also contains the alternative-model samples `ttbar_H7`, `ttbar_hdamp`,
# `Wt_DS` and `Wt_H7`. They describe the same processes as `ttbar_nom` / `Wt` and are only
# useful as two-point modelling systematics, so they are left out of this fit.
DATA_SAMPLE = ("data", "data/")

SIGNAL_SAMPLE = ("Hplus_cb", "Hplus_cb/", "H^{+} #rightarrow cb", 632)

BACKGROUND_SAMPLES = [
    ("ttbar",    "ttbar_nom/", "t#bar{t}",          800),
    ("Wt",       "Wt/",        "Wt",                807),
    ("st_tchan", "st_tchan/",  "Single top t-chan", 831),
    ("st_schan", "st_schan/",  "Single top s-chan", 833),
    ("wjets",    "wjets/",     "W+jets",            417),
    ("zjets",    "zjets/",     "Z+jets",            861),
    ("diboson",  "diboson/",   "Diboson",           616),
    ("ttV",      "ttV/",       "t#bar{t}V",         880),
    ("rare_top", "rare_top/",  "Rare top",          920),
]

# all samples a systematic can be applied to, as (name, directory) pairs
MC_SAMPLES = [(SIGNAL_SAMPLE[0], SIGNAL_SAMPLE[1])] + [(n, d) for n, d, _, _ in BACKGROUND_SAMPLES]


def write_job_block(f, args):
    poi = args.parameter_of_interest
    f.write(f"Job: {args.job_name}\n")
    f.write(f'  Label: "H^{{+}} #rightarrow cb"\n')
    f.write(f"  CmeLabel: {args.center_of_mass_energy:g} TeV\n")
    f.write(f"  LumiLabel: {args.luminosity:g} fb^{{-1}}\n")
    f.write(f"  POI: {poi}\n")
    f.write(f"  ReadFrom: HIST\n")
    f.write(f"  HistoPath: {args.histo_path}\n")
    f.write(f"  HistoNameNominal: nominal\n")
    f.write(f"  DebugLevel: 1\n")
    f.write(f"  SystControlPlots: {'TRUE' if args.plot_systematics else 'FALSE'}\n")
    f.write(f"  UseGammaPulls: TRUE\n")
    f.write(f"  BlindingThreshold: 0.1\n")
    f.write(f"  RankingMaxNP: 20\n")
    f.write(f'  RankingPOIName: "#mu"\n')
    f.write(f"  HistoChecks: NOCRASH\n")
    f.write(f"  SystPruningNorm: 0.01\n")
    f.write(f"  SystPruningShape: 0.01\n")
    f.write(f"  CorrelationThreshold: 0.2\n")
    f.write(f"  StatOnly: {'TRUE' if args.stats_only else 'FALSE'}\n")
    f.write("\n")


def write_fit_block(f, args):
    poi = args.parameter_of_interest
    f.write("% --------------- %\n")
    f.write("% ---  Fit    --- %\n")
    f.write("% --------------- %\n\n")
    f.write(f"Fit: {poi}\n")
    f.write(f"  FitType: SPLUSB\n")
    f.write(f"  FitRegion: CRSR\n")
    f.write(f"  FitBlind: {'FALSE' if args.unblind else 'TRUE'}\n")
    f.write(f"  POIAsimov: {args.mu_asimov:g}\n")
    f.write(f"  UseMinos: {poi}\n")
    f.write(f"  doLHscan: {poi}\n")
    f.write(f"  LHscanMin: {args.mu_min:g}\n")
    f.write(f"  LHscanMax: {args.mu_max:g}\n")
    f.write(f"  LHscanSteps: {args.lh_scan_steps}\n")
    f.write(f"  NumCPU: {args.num_cpu}\n")
    f.write("\n\n")


def write_normfactor_blocks(f, args):
    poi = args.parameter_of_interest
    f.write("% ----------------------------- %\n")
    f.write("% --- Normalisation Factors --- %\n")
    f.write("% ----------------------------- %\n\n")
    f.write(f"NormFactor: {poi}\n")
    f.write(f'  Title: "#it{{#{poi}}}"\n')
    f.write(f"  Nominal: 1\n")
    f.write(f"  Min: {args.mu_min:g}\n")
    f.write(f"  Max: {args.mu_max:g}\n")
    f.write(f"  Samples: {SIGNAL_SAMPLE[0]}\n")
    f.write("\n\n")


def write_region_block(f, args):
    f.write("% ----------------- %\n")
    f.write("% ---  Regions  --- %\n")
    f.write("% ----------------- %\n\n")
    f.write(f'Region: "{args.region_name}"\n')
    f.write(f"  Type: SIGNAL\n")
    f.write(f'  VariableTitle: "m_{{jj}} [GeV]"\n')
    f.write(f"  HistoName: {args.region_histo_name}\n")
    f.write(f"  HistoFile: {args.region_histo_file}\n")
    f.write(f'  Label: "Signal Region"\n')
    f.write(f'  ShortLabel: "SR"\n')
    f.write("\n\n")


def write_sample_blocks(f, args):
    name, directory = DATA_SAMPLE

    f.write("% ----------------- %\n")
    f.write("% ---  Samples  --- %\n")
    f.write("% ----------------- %\n\n")

    f.write(f'Sample: "{name}"\n')
    f.write(f"  Type: DATA\n")
    f.write(f'  Title: "Data"\n')
    f.write(f'  HistoNameSuff: "{directory}"\n')
    f.write("\n")

    name, directory, title, color = SIGNAL_SAMPLE
    f.write(f'Sample: "{name}"\n')
    f.write(f"  Type: SIGNAL\n")
    f.write(f'  Title: "{title}"\n')
    f.write(f"  FillColor: {color}\n")
    f.write(f"  LineColor: 1\n")
    f.write(f'  HistoNameSuff: "{directory}"\n')
    f.write(f"  UseMCstat: TRUE\n")
    f.write("\n")

    for name, directory, title, color in BACKGROUND_SAMPLES:
        f.write(f'Sample: "{name}"\n')
        f.write(f"  Type: BACKGROUND\n")
        f.write(f'  Title: "{title}"\n')
        f.write(f"  FillColor: {color}\n")
        f.write(f"  LineColor: 1\n")
        f.write(f'  HistoNameSuff: "{directory}"\n')
        f.write(f"  UseMCstat: TRUE\n")
        f.write("\n")

    f.write("\n")
