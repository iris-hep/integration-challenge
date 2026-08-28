"""TRExFitter config blocks for the H+ -> cb dijet-mass fit."""

from ..samples import BACKGROUND_SAMPLES, DATA_SAMPLE, SIGNAL_SAMPLE


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
    f.write(f"  RankingMaxNP: {args.ranking_max_np}\n")
    f.write(f'  RankingPOIName: "#mu"\n')
    f.write(f"  HistoChecks: NOCRASH\n")
    f.write(f"  SystPruningNorm: 0.01\n")
    f.write(f"  SystPruningShape: 0.01\n")
    f.write(f"  CorrelationThreshold: {args.correlation_threshold:g}\n")
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
    f.write(f"  Samples: {SIGNAL_SAMPLE.name}\n")
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
    f.write("% ----------------- %\n")
    f.write("% ---  Samples  --- %\n")
    f.write("% ----------------- %\n\n")

    f.write(f'Sample: "{DATA_SAMPLE.name}"\n')
    f.write(f"  Type: DATA\n")
    f.write(f'  Title: "{DATA_SAMPLE.title}"\n')
    f.write(f'  HistoNameSuff: "{DATA_SAMPLE.directory}"\n')
    f.write("\n")

    for sample in [SIGNAL_SAMPLE] + BACKGROUND_SAMPLES:
        f.write(f'Sample: "{sample.name}"\n')
        f.write(f"  Type: {'SIGNAL' if sample is SIGNAL_SAMPLE else 'BACKGROUND'}\n")
        f.write(f'  Title: "{sample.title}"\n')
        f.write(f"  FillColor: {sample.root_color}\n")
        f.write(f"  LineColor: 1\n")
        f.write(f'  HistoNameSuff: "{sample.directory}"\n')
        f.write(f"  UseMCstat: TRUE\n")
        f.write("\n")

    f.write("\n")
