import argparse
import copy

from frameworks import cabinetry, trexfitter

FRAMEWORKS = {framework.NAME: framework for framework in (trexfitter, cabinetry)}

STAT_ONLY_SUFFIX = "_stat_only"

def resolve(args):
    """return the arguments the config should actually be written with"""
    if args.stats_only and not args.job_name.endswith(STAT_ONLY_SUFFIX):
        args = copy.copy(args)
        args.job_name += STAT_ONLY_SUFFIX
    return args


def write_config(args):
    """write one config with the selected framework and return its path"""
    outpath = FRAMEWORKS[args.framework].write_config(args)
    print(f"Config written to {outpath}")
    return outpath


def main(args):
    job = resolve(args)
    outpath = write_config(job)

    if args.run_fit:
        print("Running fit...")
        FRAMEWORKS[args.framework].run_fit(job, outpath)

    if args.compare_stat_only:
        stat_args = copy.copy(args)
        stat_args.stats_only = True
        stat_args.plot_systematics = False
        stat_job = resolve(stat_args)
        stat_outpath = write_config(stat_job)
        if args.run_fit:
            print("Running stat-only fit...")
            FRAMEWORKS[args.framework].run_fit(stat_job, stat_outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Config generator for the H+ -> cb m_jj fit, for TRExFitter or cabinetry")

    job = parser.add_argument_group("Job")
    job.add_argument("-f", "--framework", choices=sorted(FRAMEWORKS), default="trexfitter",
                     help="Fitting framework to generate the config for and run")
    job.add_argument("-out", "--config-outdir", type=str, default="configs",
                     help="Output directory for generated config files")
    job.add_argument("-job", "--job-name", type=str, default="Hplus_cb_mjj",
                     help="Job name written to config (also used as filename)")
    job.add_argument("-stat", "--stats-only", action="store_true",
                     help="Stats-only fit (drops all systematics and MC stat uncertainties)")
    job.add_argument("--syst-pruning-threshold", type=float, default=0.01,
                     help="Drop a systematic from a sample when both its normalisation "
                          "and shape effects stay below this fraction of the nominal "
                          "yield; also written as SystPruningNorm/SystPruningShape")
    job.add_argument("-com", "--center-of-mass-energy", type=float, default=13.6,
                     help="Centre-of-mass energy in TeV (TRExFitter only)")
    job.add_argument("-lumi", "--luminosity", type=float, default=26.3,
                     help="Integrated luminosity in fb^-1 (TRExFitter only)")
    job.add_argument("-poi", "--parameter-of-interest", type=str, default="mu",
                     help="Parameter of interest name")
    job.add_argument("-ps", "--plot-systematics", action="store_true",
                     help="Enable SystControlPlots (TRExFitter only)")
    job.add_argument("--histo-path", type=str, default="wsi/",
                     help="HistoPath (directory containing histogram files)")

    fit = parser.add_argument_group("Fit config")
    fit.add_argument("--mu-asimov", type=float, default=1.0,
                     help="Injected Asimov signal strength (POIAsimov)")
    fit.add_argument("--mu-min", type=float, default=0.0,
                     help="Lower bound for the mu NormFactor")
    fit.add_argument("--mu-max", type=float, default=5.0,
                     help="Upper bound for the mu NormFactor")
    fit.add_argument("--lh-scan-min", type=float, default=0.3,
                     help="Lower end of the LH scan, roughly 4 sigma below the best fit")
    fit.add_argument("--lh-scan-max", type=float, default=1.7,
                     help="Upper end of the LH scan, roughly 4 sigma above the best fit")
    fit.add_argument("--lh-scan-steps", type=int, default=20,
                     help="Number of LH scan steps")
    fit.add_argument("--unblind", action="store_true",
                     help="Fit the observed data instead of an Asimov dataset")
    fit.add_argument("--num-cpu", type=int, default=8,
                     help="Number of CPUs for fitting (TRExFitter only)")
    fit.add_argument("--correlation-threshold", type=float, default=0.2,
                     help="Only show parameters above this correlation in the correlation matrix")
    fit.add_argument("--ranking-max-np", type=int, default=20,
                     help="Number of nuisance parameters to show in the ranking plot")

    region = parser.add_argument_group("Region")
    region.add_argument("--region-name", type=str, default="SR",
                        help="Region name used in config and systematics")
    region.add_argument("--region-histo-name", type=str, default="SR_",
                        help="HistoName: prefix of the per-sample TDirectories in the input file")
    region.add_argument("--region-histo-file", type=str, default="trexfitter",
                        help="HistoFile for the signal region (without the .root extension)")

    fit_run = parser.add_argument_group("Fit run options")
    fit_run.add_argument("-r", "--run-fit", action="store_true",
                         help="Whether to run the fit after generating the config")
    fit_run.add_argument("--trex-opts", type=str, default="hwdfpr",
                         help="TRExFitter command-line options for running the fit (e.g. 'hwdfpr')")
    fit_run.add_argument("--cabinetry-opts", type=str, default="hwdfpr",
                         help="cabinetry steps to run, using the same letters as --trex-opts: "
                              "h(istograms), w(orkspace), d(ata/MC), f(it), p(ost-fit), r(anking)")
    fit_run.add_argument("--cabinetry-outdir", type=str, default="",
                         help="Output directory for the cabinetry fit "
                              "(default: '<job name>_cabinetry')")

    plotting = parser.add_argument_group("Plotting")
    plotting.add_argument("-cso", "--compare-stat-only", action="store_true",
                          help="Also generate (and run) a stat-only config")

    args = parser.parse_args()
    main(args)
