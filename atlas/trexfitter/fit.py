import argparse
import copy
import os

from fitting.config_blocks import (MC_SAMPLES,
                                   write_job_block,
                                   write_fit_block,
                                   write_normfactor_blocks,
                                   write_region_block,
                                   write_sample_blocks)
from fitting.systematics import write_systematic_blocks

STAT_ONLY_SUFFIX = "_stat_only"


def write_config(args):
    """write one TRExFitter config and return its path

    Stat-only jobs get a `_stat_only` suffix so they do not overwrite the full-systematics
    config and TRExFitter output directory.
    """
    if args.stats_only and not args.job_name.endswith(STAT_ONLY_SUFFIX):
        args = copy.copy(args)
        args.job_name += STAT_ONLY_SUFFIX

    os.makedirs(args.config_outdir, exist_ok=True)
    outpath = os.path.join(args.config_outdir, f"{args.job_name}.config")

    with open(outpath, "w") as f:
        write_job_block(f, args)
        write_fit_block(f, args)
        write_normfactor_blocks(f, args)
        write_region_block(f, args)
        write_sample_blocks(f, args)
        if not args.stats_only:
            write_systematic_blocks(f, args, MC_SAMPLES)

    print(f"Config written to {outpath}")
    return outpath


def main(args):
    outpath = write_config(args)

    if args.run_fit:
        print("Running fit...")
        os.system(f"trex-fitter {args.trex_opts} {outpath}")

    if args.compare_stat_only:
        stat_args = copy.copy(args)
        stat_args.stats_only = True
        stat_args.plot_systematics = False
        stat_outpath = write_config(stat_args)
        if args.run_fit:
            print("Running stat-only fit...")
            os.system(f"trex-fitter {args.trex_opts} {stat_outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config generator for the TRExFitter H+ -> cb m_jj fit")

    job = parser.add_argument_group("Job")
    job.add_argument("-out", "--config-outdir", type=str, default="configs",
                     help="Output directory for generated config files")
    job.add_argument("-job", "--job-name", type=str, default="Hplus_cb_mjj",
                     help="Job name written to config (also used as filename)")
    job.add_argument("-stat", "--stats-only", action="store_true",
                     help="Stats-only fit (sets StatOnly: TRUE)")
    job.add_argument("-com", "--center-of-mass-energy", type=float, default=13.6,
                     help="Centre-of-mass energy in TeV")
    job.add_argument("-lumi", "--luminosity", type=float, default=26.3,
                     help="Integrated luminosity in fb^-1")
    job.add_argument("-poi", "--parameter-of-interest", type=str, default="mu",
                     help="Parameter of interest name")
    job.add_argument("-ps", "--plot-systematics", action="store_true",
                     help="Enable SystControlPlots")
    job.add_argument("--histo-path", type=str, default="wsi/",
                     help="HistoPath (directory containing histogram files)")

    fit = parser.add_argument_group("Fit config")
    fit.add_argument("--mu-asimov", type=float, default=1.0,
                     help="Injected Asimov signal strength (POIAsimov)")
    fit.add_argument("--mu-min", type=float, default=0.0,
                     help="Lower bound for the mu NormFactor and LH scan")
    fit.add_argument("--mu-max", type=float, default=5.0,
                     help="Upper bound for the mu NormFactor and LH scan")
    fit.add_argument("--lh-scan-steps", type=int, default=120,
                     help="Number of LH scan steps")
    fit.add_argument("--unblind", action="store_true",
                     help="Fit the observed data instead of an Asimov dataset")
    fit.add_argument("--num-cpu", type=int, default=8,
                     help="Number of CPUs for fitting")

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

    plotting = parser.add_argument_group("Plotting")
    plotting.add_argument("-cso", "--compare-stat-only", action="store_true",
                          help="Also generate (and run) a stat-only config")

    args = parser.parse_args()
    main(args)
