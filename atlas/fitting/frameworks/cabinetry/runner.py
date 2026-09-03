"""
Write and run a cabinetry fit. `run_fit` is steered by the same kind of option string as
`trex-fitter`, so both frameworks can be driven through the same steps:

    h   build the template histograms (`templates.collect` + `templates.postprocess`)
    w   build the pyhf workspace
    d   pre-fit data/MC plot and yield table
    f   fit, pull plot, correlation matrix and likelihood scan of the POI
    p   post-fit data/MC plot and yield table
    r   nuisance parameter ranking

The steps always run in that order, no matter how the option string is spelled, and later
steps re-use the results of the earlier ones.
"""

import logging
import os

import yaml

from ..samples import MC_SAMPLES
from . import config_blocks
from .systematics import systematic_blocks

NAME = "cabinetry"

# Collection of fit tuning parameters that loosen conversion requirements for cabinetry
# in an attempt to match trexfitter and converge the fit
MIN_EXPECTED_YIELD = 1e-6
MAX_CALLS = 1_000_000
FIT_TOLERANCE = 5.0


def config_path(args):
    return os.path.join(args.config_outdir, f"{args.job_name}.yml")


def build_config(args):
    """assemble the cabinetry configuration for `args`"""
    config = {
        "General": config_blocks.general_block(args),
        "Regions": config_blocks.region_blocks(args),
        "Samples": config_blocks.sample_blocks(args),
        "NormFactors": config_blocks.normfactor_blocks(args),
    }
    if not args.stats_only:
        config["Systematics"] = systematic_blocks(args, MC_SAMPLES)
    return config


def write_config(args):
    """write the cabinetry config for `args` and return its path"""
    from cabinetry import configuration

    os.makedirs(args.config_outdir, exist_ok=True)
    outpath = config_path(args)

    config = build_config(args)
    configuration.validate(config)

    with open(outpath, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    return outpath


def model_and_data(workspace):
    """build the pyhf model and its data, with the expected yields floored"""
    
    import pyhf

    workspace = pyhf.Workspace(workspace)
    model = workspace.model(
        # cabinetry's interpolation codes, HistFactory InterpCode=4
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
        clip_bin_data=MIN_EXPECTED_YIELD,
    )
    return model, workspace.data(model)


def run_fit(args, outpath):
    """run the requested `--cabinetry-opts` steps on a generated config"""
    import cabinetry

    # cabinetry reports fit results, yields and pruning decisions through logging, which
    # is silent by default -- `cabinetry.set_logging()` would be the DEBUG equivalent
    logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s")
    logging.getLogger("cabinetry").setLevel(logging.INFO)

    steps = set(args.cabinetry_opts)
    poi = args.parameter_of_interest

    outdir = config_blocks.output_dir(args)
    figure_folder = os.path.join(outdir, "figures")
    table_folder = os.path.join(outdir, "tables")
    workspace_path = os.path.join(outdir, "workspace.json")
    colors = config_blocks.sample_colors()

    print(f"Running cabinetry steps '{args.cabinetry_opts}' on {outpath}")
    config = cabinetry.configuration.load(outpath)

    # cabinetry creates the histogram folder itself, but not its parent
    os.makedirs(outdir, exist_ok=True)

    if "h" in steps:
        cabinetry.templates.collect(config, method="uproot")
        cabinetry.templates.postprocess(config)

    if "w" in steps:
        cabinetry.workspace.save(cabinetry.workspace.build(config), workspace_path)

    if not steps & set("dfpr"):
        return

    model, data = model_and_data(cabinetry.workspace.load(workspace_path))
    if not args.unblind:
        # the counterpart of TRExFitter's `FitBlind: TRUE`: fit an Asimov dataset built
        # at the nominal parameter values with the POI set to `--mu-asimov`
        data = cabinetry.model_utils.asimov_data(
            model, poi_name=poi, poi_value=args.mu_asimov
        )

    if "d" in steps:
        prefit = cabinetry.model_utils.prediction(model)
        cabinetry.visualize.data_mc(
            prefit, data, config=config, figure_folder=figure_folder, colors=colors
        )
        cabinetry.tabulate.yields(prefit, data, table_folder=table_folder, per_channel=True)

    if not steps & set("fpr"):
        return

    fit_results = cabinetry.fit.fit(
        model,
        data,
        minos=poi,
        goodness_of_fit=True,
        maxiter=MAX_CALLS,
        tolerance=FIT_TOLERANCE,
    )
    cabinetry.fit.print_results(fit_results)

    if "f" in steps:
        cabinetry.visualize.pulls(
            fit_results, figure_folder=figure_folder, exclude=poi
        )
        cabinetry.visualize.correlation_matrix(
            fit_results,
            figure_folder=figure_folder,
            pruning_threshold=args.correlation_threshold,
        )
        if args.lh_scan_steps > 0:
            scan_results = cabinetry.fit.scan(
                model,
                data,
                poi,
                par_range=(args.lh_scan_min, args.lh_scan_max),
                n_steps=args.lh_scan_steps,
                maxiter=MAX_CALLS,
                tolerance=FIT_TOLERANCE,
            )
            cabinetry.visualize.scan(scan_results, figure_folder=figure_folder)

    if "p" in steps:
        postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)
        cabinetry.visualize.data_mc(
            postfit, data, config=config, figure_folder=figure_folder, colors=colors
        )
        cabinetry.tabulate.yields(postfit, data, table_folder=table_folder, per_channel=True)

    if "r" in steps:
        # a stat-only fit leaves nothing but the POI in the model, and cabinetry's
        # ranking cannot handle an empty set of nuisance parameters
        if all(par == poi for par in model.config.par_names):
            print("No nuisance parameters to rank, skipping the ranking step")
            return

        ranking_results = cabinetry.fit.ranking(
            model,
            data,
            fit_results=fit_results,
            maxiter=MAX_CALLS,
            tolerance=FIT_TOLERANCE,
        )
        cabinetry.visualize.ranking(
            ranking_results, figure_folder=figure_folder, max_pars=args.ranking_max_np
        )
