"""
cabinetry config blocks for the H+ -> cb dijet-mass fit, the counterpart of
`frameworks/trexfitter/config_blocks.py`. Each function returns the object that ends up
under the corresponding top-level key of the YAML config.
"""

import os

from ..inputs import NOMINAL
from ..samples import BACKGROUND_SAMPLES, DATA_SAMPLE, MC_SAMPLES, SIGNAL_SAMPLE


def output_dir(args):
    """directory cabinetry writes histograms, workspace, figures and tables into"""
    
    return args.cabinetry_outdir or f"{args.job_name}_cabinetry"


def general_block(args):
    return {
        "Measurement": args.job_name,
        "POI": args.parameter_of_interest,
        "InputPath": os.path.join(args.histo_path, f"{args.region_histo_file}.root")
        + ":{RegionPath}{SamplePath}/{VariationPath}",
        "VariationPath": NOMINAL,
        "HistogramFolder": os.path.join(output_dir(args), "histograms"),
    }


def region_blocks(args):
    # `Variable` is only used as the x-axis label for histogram inputs, so it carries the
    # equivalent of the TRExFitter `VariableTitle`
    return [
        {
            "Name": args.region_name,
            "Variable": "$m_{jj}$ [GeV]",
            "RegionPath": args.region_histo_name,
        }
    ]


def sample_blocks(args):
    """
    sample blocks for data, signal and backgrounds
    """
    data = {"Name": DATA_SAMPLE.name, "SamplePath": DATA_SAMPLE.path, "Data": True}

    blocks = [data]
    for sample in [SIGNAL_SAMPLE] + BACKGROUND_SAMPLES:
        block = {"Name": sample.name, "SamplePath": sample.path}
        if args.stats_only:
            block["DisableStaterror"] = True
        blocks.append(block)
    return blocks


def normfactor_blocks(args):
    return [
        {
            "Name": args.parameter_of_interest,
            "Samples": SIGNAL_SAMPLE.name,
            "Nominal": 1,
            "Bounds": [args.mu_min, args.mu_max],
        }
    ]


def sample_colors():
    """`colors` argument for cabinetry's data/MC plots, keyed by sample name"""
    return {sample.name: sample.color for sample in MC_SAMPLES}
