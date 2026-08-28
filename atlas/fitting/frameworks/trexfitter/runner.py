"""
Write and run a TRExFitter fit. TRExFitter writes its output into a directory named after
the `Job` block, i.e. `args.job_name`, relative to the working directory.
"""

import os

from ..samples import MC_SAMPLES
from . import config_blocks
from .systematics import write_systematic_blocks

NAME = "trexfitter"


def config_path(args):
    return os.path.join(args.config_outdir, f"{args.job_name}.config")


def write_config(args):
    """write the TRExFitter config for `args` and return its path"""
    os.makedirs(args.config_outdir, exist_ok=True)
    outpath = config_path(args)

    with open(outpath, "w") as f:
        config_blocks.write_job_block(f, args)
        config_blocks.write_fit_block(f, args)
        config_blocks.write_normfactor_blocks(f, args)
        config_blocks.write_region_block(f, args)
        config_blocks.write_sample_blocks(f, args)
        if not args.stats_only:
            write_systematic_blocks(f, args, MC_SAMPLES)

    return outpath


def run_fit(args, outpath):
    """run `trex-fitter` on a generated config"""
    command = f"trex-fitter {args.trex_opts} {outpath}"
    print(f"Running: {command}")
    os.system(command)
