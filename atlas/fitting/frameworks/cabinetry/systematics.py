"""
cabinetry `Systematics` blocks, built from the systematics found in the input file
(discovered in `frameworks.inputs`, shared with the TRExFitter configuration).

Every variation is a `NormPlusShape` systematic, the equivalent of TRExFitter's
`Type: HISTO`: a `normsys` plus a `histosys` modifier in the pyhf workspace. One-sided
variations get `Down: {Symmetrize: true}`, so cabinetry builds the down template as
`2 * nominal - up` instead of reading a histogram.
"""

from ..inputs import DOWN_SUFFIX, UP_SUFFIX, discover_systematics, print_summary


def systematic_blocks(args, samples):
    systematics = discover_systematics(args, samples)

    blocks = []
    for name, syst_samples, onesided in systematics:
        blocks.append(
            {
                "Name": name,
                "Type": "NormPlusShape",
                "Samples": syst_samples,
                "Regions": args.region_name,
                "Up": {"VariationPath": f"{name}{UP_SUFFIX}"},
                "Down": (
                    {"Symmetrize": True}
                    if onesided
                    else {"VariationPath": f"{name}{DOWN_SUFFIX}"}
                ),
            }
        )

    print_summary(systematics)
    return blocks
