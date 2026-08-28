"""
TRExFitter `Systematic` blocks, built from the systematics found in the input file
(discovered in `frameworks.inputs`, shared with the cabinetry configuration). One-sided
variations are written with `Symmetrisation: ONESIDED`, i.e. the up variation only.
"""

from ..inputs import DOWN_SUFFIX, UP_SUFFIX, category, discover_systematics, print_summary


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
    f.write(f'  Category: "{category(name)}"\n')
    f.write(f"  NuisanceParameter: {name}\n")
    f.write("\n")


def write_systematic_blocks(f, args, samples):
    systematics, pruned = discover_systematics(args, samples)

    f.write("% --------------------- %\n")
    f.write("% ---  Systematics  --- %\n")
    f.write("% --------------------- %\n\n")

    current_category = None
    for name, syst_samples, onesided in systematics:
        if category(name) != current_category:
            current_category = category(name)
            f.write(f"% --- {current_category} --- %\n\n")
        _write_histo_syst(f, name, syst_samples, onesided, args.region_name)

    print_summary(systematics, pruned, args.syst_pruning_threshold)
