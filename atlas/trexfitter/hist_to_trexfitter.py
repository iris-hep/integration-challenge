"""Convert the multi-dimensional analysis histogram into TRExFitter-style ROOT inputs."""

import argparse
import gzip
import json
import pathlib

import hist
import numpy as np
import uhi.io.json
import uproot

UP_SUFFIX = "__1up"
DOWN_SUFFIX = "__1down"


def load_histogram(path: str) -> hist.Hist:
    """read the gzipped json histogram written by the notebook"""
    with gzip.open(path) as f:
        return hist.Hist(json.loads(f.read(), object_hook=uhi.io.json.object_hook))


def pair_variations(variations, nominal: str = "NOSYS") -> dict[str, dict[str, str]]:
    """group variation axis labels into systematics with up/down components

    Names ending in `__1up` / `__1down` are paired up, everything else is treated as a
    one-sided systematic (needs `Symmetrisation: ONESIDED` in the TRExFitter config).
    """
    systematics: dict[str, dict[str, str]] = {}
    for variation in variations:
        if variation == nominal:
            continue

        if variation.endswith(UP_SUFFIX):
            base, direction = variation[: -len(UP_SUFFIX)], "up"
        elif variation.endswith(DOWN_SUFFIX):
            base, direction = variation[: -len(DOWN_SUFFIX)], "down"
        else:
            base, direction = variation, "up"  # one-sided variation

        systematics.setdefault(base, {})[direction] = variation

    return dict(sorted(systematics.items()))


def flatten(h: hist.Hist) -> hist.Hist:
    """unroll all remaining axes into a single 1d histogram (no-op if already 1d)"""
    if h.ndim == 1:
        return h

    num_bins = int(np.prod(h.shape))
    label = " x ".join(ax.label or ax.name for ax in h.axes)
    flat = hist.Hist(
        hist.axis.Regular(num_bins, 0, num_bins, name="bin", label=label),
        storage=h.storage_type(),
    )
    flat.view(flow=False)[...] = h.view(flow=False).reshape(num_bins)
    return flat


def convert(
    h: hist.Hist,
    output_path: str,
    *,
    region: str = "SR",
    sample_axis: str = "category",
    variation_axis: str = "variation",
    nominal: str = "NOSYS",
    data_samples=("data",),
    directory_format: str = "{region}_{sample}",
    nominal_name: str = "nominal",
    variation_prefix: str = "",
    skip_empty: bool = True,
) -> dict:
    """write flattened 1d histograms for every sample and systematic to a ROOT file

    Returns a summary dict with, per sample, the nominal yield and the systematics that
    were written / skipped (systematics are skipped when the variation is empty, which
    happens for data and for samples a given variation was never filled for).
    """
    samples = list(h.axes[sample_axis])
    systematics = pair_variations(list(h.axes[variation_axis]), nominal=nominal)
    summary = {}

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with uproot.recreate(output_path) as fout:
        for sample in samples:
            directory = directory_format.format(region=region, sample=sample)
            fout.mkdir(directory)

            h_nominal = flatten(h[{sample_axis: sample, variation_axis: nominal}])
            fout[f"{directory}/{nominal_name}"] = h_nominal

            written, skipped = [], []
            if sample not in data_samples:
                for syst, components in systematics.items():
                    histograms = {}
                    for direction, variation in components.items():
                        h_var = flatten(h[{sample_axis: sample, variation_axis: variation}])
                        if skip_empty and not np.any(h_var.values()):
                            break  # incomplete systematic, do not write either side
                        histograms[direction] = h_var

                    if len(histograms) != len(components):
                        skipped.append(syst)
                        continue

                    for direction, h_var in histograms.items():
                        name = f"{variation_prefix}{syst}_{direction.capitalize()}"
                        fout[f"{directory}/{name}"] = h_var
                    written.append(syst)

            summary[sample] = {
                "directory": directory,
                "yield": float(h_nominal.values().sum()),
                "bins": h_nominal.axes[0].size,
                "one_sided": sorted(s for s in written if "down" not in systematics[s]),
                "written": written,
                "skipped": skipped,
            }

    return summary


def print_summary(summary: dict, output_path: str) -> None:
    """report what ended up in the ROOT file"""
    print(f"wrote {output_path}")
    for sample, info in summary.items():
        print(
            f"  {info['directory']:<30} yield {info['yield']:>14,.1f}  "
            f"{info['bins']:>3} bins  {len(info['written']):>3} systematics"
            + (f" ({len(info['skipped'])} skipped: empty)" if info["skipped"] else "")
        )

    one_sided = sorted({s for info in summary.values() for s in info["one_sided"]})
    if one_sided:
        print(f"\none-sided systematics (need `Symmetrisation: ONESIDED`): {', '.join(one_sided)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="hist.json.gz written by the analysis notebook")
    parser.add_argument("-o", "--output", 
                        help="output ROOT file (default: trexfitter.root next to the input)")
    parser.add_argument("--region", default="SR", 
                        help="region name used in the directory names")
    parser.add_argument("--sample-axis", default="category")
    parser.add_argument("--variation-axis", default="variation")
    parser.add_argument("--nominal", default="NOSYS", help="label of the nominal variation")
    parser.add_argument("--data-sample", action="append", default=None, 
                        help="sample without systematics (repeatable, default: data)")
    parser.add_argument("--directory-format", default="{region}_{sample}")
    parser.add_argument("--nominal-name", default="nominal", 
                        help="name of the nominal histogram in each directory")
    parser.add_argument("--variation-prefix", default="", 
                        help="prefix for variation histogram names, e.g. 'nominal_'")
    parser.add_argument("--keep-empty", action="store_true", 
                        help="also write variations that are empty")
    args = parser.parse_args()

    output = args.output or str(pathlib.Path(args.input).parent / "trexfitter.root")
    h = load_histogram(args.input)

    summary = convert(
        h,
        output,
        region=args.region,
        sample_axis=args.sample_axis,
        variation_axis=args.variation_axis,
        nominal=args.nominal,
        data_samples=tuple(args.data_sample or ["data"]),
        directory_format=args.directory_format,
        nominal_name=args.nominal_name,
        variation_prefix=args.variation_prefix,
        skip_empty=not args.keep_empty,
    )
    print_summary(summary, output)


if __name__ == "__main__":
    main()
