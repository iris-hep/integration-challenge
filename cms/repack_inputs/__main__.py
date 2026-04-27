"""CLI driver for ``repack_inputs``.

Run with::

    python -m repack_inputs \\
        --scheduler tls://localhost:8786 \\
        --input-json /path/to/fileset.json \\
        --output-dir-url root://xrootd-host//store/user/me/repack_out \\
        --output-fileset-json out_fileset.json \\
        --n-events 1000000

Add ``--dry-run`` to print the plan without dispatching tasks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from dask.distributed import Client

from repack_inputs.distributed import (
    load_fileset,
    plan_chunks,
    prepare_output_dirs,
    run_repack,
    upload_package,
    write_output_fileset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distribute ROOT file repacking over Dask + XRootD."
    )

    # Required
    parser.add_argument(
        "--input-json",
        required=True,
        help="path to fileset JSON (with pre-computed nevts per file)",
    )
    parser.add_argument(
        "--output-dir-url",
        required=True,
        help="XRootD URL prefix for outputs",
    )
    parser.add_argument(
        "--scheduler",
        required=True,
        help="Dask scheduler address, e.g. tls://localhost:8786",
    )

    # Optional outputs
    parser.add_argument(
        "--output-subdir",
        default="{dataset}/{systematic}",
        help="layout under --output-dir-url (default: %(default)s)",
    )
    parser.add_argument(
        "--output-fileset-json",
        default=None,
        help="if set, write a JSON listing the successful outputs",
    )

    # Split / merge
    parser.add_argument(
        "--n-events",
        type=int,
        default=None,
        help="cap events per output (None = one output per dataset/systematic)",
    )
    parser.add_argument("--event-tree", default="Events")

    # Tree re-encoding
    parser.add_argument(
        "--basket-size",
        default=None,
        help="single basket size for all branches, e.g. 64k",
    )
    parser.add_argument(
        "--basket-sizes",
        action="append",
        default=None,
        metavar="PATTERN=SIZE",
        help="per-branch basket size, repeatable, e.g. --basket-sizes Muon_*=128k",
    )
    parser.add_argument(
        "--auto-flush",
        default=None,
        help="TTree AutoFlush, e.g. 30M (bytes) or an int (entries)",
    )

    # TFileMerger options
    parser.add_argument("--fast", action="store_true", help="ROOT fast-merge mode")
    parser.add_argument("--keep", action="store_true", help="keep input compression")
    parser.add_argument(
        "--sort",
        choices=["branch", "offset", "entry"],
        default="branch",
        help="basket sorting mode (default: %(default)s)",
    )
    parser.add_argument(
        "--compress",
        default="same",
        help="output compression, e.g. zstd=9, lz4, same (default: %(default)s)",
    )
    parser.add_argument(
        "--iofeatures",
        action="append",
        default=None,
        help="ROOT IOFeature, repeatable",
    )
    parser.add_argument("--verbose", action="count", default=0)

    # Worker scratch
    parser.add_argument(
        "--scratch-root",
        default="/tmp/repack_inputs",
        help="worker scratch directory (default: %(default)s)",
    )
    parser.add_argument(
        "--max-scratch-gb",
        type=float,
        default=7.0,
        help="hard cap on per-task scratch usage (default: %(default)s)",
    )

    # Runtime
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="if set, xrdcp -f overwrites existing outputs on XRootD",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the tqdm progress bar",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and summarise only; skip mkdir, dask submission, output JSON",
    )

    return parser


def _print_plan_preview(plans, limit: int = 10) -> None:
    print(f"planned {len(plans)} chunks, {sum(p.total_events for p in plans):,} events")
    for plan in plans[:limit]:
        print(
            f"  {plan.output_url}  "
            f"({plan.total_events:,} events from {len(plan.unique_sources)} files)"
        )
    if len(plans) > limit:
        print(f"  ... ({len(plans) - limit} more)")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    files = load_fileset(args.input_json)
    print(
        f"loaded {len(files)} files across "
        f"{len({(f.dataset, f.systematic) for f in files})} dataset/systematic pairs, "
        f"{sum(f.nevts for f in files):,} events"
    )

    plans = plan_chunks(
        files,
        output_dir_url=args.output_dir_url,
        n_events=args.n_events,
        output_subdir=args.output_subdir,
    )
    _print_plan_preview(plans)

    if args.dry_run:
        return 0

    prepare_output_dirs(plans)
    print(f"mkdir -p done for output directories")

    with Client(args.scheduler) as client:
        upload_package(client)
        results = run_repack(
            client,
            plans,
            scratch_root=args.scratch_root,
            max_scratch_gb=args.max_scratch_gb,
            overwrite=args.overwrite,
            event_tree=args.event_tree,
            basket_size=args.basket_size,
            basket_sizes=args.basket_sizes,
            auto_flush=args.auto_flush,
            fast=args.fast,
            keep=args.keep,
            sort=args.sort,
            compress=args.compress,
            iofeatures=args.iofeatures,
            verbose=args.verbose,
            progress=not args.no_progress,
        )

    n_ok = sum(1 for v in results.values() if isinstance(v, str))
    n_fail = len(results) - n_ok
    print(f"wrote {n_ok}/{len(results)} outputs ({n_fail} failed)")
    for url, outcome in results.items():
        if not isinstance(outcome, str):
            print(f"  FAIL {url}: {outcome!r}")

    if args.output_fileset_json:
        out = write_output_fileset(plans, results, args.output_fileset_json)
        print(f"output fileset JSON: {out}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
