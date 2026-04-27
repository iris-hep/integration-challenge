# repack_inputs

Repack ROOT files in parallel over Dask + XRootD.

## Requirements

- Python 3.10+
- ROOT (PyROOT) on the driver and on every worker
- `xrdcp` and `xrdfs` on each worker's `PATH`
- `dask`, `distributed`, `tqdm`

The package itself does not need to be pre-installed on the cluster: `upload_package(client)` (called by the CLI and the notebook) ships `repack_inputs/` to the scheduler and workers.

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Fileset JSON schema](#fileset-json-schema)
- [Library API](#library-api)
- [Scratch budget](#scratch-budget)
- [XRootD caveats](#xrootd-caveats)

## What it does

For each chunk (one output file), one Dask task does:

1. `xrdcp` the chunk's input files into local scratch.
2. Slice / re-basket each segment with the vendored `root_repack`.
3. Merge the segments into one local file.
4. `xrdcp` the merged file to the final XRootD URL.
5. Clean up scratch.

A sample fileset (125 datasets, 13518 files) is bundled at `repack_inputs/nanoaods.json`.

## Quick start

### Notebook

Open `repack_inputs/notebook.ipynb`. Set `CLIENT_ADDRESS` to your scheduler URL. Default `INPUT_JSON` is the bundled sample. Set `DRY_RUN = True` and run all cells to see the plan, then flip to `False` to dispatch.

### CLI

Dry-run:

```
python -m repack_inputs \
    --scheduler tls://localhost:8786 \
    --input-json repack_inputs/nanoaods.json \
    --output-dir-url root://xrootd-host//store/user/me/repack_out \
    --n-events 1000000 \
    --dry-run
```

Drop `--dry-run` once the plan looks right.

## Configuration

CLI flags map 1:1 to `run_repack` kwargs.

### Inputs and outputs

| Flag | Default | What it does |
| --- | --- | --- |
| `--input-json` | required | Fileset JSON path |
| `--output-dir-url` | required | XRootD URL prefix for outputs |
| `--output-subdir` | `{dataset}/{systematic}` | Layout under the output URL |
| `--output-fileset-json` | none | Write a JSON listing the successful outputs |

### Split / merge

| Flag | Default | What it does |
| --- | --- | --- |
| `--n-events` | None | Events per output. None = one output per (dataset, systematic) |
| `--event-tree` | `Events` | TTree name |

### Tree re-encoding

Triggers a per-input rewrite pass and raises scratch usage.

| Flag | Default | What it does |
| --- | --- | --- |
| `--basket-size` | None | Basket size for all branches, e.g. `64k` |
| `--basket-sizes PATTERN=SIZE` | none | Per-branch basket size, repeatable |
| `--auto-flush` | None | TTree AutoFlush: bytes (`30M`) or entries (int) |

### TFileMerger

| Flag | Default | What it does |
| --- | --- | --- |
| `--fast` | off | ROOT fast-merge mode |
| `--keep` | off | Keep input compression |
| `--sort` | `branch` | `branch` / `offset` / `entry` |
| `--compress` | `same` | Output compression, e.g. `zstd=9`, `lz4`, `same` |
| `--iofeatures NAME` | none | ROOT IOFeature, repeatable |
| `--verbose` | 0 | Repeat for more |

### Worker scratch

| Flag | Default | What it does |
| --- | --- | --- |
| `--scratch-root` | `/tmp/repack_inputs` | Where workers stage temps |
| `--max-scratch-gb` | 7.0 | Per-task cap; raises if exceeded |

### Runtime

| Flag | Default | What it does |
| --- | --- | --- |
| `--scheduler` | required | Dask scheduler address |
| `--overwrite` | off | `xrdcp -f` overwrites existing outputs |
| `--no-progress` | off | Disable the tqdm progress bar |
| `--dry-run` | off | Plan only; skip mkdir, dask, output JSON |

## Fileset JSON schema

Input and output share this shape:

```json
{
  "<dataset>": {
    "<systematic>": {
      "files": [
        {"path": "root://...", "nevts": 100000},
        {"path": "root://...", "nevts": 50000}
      ],
      "nevts_total": 150000
    }
  }
}
```

The output JSON lists only the chunks that succeeded.

## Library API

```python
from dask.distributed import Client
from repack_inputs import (
    load_fileset,
    plan_chunks,
    prepare_output_dirs,
    run_repack,
    upload_package,
    write_output_fileset,
)

files = load_fileset("repack_inputs/nanoaods.json")
plans = plan_chunks(files, output_dir_url="root://...", n_events=1_000_000)

with Client("tls://localhost:8786") as client:
    upload_package(client)
    prepare_output_dirs(plans)
    results = run_repack(client, plans, max_scratch_gb=7.0, compress="zstd=9")

write_output_fileset(plans, results, "out_fileset.json")
```

`run_repack` returns `dict[output_url, str | Exception]`. Success values are the URL written. Failures are the raised exception, but only when `raise_on_error=False`; otherwise the first failure cancels the rest and re-raises.

## Scratch budget

Per task, peak local scratch is roughly:

```
sum(input file sizes) + sum(slice temps) + merged output size
```

- Whole input files get downloaded even when only a slice is needed.
- Slice temps exist for segments where `start != 0` or `count != total_entries`, or when re-basketing is on.
- For 1M events of NanoAOD (~2.5 GB), expect 5-8 GB peak. Keep `--max-scratch-gb` at least 1 GB below your worker quota.

If a chunk hits the cap, lower `--n-events` or raise the quota.

## XRootD caveats

These are things we observed against `xrootd-local.unl.edu`. Other SEs may differ.

- `TFileMerger` writing direct to `root://` URLs failed. ROOT seeks back to write the file header at the end of a `TFile` and the SE rejected the random-access write. We merge to a local file and `xrdcp` that to the destination.
- `xrdfs mv` returned "permission denied" even between two paths the user could write. So no `<dst>.tmp` + rename: we `xrdcp` straight to the final URL with `-f` if `OVERWRITE` is set. A worker crashing mid-`xrdcp` leaves a partial file; re-run with `--overwrite` to replace.
- `xrdfs rm` sometimes failed with "Too many DFS write attempts". We don't try to clean up partials.
