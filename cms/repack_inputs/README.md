# repack_inputs

Distribute ROOT file repacking over Dask + XRootD.

Reads a fileset JSON listing remote `.root` inputs, splits and merges them into output files capped at N events each, optionally re-baskets / re-compresses, and writes them back to XRootD.

Standalone package. No dependency on `intccms`.

## What it does

For each chunk (one output file), one Dask task:

1. `xrdcp`s the chunk's input files into local scratch.
2. Slices / re-baskets / re-compresses each segment via the vendored `root_repack`.
3. Merges the segments into one local file.
4. `xrdcp`s the merged file straight to the final XRootD URL (with `-f` if `OVERWRITE` is set).
5. Cleans up local scratch on task exit.

## Quick start

### Notebook

Open `repack_inputs/notebook.ipynb`, set `CLIENT_ADDRESS` to your scheduler URL, edit the config cell (input JSON, output URL, `N_EVENTS`, etc.), run all cells. Set `DRY_RUN = True` first to inspect the plan without dispatching tasks.

### CLI

```
python -m repack_inputs \
    --scheduler tls://localhost:8786 \
    --input-json /path/to/fileset.json \
    --output-dir-url root://xrootd-host//store/user/me/repack_out \
    --output-fileset-json out_fileset.json \
    --n-events 1000000 \
    --max-scratch-gb 7
```

Add `--dry-run` to print the plan without dispatching tasks.

## Configuration

Every CLI flag has a Python kwarg of the same name on `run_repack`.

### Inputs and outputs

| Flag | Default | Description |
| --- | --- | --- |
| `--input-json` | required | Fileset JSON, see schema below |
| `--output-dir-url` | required | XRootD URL prefix for outputs |
| `--output-subdir` | `{dataset}/{systematic}` | Layout under the output URL prefix |
| `--output-fileset-json` | none | If set, write a JSON listing the successful outputs |

### Split / merge

| Flag | Default | Description |
| --- | --- | --- |
| `--n-events` | None | Cap events per output. None = one output per (dataset, systematic) |
| `--event-tree` | `Events` | TTree name |

### Tree re-encoding (triggers a per-input rewrite pass; raises scratch usage)

| Flag | Default | Description |
| --- | --- | --- |
| `--basket-size` | None | Single basket size for all branches, e.g. `64k` |
| `--basket-sizes PATTERN=SIZE` | none | Per-branch basket size, repeatable, e.g. `--basket-sizes Muon_*=128k` |
| `--auto-flush` | None | TTree AutoFlush: bytes (`30M`) or entries (int) |

### TFileMerger options

| Flag | Default | Description |
| --- | --- | --- |
| `--fast` | off | ROOT fast-merge mode |
| `--keep` | off | Keep input compression instead of re-compressing |
| `--sort` | `branch` | `branch` / `offset` / `entry` |
| `--compress` | `same` | Output compression, e.g. `zstd=9`, `lz4`, `same` |
| `--iofeatures NAME` | none | ROOT IOFeature, repeatable |
| `--verbose` | 0 | Repeat for more verbosity |

### Worker scratch

| Flag | Default | Description |
| --- | --- | --- |
| `--scratch-root` | `/tmp/repack_inputs` | Worker scratch directory |
| `--max-scratch-gb` | 7.0 | Hard cap on per-task scratch usage; raises if exceeded |

### Runtime

| Flag | Default | Description |
| --- | --- | --- |
| `--scheduler` | required | Dask scheduler address |
| `--overwrite` | off | `xrdcp -f`: overwrite existing outputs on XRootD |
| `--no-progress` | off | Disable the tqdm progress bar |
| `--dry-run` | off | Plan only; skip mkdir, dask submission, output JSON |

## Fileset JSON schema

Both input and output share the same shape:

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

The output JSON written by `--output-fileset-json` lists only the chunks that succeeded.

## Scratch budget guidance

Per task, peak local scratch is roughly:

```
sum(input file sizes) + sum(slice temps for sliced segments) + merged output size
```

Notes:

- Whole input files are downloaded even when only a slice is needed. A 1M-event chunk that comes from one larger input file means downloading the whole input.
- Slice temps only exist for segments where `start != 0` or `count != total_entries`, or when re-basketing/AutoFlush is requested.
- For 1M events of typical NanoAOD (~2.5 GB), expect ~5-8 GB peak. Set `--max-scratch-gb` at least 1 GB below your worker disk quota.

If you run with `--n-events` set and hit the scratch cap on big inputs, lower `--n-events` or raise the worker disk quota.

## XRootD caveats

- **Random writes are not supported by most CMS SEs.** `TFileMerger` writes random-access (it seeks back to write the file header at the end), so the merge happens locally. The merged file is then `xrdcp`-ed sequentially to the destination. We never write a ROOT file directly to XRootD.
- **No `.tmp + mv` atomicity.** Many CMS SEs disable user-level `mv` and `rm`. We `xrdcp` straight to the final URL with `-f` when `OVERWRITE` is set. If a worker crashes mid-`xrdcp`, the destination has a partial file; re-run with `--overwrite` to replace it.
- **`xrdcp` and `xrdfs` are subprocess calls.** Both must be on the worker's `PATH`. They're standard in CMS analysis facility images.

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

files = load_fileset("fileset.json")
plans = plan_chunks(files, output_dir_url="root://...", n_events=1_000_000)

with Client("tls://localhost:8786") as client:
    upload_package(client)             # ships repack_inputs to scheduler+workers
    prepare_output_dirs(plans)         # xrdfs mkdir -p per output dir
    results = run_repack(client, plans, max_scratch_gb=7.0, compress="zstd=9")

write_output_fileset(plans, results, "out_fileset.json")
```

`run_repack` returns a `dict[output_url, str | Exception]`: success values are the URL written, failure values are the raised exception (when `raise_on_error=False`).

## Requirements

- Python 3.10+
- ROOT (PyROOT) on driver and workers (the vendored `root_repack.py` does `import ROOT` at module load)
- `xrdcp` and `xrdfs` on each worker's `PATH`
- `dask`, `distributed`, `tqdm`

`upload_package(client)` (called from the CLI and notebook) ships the `repack_inputs/` source directory to the scheduler and every worker, so the package does not need to be pre-installed on the cluster image.
