# IRIS-HEP CMS integration challenge

## Environment (pixi)
This project uses [pixi](https://pixi.sh/) for environment management. To set up the environment, run:

```sh
pixi install
source pixi_activate.sh     # optional helper for a conda-like shell
```

For notebooks, start JupyterLab via:

```sh
pixi run lab
```

## Code structure
``` 
cms/
├── example_cms/                # CMS-style configs, cuts, datasets
├── example_opendata/           # Open-data configs
├── pixi.toml / pixi.lock       # Environment definition
└── src/intccms/
    ├── analysis/               # Runner, processor glue
    ├── metrics/                # Inspector + metrics tooling
    ├── utils/                  # Output manager, schema, skimming helpers, …
    └── ...
```
Use the example configurations as templates—they already tie together datasets, skimming, channels, and systematics. The notebooks show how to drive the full pipeline interactively.

## Metadata and preprocessing
Metadata extraction and preprocessing are handled before the main analysis. Metadata includes information about datasets, event counts, and cross-sections, and is used to configure the analysis and normalization. Preprocessing steps may include filtering, object selection, and preparing input files for skimming and analysis.

## Skimming
Preprocessing is controlled by the `preprocess` block in the configuration. The `skimming` subsection now uses a single `output` stanza to steer how skimmed NanoAOD chunks are persisted:

```python
preprocess = {
    "skimming": {
        "function": default_skim_selection,
        "use": [("PuppiMET", None), ("HLT", None)],
        "output": {
            "format": "parquet",          # other options: ttree, rntuple
            "output_dir": "s3://bucket/path",  # optional; defaults to local skimmed_dir
            "to_kwargs": {"compression": "zstd"},   # forwarded to ak.to_parquet
            "from_kwargs": {"storage_options": {...}}  # forwarded to NanoEventsFactory.from_parquet
        },
    },
}
```

The file suffix is fixed to `{dataset}/file_{index}/part_{chunk}.{ext}`, so switching between local and remote storage only requires changing `output_dir`.

- Set `general.run_skimming=True` to regenerate skims. Use `datasets.max_files` to limit input size when experimenting.
- Downstream steps load the same path, so no separate cache copy is needed; cached Awkward objects are still produced automatically for faster reruns.
- Dataset-level options such as lumi masks live next to each dataset definition (for example `lumi_mask`: `{ "function": cuts.lumi_mask, "use": [...], "static_kwargs": {"lumifile": "...json"} }`).

### Quick start: workflow modes

The processor supports four workflow modes, controlled by two config flags (`save_skimmed_output` and `run_analysis`):

| Mode | `save_skimmed_output` | `run_analysis` | Description |
|------|-----------------------|----------------|-------------|
| Skim + Analysis | `True` | `True` | Skim events to disk **and** run histogramming in one pass |
| Analysis only | `False` | `True` | Apply skim filter on-the-fly, no files saved |
| Skim only | `True` | `False` | Save skimmed files to disk, skip histogramming |
| Analysis on skimmed | `False` | `True` | Read previously skimmed files as input (set `use_skimmed_input=True`) |

See `full_run_with_skimming.ipynb` for a notebook that runs all four modes with performance metrics.

### Output format and destination

The `output` stanza controls serialization and storage:

| `format` | `output_dir` example | Notes |
|-----------|----------------------|-------|
| `parquet` | `./skimmed/` or `s3://bucket/path` | Default; uses `ak.to_parquet` / `ak.from_parquet` |
| `ttree` | `root://xrootd.server/path` | ROOT TTree via `uproot.WritableDirectory` |
| `rntuple` | `root://xrootd.server/path` | ROOT RNTuple (experimental) |

### Secrets and credentials

Storage credentials (e.g. AWS keys for S3) belong in an **untracked `.env` file** at the repo root — never hardcode them in notebooks or config modules. See `.env.example` for the expected variable names. Load them early with:

```python
from intccms.utils.tools import load_dotenv
load_dotenv()  # reads .env into os.environ
```

Dask workers run in separate processes and don't inherit the client's environment. Pass `propagate_aws_env=True` to `acquire_client` so credentials are forwarded to every worker via a built-in plugin:

```python
from intccms.utils.dask_client import acquire_client
with acquire_client(AF, propagate_aws_env=True) as (client, cluster):
    ...
```

For skimming output to remote storage, use `WorkerEval` to defer credential resolution until the code actually runs on the worker (avoiding serialization of raw secrets):

```python
from intccms.schema.base import WorkerEval
"storage_options": {
    "key": WorkerEval(lambda: os.environ["AWS_ACCESS_KEY_ID"]),
    "secret": WorkerEval(lambda: os.environ["AWS_SECRET_ACCESS_KEY"]),
}
```

## Systematics

Systematic uncertainties are configured per year in `example_cms/configs/systematics.py`. Each correction can carry nominal scale factors and one or more uncertainty sources (up/down variations).

### Correction types

- **Object-level** (`type: "object"`): modify a physics quantity (e.g. jet pT) and propagate through selection and histogramming.
- **Event-weight** (`type: "event"`): multiply the event weight (e.g. pileup reweighting, b-tag SFs).

### Year-keyed (decorrelated) systematics

When the corrections config is a `dict` keyed by year (e.g. `{"2016preVFP": [...], "2017": [...], "2018": [...]}`), sources whose names include a year suffix (e.g. `pileup_2017`, `jesAbsoluteStat_2018`) are decorrelated: each year's variation is independent. Sources without a year suffix (e.g. `muon_id_sf`, `jesFlavorQCD`) are correlated across all years.

When processing year Y, the processor automatically fills other years' decorrelated variation names with nominal content so that merged histograms are complete.

### Currently implemented corrections

| Category | Sources | Correlation | Notes |
|----------|---------|-------------|-------|
| **Muon** | ID SF, ISO SF, trigger SF | 3 correlated | correctionlib, leading muon only |
| **Pileup** | pileup reweighting | 3 decorrelated (1/year) | correctionlib, `nTrueInt` |
| **JEC** | L1FastJet + L2Relative nominal, 27 uncertainty sources | 17 correlated + 10 decorrelated | custom functions wrapping correctionlib |
| **B-tagging** | deepJet_shape nominal, 35 uncertainty sources | 4 shape + 4 stats (decorrelated) + 27 JES-linked | JES btag sources co-vary with JEC via `varies_with` |

### Adding a correction with uncertainties

```python
{
    "name": "pileup_2017",
    "file": "path/to/pileup.json.gz",
    "type": "event",
    "args": [ObjVar("Pileup", "nTrueInt"), SYS],
    "op": "mult",
    "key": "Collisions17_UltraLegacy_goldenJSON",
    "use_correctionlib": True,
    "nominal_idx": "nominal",
    "uncertainty_sources": [
        {"name": "pileup_2017", "up_and_down_idx": ["up", "down"]},
    ],
}
```

## Running the workflows
Four notebooks cover the common workflows:

1. **Full processing** (`full_run.ipynb`) – runs the production chain (metadata → skimming → histogramming) over Run‑2.
2. **Processing + metrics** (`full_run_with_metrics.ipynb`) – identical pipeline plus worker/throughput dashboards (uses [roastcoffea](https://github.com/MoAly98/roastcoffea)).
3. **Skimming workflows** (`full_run_with_skimming.ipynb`) – demonstrates all four skimming workflow modes with per-mode performance metrics.
4. **Input inspector** – characterises inputs (event counts, branch/compression sizes, optional Rucio-backed file sizes) without running the full analysis.


```sh
python3 analysis.py general.run_skimming=False general.read_from_cache=True general.run_mva_training=False general.run_plots_only=False general.run_metadata_generation=False
```

## Developer guidance
Some practical knobs you’re likely to tweak:

- **Executors**  
  - Local Dask (default): set up via `analysis.runner` which spawns a `LocalCluster`.
  - Remote Dask/coffea executors: write a client factory (e.g., `Client("tcp://scheduler:8786")`) and pass it through the runner, or switch coffea backends via CLI flags (`analysis.py general.executor="futures"`).
  - For custom setups, create your own `Client` wrapper (set env vars, install dependencies) and hand it to the runner before calling `run()`.

- **Skimming mode**  
  - Toggle `general.run_skimming` / `general.use_skimmed_input` to control whether new skims are produced or existing ones are consumed.
  - Adjust `preprocess.skimming.output` (`format`, `output_dir`) to change serialization (parquet, ROOT, etc.) and storage destination. Use the `inspector.rucio` helper if you want accurate file sizes when skims live remotely.
  - To swap in custom logic, provide your own `preprocess.skimming.function` and `use` arguments (mirrors coffea processors) or set `general.analysis="skip"` and plug in a bespoke skimmer.

- **Redirectors**  
  - Configured per dataset (`redirector` field). Override them to point at XCache/local storage as needed; `DatasetManager.get_redirector()` exposes the current value.
  - When changing redirectors remember to regenerate metadata/skims so downstream references stay in sync.

- **Rucio helpers**  
  - `src/intccms/metrics/inspector/rucio.py` can resolve datasets via the shared queries logic and fetch byte counts using the official client. Feed its output into `aggregate_statistics(..., size_summary=...)` or `plot_file_size_distribution(..., size_summary=...)`.

- **Custom client setup**  
  - For bespoke clusters, write a helper that creates the `Client` you want (e.g., Dask Gateway, SLURM, YARN) and pass it into the analysis runner. Ensure environment packages match `pixi.toml` (use `pixi export` or `conda-pack` if needed).
  - When running under coffea-casa, use the provided notebooks—cells already demonstrate how to register plugins, set env vars, and integrate the metrics dashboard.
