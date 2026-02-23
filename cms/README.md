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

## Running the workflows
Three notebooks cover the common workflows:

1. **Full processing** – runs the production chain (metadata → skimming → histogramming) over Run‑2.
2. **Processing + metrics** – identical pipeline plus worker/throughput dashboards (uses `metrics.worker_tracker`).
3. **Input inspector** – characterises inputs (event counts, branch/compression sizes, optional Rucio-backed file sizes) without running the full analysis.


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
