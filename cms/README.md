# CMS integration challenge

A coffea-based distributed analysis framework for the CMS Z' &rarr; t&tbar; single-lepton search on Run 2 NanoAOD.

## Setup

```sh
pixi install
pixi run lab   # starts JupyterLab
```

On an analysis facility (AF), use the notebooks directly &mdash; they install dependencies (`coffea`, `roastcoffea`, `omegaconf`) into the AF environment at runtime.

## Configuration

All configuration lives under `example_cms/configs/`:

| File | Controls |
|------|----------|
| `configuration.py` | Master config &mdash; assembles everything, sets global flags and luminosity |
| `skim.py` | Dataset definitions (files, xsecs, redirectors) and skimming settings (selection, output format) |
| `cuts.py` | Event selection functions (signal regions, control regions) |
| `observables.py` | Histogram definitions (binning, observable functions) and ghost variables for MVA |
| `systematics.py` | Corrections and systematic uncertainties, keyed by year |

### How is the master config structured?

`configuration.py` builds a single dict with these top-level keys:

```python
config = {
    "general": { ... },       # run flags, luminosity, output paths
    "datasets": { ... },      # dataset list, max_files, skip_files
    "preprocess": { ... },    # skimming settings, branches to keep
    "channels": [ ... ],      # analysis channels (selection + observables)
    "corrections": { ... },   # year-keyed correction configs
    "ghost_observables": [...],  # MVA training variables
}
```

### How do I modify config in a notebook?

Deepcopy the config and override what you need:

```python
from example_cms.configs.configuration import config as original_config
import copy

config = copy.deepcopy(original_config)

# Common overrides
config["datasets"]["max_files"] = 5           # limit input for testing
config["general"]["output_dir"] = "my_outputs/"
config["general"]["run_metadata_generation"] = False  # reuse cached metadata
config["general"]["run_analysis"] = True
config["general"]["save_skimmed_output"] = False
config["general"]["run_histogramming"] = True
config["general"]["run_systematics"] = False   # skip systematics for speed

# Validate with pydantic
from intccms.schema import Config, load_config_with_restricted_cli
full_config = load_config_with_restricted_cli(config, [])
validated_config = Config(**full_config)
```

## Datasets

### How are datasets defined?

In `skim.py`, each dataset specifies:

```python
{
    "name": "ttbar_semilep",
    "directories": ("TTToSemiLeptonic/.../2016", "TTToSemiLeptonic/.../2017", ...),
    "cross_sections": (365.34, 365.34, ...),  # one per directory/year
    "redirector": "root://xcache/",
    "tree_name": "Events",
    "weight_branch": "genWeight",
    "is_data": False,
}
```

Data datasets set `is_data=True`, `weight_branch=None`, and include `lumi_mask` configuration. HT-binned samples (W+jets, DY+jets) have multiple entries per process.

### How do I change the redirector?

Override per dataset after validation:

```python
for dataset in validated_config.datasets.datasets:
    dataset.redirector = "root://xcache/"
```

## Metadata generation

### What is it and when do I need to re-run it?

Metadata generation runs coffea preprocessing: enumerates files, counts events, extracts cross-sections, and builds chunked workitems. It produces a `metadata_lookup` dict that the processor uses for per-dataset normalization.

Set `config["general"]["run_metadata_generation"] = True` to run it (requires a Dask client). Results are cached locally, so set it to `False` on subsequent runs.

**Re-run when**: datasets change, files are added/removed, or you switch redirectors.

## Workflow modes

Two flags control what the processor does:

| Mode | `save_skimmed_output` | `run_analysis` | Use when |
|------|-----------------------|----------------|----------|
| Skim + Analysis | `True` | `True` | Production: save skims and fill histograms in one pass |
| Analysis only | `False` | `True` | Quick iteration: filter on-the-fly, no disk I/O |
| Skim only | `True` | `False` | Prepare skims for later analysis or sharing |
| Analysis on skims | `False` | `True` | Re-analyze saved skims (`use_skimmed_input=True`) |

## Skimming

### How do I skim events?

The selection function is defined in `skim.py`:

```python
"skimming": {
    "function": default_skim_selection,  # from cuts.py
    "use": [("PuppiMET", None), ("HLT", None)],
    "output": {
        "format": "parquet",              # or "ttree", "rntuple"
        "output_dir": "example_cms/outputs/skimmed/",
    },
}
```

Branch selection is controlled by `config["preprocess"]["branches"]` and `config["preprocess"]["mc_branches"]`. Only listed branches are saved to disk.

### What output formats are available?

| `format` | Writer | Destination examples | Notes |
|----------|--------|---------------------|-------|
| `parquet` | `ak.to_parquet` | `./skimmed/`, `s3://bucket/path` | Default. Supports compression via `to_kwargs` (e.g. `{"compression": "zstd"}`) |
| `ttree` | `uproot.WritableDirectory` | `./skimmed/`, `root://xrootd.server/path` | ROOT TTree. Single-threaded writes. |
| `rntuple` | `uproot.mkrntuple` | `./skimmed/`, `root://xrootd.server/path` | ROOT RNTuple (experimental). |

Set the format and destination in the output stanza:

```python
config["preprocess"]["skimming"]["output"]["format"] = "parquet"
config["preprocess"]["skimming"]["output"]["output_dir"] = "s3://my-bucket/skims/"
config["preprocess"]["skimming"]["output"]["to_kwargs"] = {"compression": "zstd"}
```

### How do I read back skimmed files?

Set `use_skimmed_input=True` in the config. The runner auto-discovers skimmed files from the output directory and builds the fileset.

## The processor

### Where does it live?

`src/intccms/analysis/processors.py` has two processors:

- **`SkimAndAnalyseProcessor`**: The main processor. Per chunk, it: (1) applies skim selection, (2) saves filtered events to disk if enabled, (3) runs analysis (corrections + histogramming) if enabled. All controlled by the config flags above.
- **`TwoHundredGbpsProcessor`**: Minimal I/O benchmark. Reads and materializes configured branches, returns event count only.

Both are passed to `run_processor_workflow()` in `src/intccms/analysis/runner.py`.

### How do I write a custom processor?

Inherit from `ProcessorABC` and pass your instance to the workflow:

```python
from coffea.processor import ProcessorABC
from roastcoffea import track_metrics

class MyProcessor(ProcessorABC):
    def __init__(self, config, output_manager, metadata_lookup):
        self.config = config

    @property
    def accumulator(self):
        return {"events": 0}

    @track_metrics  # optional: enables roastcoffea metrics
    def process(self, events):
        # events is an ak.Array with NanoAODSchema
        return {"events": len(events)}

    def postprocess(self, accumulator):
        return accumulator

# In the notebook:
processor = MyProcessor(validated_config, output_manager, metadata_lookup)
output, report = run_processor_workflow(
    config=validated_config,
    output_manager=output_manager,
    metadata_lookup=metadata_lookup,
    processor=processor,
    workitems=workitems,
    executor=DaskExecutor(client=client),
    schema=NanoAODSchema,
)
```

## Histogramming

### How are histograms configured?

Observables are defined in `observables.py`:

```python
{
    "name": "workshop_mtt",
    "binning": "200,3000,20",       # "xmin,xmax,nbins"
    "function": get_mtt,            # computes the observable from event arrays
    "use": [("Muon", ...), ("Jet", ...), ("FatJet", ...), ("PuppiMET", ...)],
}
```

Channels (in `configuration.py`) tie a selection function to a list of observables:

```python
{
    "name": "Zprime_1tag_SR",
    "selection": {"function": cuts.Zprime_1tag_SR, "use": [...]},
    "observables": LIST_OF_VARS,
}
```

### Where does histogramming happen?

`CMSAnalysis.histogramming()` in `src/intccms/analysis/cms.py`. For each chunk, it:

1. Applies the channel selection
2. Computes event weights (genWeight &times; xsec normalization &times; correction SFs)
3. Evaluates the observable function
4. Fills a `hist.Hist` with axes: observable (Variable), process (StrCategory), variation (StrCategory)

### Where are histograms saved?

In `postprocess()`, histograms are saved to:

- `processor_histograms.pkl` &mdash; for fast re-loading without re-processing (set `run_processor=False`)
- `histograms.root` &mdash; for downstream statistical tools (cabinetry)

Both go to `output_manager.histograms_dir`.

## Corrections and systematics

### How are corrections configured?

`systematics.py` defines a year-keyed dict:

```python
corrections_config = {
    "2016preVFP": [correction1, correction2, ...],
    "2017": [...],
    "2018": [...],
}
```

Each correction has a type that determines how it is applied:

- **Object-level** (`type: "object"`): modifies physics quantities (e.g. jet pT via JEC). Changes propagate through selection and histogramming.
- **Event-weight** (`type: "event"`): multiplies the event weight (e.g. pileup reweighting, muon SFs, b-tag SFs).

Each correction can carry `uncertainty_sources` &mdash; a list of up/down variations that produce separate histogram fills.

### What corrections are implemented?

| Category | Type | Sources | Correlation |
|----------|------|---------|-------------|
| Muon (ID, ISO, trigger) | event | 3 | correlated across years |
| Pileup | event | 1 per year | decorrelated |
| JEC | object | 17 correlated + 10 decorrelated | mixed |
| B-tagging (deepJet) | event | 4 shape + 8 stats + 27 JES-linked | mixed |

**Correlated** sources (no year suffix, e.g. `muon_id_sf`) apply the same variation across all years. **Decorrelated** sources (year-suffixed, e.g. `pileup_2017`) are independent per year &mdash; other years get filled with nominal.

### How do I add a new correction?

Add a dict to the year's list in `systematics.py`:

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

For object-level corrections, set `type: "object"` and provide `nominal_function` / `up_function` / `down_function`. JES b-tag sources can co-vary with JEC via the `varies_with` field.

## Metrics and profiling

### How do I collect performance metrics?

Wrap the workflow in roastcoffea's `MetricsCollector`:

```python
from roastcoffea import MetricsCollector

with MetricsCollector(
    client=client,
    processor_instance=processor,
    track_workers=True,
    worker_tracking_interval=1.0,
) as collector:
    output, report = run_processor_workflow(...)
    collector.extract_metrics_from_output(output)
    collector.set_coffea_report(report)

metrics = collector.get_metrics()
tracking_data = collector.tracking_data
```

The processor already uses `@track_metrics`, `track_time`, and `track_memory` decorators from roastcoffea &mdash; these are picked up automatically by the collector.

roastcoffea provides summary tables (`format_throughput_table`, `format_resources_table`, etc.) and timeline plots (worker count, throughput, CPU/memory utilization). See `full_run_with_metrics.ipynb` for the full pattern.

### How do I profile worker activity?

Capture Dask's statistical call stack profile while the client is alive:

```python
dask_profile = client.profile()
```

Render as an interactive flamegraph:

```python
from distributed.profile import plot_data, plot_figure
from bokeh.plotting import save, output_file

data = plot_data(dask_profile)
fig, source = plot_figure(data, width=1600, height=800)
output_file("dask_profile.html")
save(fig)
```

This shows where worker CPU time is spent (decompression, network I/O, array construction, etc.). The "time" values are cumulative across all workers.

## Notebooks

| Notebook | Use when |
|----------|----------|
| `full_run.ipynb` | Standard analysis workflow (metadata &rarr; processing &rarr; histograms) |
| `full_run_with_metrics.ipynb` | Same workflow with roastcoffea dashboards |
| `full_run_with_skimming.ipynb` | Demonstrates all four workflow modes |
| `full_run_200gbps.ipynb` | I/O throughput benchmark with profiling |
| `input_inspector.ipynb` | Characterize inputs (event counts, branch sizes, compression) |

## Credentials

Storage credentials (AWS keys for S3) go in an untracked `.env` file. Load early with:

```python
from intccms.utils.tools import load_dotenv
load_dotenv()
```

Forward credentials to Dask workers with `propagate_aws_env=True` in `acquire_client`. For skimming output, use `WorkerEval` to defer credential resolution to the worker:

```python
from intccms.schema.base import WorkerEval
"storage_options": {
    "key": WorkerEval(lambda: os.environ["AWS_ACCESS_KEY_ID"]),
    "secret": WorkerEval(lambda: os.environ["AWS_SECRET_ACCESS_KEY"]),
}
```
