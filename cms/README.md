# CMS integration challenge

A coffea-based distributed analysis framework for the CMS Z' → tt̄ single-lepton search on Run 2 NanoAOD.

**Contents**
- [Setup](#setup)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Metadata generation](#metadata-generation)
- [Handling bad files](#handling-bad-files)
- [Workflow modes](#workflow-modes)
- [Skimming](#skimming)
- [The processor](#the-processor)
- [Histogramming](#histogramming)
- [Corrections and systematics](#corrections-and-systematics)
- [Metrics and profiling](#metrics-and-profiling)
- [Notebooks](#notebooks)
- [Credentials](#credentials)

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

### [Advanced] What are the components of the preprocessing workflow?

The metadata extraction pipeline lives in `src/intccms/metadata_extractor/` and has four components:

| Module | Class/Functions | Role |
|--------|----------------|------|
| `manager.py` | `DatasetMetadataManager` | Top-level orchestrator. Calls the builder and extractor in sequence, caches results as JSON, and builds `metadata_lookup` for the processor. |
| `builders.py` | `FilesetBuilder` | Reads dataset configs from `DatasetManager`, enumerates file paths (applying `skip_files`), and builds a coffea-compatible fileset dict + `Dataset` objects. |
| `extractor.py` | `CoffeaMetadataExtractor` | Runs coffea's preprocessing over the fileset using a `Runner` with the configured executor (Dask or local). Produces `WorkItem` objects with file paths, entry ranges, and chunk boundaries. |
| `core.py` | `parse_dataset_key`, `aggregate_workitem_events`, `format_event_summary` | Pure functions for dataset key parsing, event count aggregation across workitems, and summary formatting. |
| `io.py` | `collect_file_paths`, `save_json`, `load_json`, `serialize_workitems` | File I/O helpers: path enumeration from dataset directories, JSON persistence for fileset/workitems/summaries. |

The flow when `run_metadata_generation=True`:

1. `DatasetMetadataManager.run(executor=...)` is called
2. `FilesetBuilder.build_fileset()` enumerates files from dataset directories → produces `fileset` dict and `Dataset` objects
3. `CoffeaMetadataExtractor.extract_metadata(fileset)` runs coffea preprocessing on workers → produces `WorkItem` list with chunked entry ranges
4. Results are saved to `metadata_dir/` as JSON (`fileset.json`, `workitems.json`, `nanoaods.json`)
5. `build_metadata_lookup()` combines dataset configs with event counts into a `MetadataLookup` dict keyed by dataset name, containing process, xsec, nevts, is_data, year, etc.

When `run_metadata_generation=False`, step 2-3 are skipped and cached JSON files are loaded instead.

## Handling bad files

There are two levels of protection against bad files:

### Skipping known bad files before processing

If you know specific files are corrupted, exclude them via `skip_files` in the dataset config. Any file whose path contains one of these strings is filtered out during dataset building (before metadata generation or processing):

```python
config["datasets"]["skip_files"] = [
    "92D0BDF3-91AE-514F-88B5-8F591450B8AD.root",
    "8E2613E5-9327-D644-9567-C3A5CE721D27.root",
]
```

### Tolerating bad files during processing

Both the metadata extractor and the processor runner use coffea's `skipbadfiles` parameter to catch and skip files that fail at read time. The errors caught include `OSError`, `LZMAError`, `DecompressionError`, `DeserializationError`, and `AssertionError`.

This is configured in:
- `src/intccms/metadata_extractor/extractor.py` (preprocessing)
- `src/intccms/analysis/runner.py` (processing)

To change which errors are tolerated, edit the `skipbadfiles` tuple in the `Runner(...)` constructor call in those files. Setting `skipbadfiles=False` disables this and makes any bad file a hard failure.

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

### How is the analysis code organized?

Three layers in `src/intccms/analysis/`:

| File | Class | Role |
|------|-------|------|
| `base.py` | `Analysis` | Generic base class. Handles corrections (correctionlib + custom functions), object masks, baseline selection, ghost observables. Experiment-agnostic. |
| `cms.py` | `CMSAnalysis(Analysis)` | CMS-specific implementation. Initializes histograms, runs the histogramming loop (nominal + systematic variations), runs statistical inference via cabinetry. |
| `processors.py` | `SkimAndAnalyseProcessor`, `TwoHundredGbpsProcessor` | Coffea processors. `SkimAndAnalyseProcessor` owns a `CMSAnalysis` instance and dispatches to it for the analysis step. |

The flow per chunk: `SkimAndAnalyseProcessor.process()` → skim selection → optionally save to disk → `CMSAnalysis.process()` → `Analysis.prepare_objects()` (corrections + masks) → `CMSAnalysis.histogramming()` (fill histograms).

### Where does the processor live?

`src/intccms/analysis/processors.py` has two processors:

- **`SkimAndAnalyseProcessor`**: The main processor. Per chunk, it: (1) applies skim selection, (2) saves filtered events to disk if enabled, (3) calls `CMSAnalysis.process()` for corrections and histogramming if enabled. All controlled by the config flags above.
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

### How are histograms initialized?

`CMSAnalysis._init_histograms()` in `src/intccms/analysis/cms.py` creates one `hist.Hist` per (channel, observable) combination. Each histogram has three axes:

- **Observable** (`hist.axis.Variable`): binning from the observable config
- **Process** (`hist.axis.StrCategory`, growth): filled dynamically with dataset names
- **Variation** (`hist.axis.StrCategory`, growth): `"nominal"` plus one entry per systematic up/down

Storage is `hist.storage.Weight()` (supports weighted events). The histograms are created once during processor initialization and filled per chunk.

### Where does histogramming happen?

`CMSAnalysis.histogramming()` in the same file. For each chunk, it:

1. Applies the channel selection
2. Computes event weights (genWeight &times; xsec normalization &times; correction SFs)
3. Evaluates the observable function
4. Fills the histogram for the matching (process, variation) bin

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

| Notebook | Purpose |
|----------|---------|
| `full_run.ipynb` | Standard analysis workflow (metadata &rarr; processing &rarr; histograms) |
| `full_run_with_metrics.ipynb` | Same workflow with roastcoffea performance dashboards |
| `full_run_workflow_modes.ipynb` | Runs all four workflow modes (skim+analysis, analysis only, skim only, analysis on skims) with per-mode metrics comparison |
| `full_run_skim_formats.ipynb` | Tests skimming output formats (Parquet/S3, TTree/XRootD, RNTuple/XRootD) with read-back verification |
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
