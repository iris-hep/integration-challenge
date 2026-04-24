# CMS integration challenge

A coffea-based distributed analysis framework for the CMS Z' → tt̄ single-lepton search on Run 2 NanoAOD.

**Contents**
- [Setup](#setup)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Dask client](#dask-client)
- [Metadata generation](#metadata-generation)
- [Handling bad files](#handling-bad-files)
- [Workflow modes](#workflow-modes)
- [Skimming](#skimming)
- [The processor](#the-processor)
- [Histogramming](#histogramming)
- [Histogramming as a Service (HaaS)](#histogramming-as-a-service-haas)
- [Corrections and systematics](#corrections-and-systematics)
- [Metrics and profiling](#metrics-and-profiling)
- [Inspecting inputs](#inspecting-inputs)
- [Notebooks](#notebooks)
- [Hacks](#hacks)
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

## Dask client

### How do I connect to a Dask cluster?

Use `acquire_client` from `intccms.utils.dask_client`. It is a context manager that handles the facility-specific connection logic, registers a few useful worker plugins, and cleans up on exit:

```python
from intccms.utils.dask_client import acquire_client

AF = "coffeacasa-condor"   # or coffeacasa-gateway, purdue-af-k8s, purdue-af-slurm
AUTO_CLOSE_CLIENT = False
WORKER_DEPENDENCIES = [COFFEA_PIP, "roastcoffea==0.1.2", "histserv==0.1.3"]

with acquire_client(AF, close_after=AUTO_CLOSE_CLIENT, pip_packages=WORKER_DEPENDENCIES) as (client, cluster):
    output, report = run_processor_workflow(..., executor=DaskExecutor(client=client))
```

It yields `(client, cluster)`. `cluster` is `None` for `coffeacasa-condor`, else it is the gateway cluster object.

### Which facilities are supported?

| `af` | How it connects |
|------|-----------------|
| `coffeacasa-condor` | Direct connect to `tls://localhost:8786` |
| `coffeacasa-gateway` | Via `dask_gateway.Gateway()`, with X509 proxy and access token uploaded to workers |
| `purdue-af-k8s` | Via `dask_gateway.Gateway()` against Purdue's k8s dask-gateway |
| `purdue-af-slurm` | Via `dask_gateway.Gateway()` against Purdue's slurm dask-gateway |

### What options does `acquire_client` take?

| Argument | Default | What it controls |
|----------|---------|------------------|
| `af` | required | Facility identifier from the table above |
| `num_workers` | `None` | Scales the gateway cluster to this many workers and waits for them via `client.wait_for_workers`. Ignored for `coffeacasa-condor` |
| `close_after` | `False` | Close the client on context exit. Leave `False` when you want to reuse the client for follow-up work (pulling metrics, inspecting state) |
| `pip_packages` | `None` | List of pip specifiers to install on every worker via `PipInstall` (coffea, roastcoffea, histserv, etc.). Gateway facilities reject git URLs here |
| `propagate_aws_env` | `False` | Captures `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from the client environment and sets them on workers. Needed for S3 skim output |
| `profile_output_dir` | `None` | If set, dumps a Dask profile HTML to `<dir>/<timestamp>/dask_profile[_<suffix>].html` on context exit |
| `profile_suffix` | `None` | Suffix appended to the profile filename, e.g. `"200gbps_preload"` |

### What are `WORKER_DEPENDENCIES` for?

Client-side installed packages are not automatically present on workers. `pip_packages` triggers `dask.distributed.PipInstall`, which pip-installs the listed specs on every worker before any task runs. The notebooks pin this in sync with the client-side installs:

```python
COFFEA_PIP = COFFEA_VERSION if "git" in COFFEA_VERSION else f"coffea=={COFFEA_VERSION}"
WORKER_DEPENDENCIES = [COFFEA_PIP, "roastcoffea==0.1.2", "histserv==0.1.3"]
```

### Where is the client used?

- **Metadata generation**: `metadata_generator.run(executor=DaskExecutor(client=client))` preprocesses files on the cluster.
- **Processing**: `run_processor_workflow(..., executor=DaskExecutor(client=client))` fans out processor chunks.
- **Metrics and profiling**: `MetricsCollector(client=client, ...)` tracks per-worker time and memory; `client.profile(...)` dumps a flamegraph.
- **xcache pre-warm**: `warm_xcache(dataset_manager, client)` pre-pulls files into xcache from inside the cluster.

### Where do I add support for a new facility?

All facility wiring lives in `acquire_client()` in `src/intccms/utils/dask_client.py`. Add a new branch to the `if/elif` chain that:

1. Builds the connection (direct `Client(...)` or `dask_gateway.Gateway(...)`).
2. Optionally scales the cluster and calls `client.wait_for_workers(num_workers)`.
3. Uploads credentials or registers worker callbacks if the site needs custom env setup (see the `coffeacasa-gateway` branch for a proxy + token example).

Then add the new identifier to the `NotImplementedError` message in the final `else` branch, to the docstring's supported-facilities list, and to the table above.

Shared setup (`PrintForwarder`, AWS env, `PipInstall`, `forward_logging`, optional profile dump) runs after your branch, so as long as you assign `client` and (if relevant) `cluster`, you get those plugins and the cleanup path for free.

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

### How do I pick which branches `TwoHundredGbpsProcessor` reads?

`TwoHundredGbpsProcessor` reads whatever is in `config["preprocess"]["branches"]` (and `mc_branches`). Two helpers in `intccms.utils.tools` build those dicts:

- **`get_branches_for_fraction(file_path, target_fraction=..., data_file=..., cache_path=..., strategy=...)`**: measures per-branch sizes in a representative file, then picks branches until they cover the target fraction of the file. `strategy="largest-first"` (default) gives the fewest branches covering that fraction; `strategy="smallest-first"` gives a larger number of smaller branches at the same total data volume, for studying how branch count affects throughput. Used in `full_run_200gbps.ipynb` to sweep I/O throughput at different read fractions.
- **`prepare_branches_from_list(branch_list, mc_file=..., data_file=...)`**: takes a fixed flat list of NanoAOD branch names and groups them into the `preprocess.branches` shape. Use when the branch set is known up front, for example when replicating a reference workflow. Used in `full_run_200gbps_replicate_legacy.ipynb` to reproduce the idap-200gbps legacy benchmark branch-for-branch.

Both return the same `(branches, mc_branches)` pair. The second dict is empty unless a representative data file is passed, in which case `find_mc_only_branches` splits out the MC-only subset.

### How do I preload only the branches the processor accesses?

Pass `preload=True` to `run_processor_workflow`:

```python
output, report = run_processor_workflow(
    config=validated_config,
    output_manager=output_manager,
    metadata_lookup=metadata_lookup,
    processor=processor,
    workitems=workitems,
    executor=DaskExecutor(client=client),
    preload=True,
)
```

This passes coffea's `trace` function to the Runner. The Runner runs your processor on a typetracer (no real data) to discover which branches are accessed, then preloads only those branches eagerly per chunk instead of loading the full set of configured branches lazily.

Requires coffea 2026.4.0 or newer.

### How do I post-process the processor output?

Pass a callable via the `post` argument of `run_processor_workflow`. The required signature is:

```python
def post(processor: coffea.processor.ProcessorABC, accumulator: dict) -> dict
```

It receives the processor instance and the merged accumulator (after coffea has combined all chunks) and must return the (possibly updated) accumulator. Example:

```python
def add_totals(processor, accumulator):
    accumulator["total_weighted_events"] = sum(
        h.view()["value"].sum()
        for by_obs in accumulator.get("histograms", {}).values()
        for h in by_obs.values()
    )
    return accumulator

output, report = run_processor_workflow(
    config=validated_config,
    output_manager=output_manager,
    metadata_lookup=metadata_lookup,
    processor=processor,
    workitems=workitems,
    executor=DaskExecutor(client=client),
    post=add_totals,
)
```

Useful for pulling histograms off a remote store (e.g. histserv), deriving cross-output summaries, or any custom output manipulation that does not belong in the processor's own `postprocess`.

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

## Histogramming as a Service (HaaS)

Instead of filling histograms on workers and reducing them at the end of the job, HaaS sends every fill over gRPC to a remote [histserv](https://pypi.org/project/histserv/) process that owns the histogram state. Workers hold only a client handle, so the reduce step disappears.

### When would I use this?

Use it to experiment with replacing coffea's reduce-aggregate step with a server-side histogram, or when the reduce is the bottleneck (many chunks, many histograms, or large growth-axis categories).

### How do I run it?

Use `full_run_histserv.ipynb`. The differences from `full_run.ipynb` are:

1. `histserv` is added to the client install and to `WORKER_DEPENDENCIES` so Dask workers can talk to the server:
    ```python
    ensure("histserv", "histserv==0.1.9", "0.1.9")
    WORKER_DEPENDENCIES = [COFFEA_PIP, "roastcoffea==0.1.2", "histserv==0.1.9"]
    ```
2. The processor is `HistServProcessor` instead of `SkimAndAnalyseProcessor`:
    ```python
    from intccms.analysis import HistServProcessor
    processor = HistServProcessor(
        config=validated_config,
        output_manager=output_manager,
        metadata_lookup=metadata_lookup,
    )
    ```

`HistServProcessor` applies the skim selection just like `SkimAndAnalyseProcessor` but does not save skimmed events to disk, run statistics, or write ROOT/pickle outputs. It is a stripped-down sibling focused on the histogramming step.

### Is there a local-histogramming counterpart?

Yes: `HistLocalProcessor`. Same shape as `HistServProcessor` (applies the skim selection, runs the analysis, no file saving or statistics) but uses `CMSAnalysis` so histograms are filled on workers and reduced by coffea. Useful for debugging and as the local arm in HaaS vs local-reduce comparisons. See `full_run_haas_or_not.ipynb`, which runs both back-to-back on the same inputs and reports side-by-side metrics plus a per-category bin-by-bin correctness check.

### How does `HistServAnalysis` differ from `CMSAnalysis`?

Two overrides in `src/intccms/analysis/histserv.py`:

| Method | Change vs `CMSAnalysis` |
|--------|-------------------------|
| `_init_histograms()` | Builds the same `hist.Hist` template, then registers it with `histserv.Client.init(h)`. The per-region dict stores server-side client handles, not local histograms. |
| `histogramming()` | Does not fill directly. Appends one dict of fill kwargs (`observable`, `process`, `variation`, `weight`) per observable to `self._fill_buffer`. |

A third method, `_flush_fills()`, is called at the end of each chunk's `process()`. It collapses the buffer into a single `fill_many(...)` gRPC call per `(channel, observable)` pair &mdash; batching amortizes the round-trip cost.

### Where is the histserv server address set?

Currently hardcoded in `src/intccms/analysis/histserv.py`:

```python
histserv_client = Client(address="...:8788")
```

Edit this line to point at your own `histserv` instance.

### How do I read back the final histograms?

The accumulator only carries `processed_events` &mdash; the histogram lives on the server. Call `.to_hist()` on the client handle to pull it back as a regular `hist.Hist`:

```python
output["histograms"]["CMS_WORKSHOP"]["workshop_mtt"].to_hist()
```

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

### How do I pre-warm xcache before a run?

If you want benchmark numbers that reflect cache-hit performance (not first-touch fetches), pre-warm xcache with `warm_xcache`. It dispatches `xrdcp <file> /dev/null` to dask workers so files are pulled into the cache from inside the cluster:

```python
from intccms.utils.tools import warm_xcache

with acquire_client(AF, close_after=False) as (client, cluster):
    results, meta = warm_xcache(dataset_manager, client)

print(f"Files: {meta['n_files']}, total GB: {meta['total_GB']:.1f}")
```

Optional kwargs: `redirector` (override per-dataset), `max_files` (cap total), `processes` (subset by process name), `max_retries`.

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

## Inspecting inputs

### How do I inspect NanoAOD input files?

`input_inspector.ipynb` uses the `intccms.metrics.inspector` module to characterize input ROOT files. It runs distributed inspection via Dask and reports:
- Event counts per file and per dataset
- Branch sizes (compressed and uncompressed)
- Compression ratios
- Optional Rucio-backed file size lookups

The inspector produces rich tables (via `rich`) and matplotlib visualizations (event distributions, branch size distributions, dataset comparisons, summary dashboard).

### How do I inspect skimmed output files?

`skim_inspector.ipynb` uses the same inspector module but pointed at skimmed files on XRootD (or other remote storage). It:
1. Derives skimmed subdirectory paths from the original dataset config and a configurable `SKIM_BASE` / `SKIM_REDIRECTOR`
2. Lists files via `xrdfs ls`
3. Runs the same distributed inspection and visualization pipeline

To use it, edit the `SKIM_REDIRECTOR` and `SKIM_BASE` variables in the config cell to point at your skimmed file location.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `full_run.ipynb` | Standard analysis workflow (metadata &rarr; processing &rarr; histograms) |
| `full_run_with_metrics.ipynb` | Same workflow with roastcoffea performance dashboards |
| `full_run_workflow_modes.ipynb` | Runs all four workflow modes (skim+analysis, analysis only, skim only, analysis on skims) with per-mode metrics comparison |
| `full_run_skim_formats.ipynb` | Tests skimming output formats (Parquet/S3, TTree/XRootD, RNTuple/XRootD) with read-back verification |
| `full_run_200gbps.ipynb` | I/O throughput benchmark with profiling |
| `full_run_200gbps_preload_vs_not.ipynb` | Compares the 200gbps workflow with and without `preload=True` |
| `full_run_200gbps_replicate_legacy.ipynb` | 200gbps run with the branch list hardcoded to the idap-200gbps legacy benchmark (via `prepare_branches_from_list`) for direct comparison |
| `full_run_histserv.ipynb` | Standard workflow with histograms filled via HaaS (histserv) instead of reduce-aggregate |
| `full_run_haas_or_not.ipynb` | Runs `HistServProcessor` and `HistLocalProcessor` back-to-back and compares metrics plus bin-by-bin histogram outputs |
| `input_inspector.ipynb` | Inspect NanoAOD input files (event counts, branch sizes, compression) |
| `skim_inspector.ipynb` | Inspect skimmed output files on XRootD or other remote storage |

## Hacks

Things that should work but might break, which we are working on. Temporary workarounds until each is fixed upstream.

### Roastcoffea metrics can crash the 200gbps run at large read fractions

At large `TARGET_FRACTION` values in `full_run_200gbps.ipynb`, the run can crash while `MetricsCollector` pulls Dask span data back to the client. Fix pending in roastcoffea.

To run without metrics, comment out:

1. In the processor-run cell, the `with MetricsCollector(...) as collector:` block and every `collector.*` line (inside and after). De-indent the `run_processor_workflow(...)` call so it runs directly under `acquire_client`, and keep the `t0` / `t1` timers.
2. The cells below "Performance Metrics" that reference `metrics`, `tracking_data`, or `span_metrics` ("Timeline Plots" and "Task and Processor Breakdown").

The Dask profile via `profile_output_dir` on `acquire_client` is separate and still works.

## Credentials

Storage credentials (AWS keys for S3) go in an untracked `.env` file at the repo root (`cms/.env`). See `.env.example` for the expected variable names. Copy it and fill in your values:

```sh
cp .env.example .env
# edit .env with your credentials
```

Load early with:

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
