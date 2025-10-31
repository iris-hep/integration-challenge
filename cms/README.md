# IRIS-HEP CMS integration challenge

## Environment (pixi)
This project uses [pixi](https://pixi.sh/) for environment management. To set up the environment, run:

```sh
pixi install
source pixi_activate.sh # for conda-like environment
```

## Code structure
The main codebase is organized as follows:

```
cms/
  analysis.py                # Main analysis script
  pixi.toml, pixi.lock       # Environment configuration (pixi)
  analysis/                  # Analysis logic and base classes
  example_opendata/          # Open-data example configs, cuts, datasets
  example_cms/               # CMS internal-style example configs
  utils/                     # Utility modules (output manager, skimming, schema, etc.)
```
- Start from the example configurations in `example_opendata/configs/` or `example_cms/configs/`—they provide complete analysis dictionaries (datasets, skimming, channels) that you can copy and adapt for your own campaign.
- Main scripts and entry points are in `cms/`.

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
            "format": "parquet",          # other options: root_ttree, rntuple, safetensors (stubs)
            "local": True,
            "base_uri": "s3://bucket",    # optional override for remote storage
            "to_kwargs": {"compression": "zstd"},   # forwarded to ak.to_parquet
            "from_kwargs": {"storage_options": {...}}  # forwarded to NanoEventsFactory.from_parquet
        },
    },
}
```

The file suffix is fixed to `{dataset}/file_{index}/part_{chunk}.{ext}`, so switching between local and remote storage only requires changing the `local` flag and optional `base_uri`.

- Set `general.run_skimming=True` to regenerate skims. Use `datasets.max_files` to limit input size when experimenting.
- Downstream steps load the same path, so no separate cache copy is needed; cached Awkward objects are still produced automatically for faster reruns.
- Dataset-level options such as lumi masks live next to each dataset definition (for example `lumi_mask`: `{ "function": cuts.lumi_mask, "use": [...], "static_kwargs": {"lumifile": "...json"} }`).

## Running code
To run the main analysis chain, execute the relevant Python scripts or notebooks. Outputs such as histograms and fit results will be saved in the `outputs/` directory. For example:

```sh
python analysis.py
```

Check the configuration files for additional options and details on running systematic variations or fits.

The following is guaranteed to produce a result if skimming is already performed

```sh
python3 analysis.py general.run_skimming=False general.read_from_cache=True general.run_mva_training=False general.run_plots_only=False general.run_metadata_generation=False
```
