# IRIS-HEP CMS integration challenge

## Environment (pixi)
This project uses [pixi](https://pixi.sh/) for environment management. To set up the environment, run:

```sh
pixi install
pixi run pixi activate
```

## Code structure
The main codebase is organized as follows:

```
cms/
  analysis.py                # Main analysis script
  pixi.toml, pixi.lock       # Environment configuration (pixi)
  analysis/                  # Analysis logic and base classes
  corrections/               # Correction files (JSON, text, etc.)
  example/                   # Example datasets and outputs
  user/                      # User analysis configuration, cuts, observables
  utils/                     # Utility modules (output, plotting, stats, etc.)
```
- Configuration files for the analysis are found in `cms/user/` (e.g., `configuration.py`).
- Main scripts and entry points are in `cms/`.

## Metadata and preprocessing
Metadata extraction and preprocessing are handled before the main analysis. Metadata includes information about datasets, event counts, and cross sections, and is used to configure the analysis and normalization. Preprocessing steps may include filtering, object selection, and preparing input files for skimming and analysis.

## Skimming
To skim NanoAOD datasets, use the provided scripts and configuration files in the `analysis/` and `user/` directories. Adjust the configuration as needed for your analysis channels and observables.

Currently, the code writes out skimmed files as intermediate outputs. The plan is to integrate the workflow so that all steps, including skimming, are performed on-the-fly without writing intermediate files, streamlining the analysis process.

If you need pre-skimmed data, it is available on CERNBox upon request. Please contact Mohamed Aly (mohamed.aly@cern.ch) for access.
If you want to reproduce the skimmed files yourself, set the option `general.run_skimming=True` in the configuration file `cms/user/configuration.py`.

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