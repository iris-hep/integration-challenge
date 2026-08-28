A simple fit of the dijet mass `m_jj` in the H+ -> cb signal region, run either with
[TRExFitter](https://trexfitter-docs.web.cern.ch/trexfitter-docs/latest/) or with
[cabinetry](https://cabinetry.readthedocs.io/) so the two can be compared on identical
inputs.

## Input

`wsi/trexfitter.root`, produced from the analysis histogram by `hist_to_wsi.py`:
one TDirectory per sample (`SR_ttbar_nom`, `SR_wjets`, ...), each holding `nominal` plus one
histogram per systematic variation (`<syst>_Up` / `<syst>_Down`).

Both frameworks read those very same histograms. TRExFitter builds the histogram name as
`Region.HistoName + Sample.HistoNameSuff + Systematic.HistoNameSufUp`; cabinetry fills the
`{RegionPath}`, `{SamplePath}` and `{VariationPath}` placeholders of `General.InputPath`
the same way.

## Environment setup

```
source setup.sh
```

## Running the fit

Everything is steered by `fit.py`, which writes the config and optionally runs it.
`--framework` picks the tool; the steps are the same either way (make a config, then run
the fit), only the config syntax and the fit backend change:

```
python fit.py --framework trexfitter --run-fit
python fit.py --framework cabinetry  --run-fit
```

The systematics are discovered from the input file rather than hard-coded, so the configs
always match whatever `hist_to_wsi.py` wrote. A variation is treated as one-sided when it
has no down histogram, or when its up and down histograms are identical; otherwise both
sides are used. That becomes `Symmetrisation: ONESIDED` in TRExFitter and
`Down: {Symmetrize: true}` in cabinetry.

To see all available options, run `python fit.py --help`.

### Layout

```
frameworks/samples.py       samples of the fit, shared by both frameworks
frameworks/inputs.py        systematics discovery from the input file, shared
frameworks/trexfitter/      the same fit in TRExFitter config syntax
frameworks/cabinetry/       the same fit in cabinetry config syntax
```

### TRExFitter

Writes `configs/<job name>.config` and runs `trex-fitter <--trex-opts> <config>`, with
output in `<job name>/`. To run the fit directly:

```
trex-fitter hwdfpr configs/Hplus_cb_mjj.config
```

See [running options](https://trexfitter-docs.web.cern.ch/trexfitter-docs/latest/running/howto/)
and the [TRExFitter walkthrough](https://trexfitter-docs.web.cern.ch/trexfitter-docs/latest/tutorials/short_walkthrough/#running-a-fit).

### cabinetry

Writes `configs/<job name>.yml`, with output in `<job name>_cabinetry/`
(`--cabinetry-outdir` overrides). `--cabinetry-opts` selects the steps using the same
letters as `--trex-opts`:

| letter | step                                                           |
|--------|----------------------------------------------------------------|
| `h`    | build template histograms (`templates.collect` + `postprocess`) |
| `w`    | build the `pyhf` workspace                                      |
| `d`    | pre-fit data/MC plot and yield table                            |
| `f`    | fit, pull plot, correlation matrix, likelihood scan of the POI  |
| `p`    | post-fit data/MC plot and yield table                           |
| `r`    | nuisance parameter ranking                                      |

The steps always run in that order, and the workspace can also be handed to cabinetry's
own CLI or to `pyhf` directly:

```
cabinetry workspace configs/Hplus_cb_mjj.yml Hplus_cb_mjj_cabinetry/workspace.json
cabinetry fit --pulls Hplus_cb_mjj_cabinetry/workspace.json
```

### Differences between the two configurations

* Stat-only (`--stats-only`) drops all systematics in both cases; for cabinetry it also
  sets `DisableStaterror` on every sample, to match TRExFitter dropping the MC stat
  gamma parameters under `StatOnly: TRUE`.
* Blinding is TRExFitter's `FitBlind` and, for cabinetry, fitting
  `model_utils.asimov_data` built at `--mu-asimov`.
* Systematics are `Type: HISTO` in TRExFitter and `Type: NormPlusShape` in cabinetry
  (a `normsys` plus a `histosys` modifier in the workspace).
* `--center-of-mass-energy`, `--luminosity`, `--plot-systematics` and `--num-cpu` only
  affect the TRExFitter config; TRExFitter's systematic pruning has no cabinetry
  equivalent.
