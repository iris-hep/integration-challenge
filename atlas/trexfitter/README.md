A simple TRExFitter fit of the dijet mass `m_jj` in the H+ -> cb signal region.

## Input

`wsi/trexfitter.root`, produced from the analysis histogram by `../hist_to_trexfitter.py`:
one TDirectory per sample (`SR_ttbar_nom`, `SR_wjets`, ...), each holding `nominal` plus one
histogram per systematic variation (`<syst>_Up` / `<syst>_Down`).

## Environment setup

```
source setup.sh
```

## Running the fit

Everything is steered by `fit.py`, which writes the config and optionally runs it:

```
python fit.py --run-fit --compare-stat-only
```

The systematics are discovered from the input file rather than hard-coded, so the config
always matches whatever `hist_to_trexfitter.py` wrote. A variation is written with
`Symmetrisation: ONESIDED` when it has no down histogram, or when its up and down
histograms are identical; otherwise both sides are used.

To see all available options, run `python fit.py --help`. To run the fit directly:

```
trex-fitter hwdfpr configs/Hplus_cb_mjj.config
```

For further info, see [running options](https://trexfitter-docs.web.cern.ch/trexfitter-docs/latest/running/howto/) and the [TRExFitter walkthrough](https://trexfitter-docs.web.cern.ch/trexfitter-docs/latest/tutorials/short_walkthrough/#running-a-fit) page.
