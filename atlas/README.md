# IRIS-HEP ATLAS integration challenge

## Step 1: TopCPToolkit

Tutorial: https://topcptoolkit.docs.cern.ch/latest/tutorials/setup/

Get a ttbar sample to test with:
- PMG central page: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/CentralMC23ProductionListNew
- e.g. `mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.recon.AOD.e8514_s4162_r15540`
- `lsetup centralpage`
- `centralpage --scope=mc23_13p6TeV --physlite Top TTbar Baseline PowhegPythia` (`-l` to help navigate)
- `mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697`, download smallest file `mc23_13p6TeV:DAOD_PHYSLITE.43700597._000010.pool.root.1` (10k events)


## Step 2: coffea for ntuple processing

Using https://github.com/scipp-atlas/atlas-schema, example notebook for usage is `process_ntuples.ipynb`.

related issues / PRs / MRs:
- `atlasopenmagic` warnings in notebooks: https://github.com/atlas-outreach-data-tools/atlasopenmagic/issues/31
- identifying weight systematics more easily: https://gitlab.cern.ch/atlas/athena/-/merge_requests/81794
