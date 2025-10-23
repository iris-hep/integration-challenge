# ntuple production information

Tutorial: https://topcptoolkit.docs.cern.ch/latest/tutorials/setup/

Get a ttbar sample to test with:
- `centralpage --scope=mc23_13p6TeV --physlite Top TTbar Baseline PowhegPythia` (`-l` to help navigate)
- `mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.deriv.DAOD_PHYSLITE.e8514_s4162_r15540_p6697`, download smallest file `mc23_13p6TeV:DAOD_PHYSLITE.43700597._000010.pool.root.1` (10k events)

Basic setup:

```bash
git clone ssh://git@gitlab.cern.ch:7999/atlasphys-top/reco/TopCPToolkit.git
cd TopCPToolkit
git checkout v2.22.0
setupATLAS
asetup AnalysisBase,25.2.66
cmake -S source -B build
cmake --build build --parallel 4
source build/*/setup.sh
mkdir -p run
cd run/
```

Restore later on:

```bash
setupATLAS
asetup --restore
source build/*/setup.sh
cd run/
```

To run:

```bash
runTop_el.py -h
runTop_el.py -i inp.txt -o output -t integration-challenge -e 1000
```

- Central page resources:
    - https://twiki.cern.ch/twiki/bin/view/AtlasProtected/CentralMC20ProductionListNew
    - https://twiki.cern.ch/twiki/bin/view/AtlasProtected/CentralMC23ProductionListNew

- p-tags: https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/DerivationProductionTeam#p_tags
- r-tags: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/AtlasProductionGroup#Specific_Information_on_MC_campa

- TopCPToolkit
    - grid submission: https://topcptoolkit.docs.cern.ch/latest/tutorials/submit_grid/
        - add "--no-systematics" for data
    - example config https://gitlab.cern.ch/atlasphys-top/reco/TopCPToolkit/-/blob/main/source/TopCPToolkit/share/configs/exampleTtbarLjets/reco.yaml

- recommendations: https://atlas-topq.docs.cern.ch/Reco/Recommendations/

data input files:
```
data15_13TeV:data15_13TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp15_v01_p6697
data16_13TeV:data16_13TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp16_v01_p6697
data17_13TeV:data17_13TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp17_v01_p6697
data18_13TeV:data18_13TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp18_v01_p6697
data22_13p6TeV:data22_13p6TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp22_v02_p6700
data23_13p6TeV:data23_13p6TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp23_v01_p6700
data24_13p6TeV:data24_13p6TeV.periodAllYear.physics_Main.PhysCont.DAOD_PHYSLITE.grp24_v01_p6700
```
-> total 551k files, 308 TB, 39 B events
