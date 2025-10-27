# Ntuple production

- Basic setup:

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

- Restore later on:

    ```bash
    setupATLAS
    asetup --restore
    source build/*/setup.sh
    cd run/
    ```

- The relative path of `reco.yaml` from within the `run/` directory should be `../source/TopCPToolkit/share/configs/integration-challenge/reco.yaml`. Run locally for testing:

    ```bash
    runTop_el.py -h
    runTop_el.py -i inp.txt -o output -t integration-challenge -e 1000
    ```

- Structure for grid submission: `grid/` subfolder in `run/`, needs `input_containers.py` and `submit_to_grid.py`. Data needs no systematics via `--no-systematics`.

- Extra environment setup for submission:

    ```bash
    lsetup "rucio -w"
    lsetup panda pyami
    voms-proxy-init -voms atlas
    ```

- After ntuple production: `write_ntuple_metadata.py` saves relevant metadata of input and output containers plus xrootd ntuple file paths to disk.

## Notes for first and potential second production

- workaround for electron efficiency in Run-3 fastsim for v1.1 production: `forceFullSimConfig: True`
- MCMC SFs for 545027/8 in v1.2: `generator: 'default'`, should be fixed for next production via https://gitlab.cern.ch/atlas/athena/-/merge_requests/82736
- electron efficiency: use (and test first) `correlationModelId` / `correlationModelIso` / `correlationModelReco` set to `TOTAL` for less NPs

## More information

- Central page resources:

    - https://twiki.cern.ch/twiki/bin/view/AtlasProtected/CentralMC20ProductionListNew
    - https://twiki.cern.ch/twiki/bin/view/AtlasProtected/CentralMC23ProductionListNew

- r-tags: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/AtlasProductionGroup#Specific_Information_on_MC_campa
- p-tags: https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/DerivationProductionTeam#p_tags

- TopCPToolkit

    - tutorials: https://topcptoolkit.docs.cern.ch/latest/tutorials/
    - [example config](https://gitlab.cern.ch/atlasphys-top/reco/TopCPToolkit/-/blob/main/source/TopCPToolkit/share/configs/exampleTtbarLjets/reco.yaml)
    - top reco recommendations: https://atlas-topq.docs.cern.ch/Reco/Recommendations/
