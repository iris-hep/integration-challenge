#!/bin/bash
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
export ALRB_localConfigDir=$HOME/localConfig
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh

# TopCPToolkit setup
asetup AnalysisBase,25.2.58
source /home/alheld/integration-challenge/TopCPToolkit/build/*/setup.sh

DSID=$1
INPUTFILE=$2
rm -f input.txt
echo ${INPUTFILE} >> input.txt
cat input.txt
OUTNAME=$(grep -P "(?<=\.)(\d+._\d+)(?=\.)" input.txt -o)
echo "running in: $(pwd)" 
echo "input: ${DSID} ${INPUTFILE}"
echo "output: ${OUTNAME}.root"

runTop_el.py -i input.txt -o ${OUTNAME} -t integration-challenge # -e 1000
mkdir -p /data/alheld/integration-challenge/${DSID}
cp ${OUTNAME}.root /data/alheld/integration-challenge/${DSID}/.
