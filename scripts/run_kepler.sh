#!/bin/bash
#=============================================
# Iterates and runs a set of kepler models sequentially
#   Useful for testing model setups
#=============================================
if [ $# -ne 2 ]; then
  echo "Must supply 2 arguments:
1. source
2. batch
"
  exit
fi
#=============================================
SOURCE=${1}
BATCH=${2}
KEP_EXE=${KEPLER_PATH}/gfortran/keplery

GPATH=${KEPLER_MODELS}/${SOURCE}_${BATCH}
NRUNS=$(wc -l < ${GPATH}/MODELS.txt)

for run in $(seq 1 ${NRUNS})
do
    cd ${GPATH}/xrb${run}/
    ${KEP_EXE} xrb${run} xrb_g
done
