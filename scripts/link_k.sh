#!/bin/bash
#=============================================
# Links kepler executable to a set of models
#=============================================
if [ "$1" == "-h" ]; then
  echo "usage: `basename $0` [source] [batch0] [batch1] [n_runs]
where:
    -source   the X-ray source being modelled (e.g. gs1826)
    -batch0   first batch number being collected (e.g., 3 for gs1826_3)
    -batch1   last batch (all inbetween will be collected)
    -nruns    number of runs in each batch"
  exit 0
fi
if [ $# -ne 4 ]; then
  echo "Error: must supply 4 arguments (use option -h for help):
1. source
2. batch1
3. batch3
4. nruns"
  exit 1
fi
#=============================================
GRID_DIR=$KEPLER_MODELS
EXE_PATH=${KEPLER_PATH}/gfortran/keplery

GRID_NAME=$1
GRID_FIRST=$2
GRID_LAST=$3
NRUNS=$4
RUN_NAME='xrb'

for g in $(seq ${GRID_FIRST} ${GRID_LAST})
do
    for r in $(seq 1 ${NRUNS})
    do
      ln -sf ${EXE_PATH} ${GRID_DIR}/${GRID_NAME}_${g}/${RUN_NAME}${r}/k
    done
done
