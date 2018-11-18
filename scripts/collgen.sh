#!/bin/bash
#=============================================
# Collects kepler generator files from a grid of runs into a single location
#=============================================
if [ "$1" == "-h" ]; then
  echo "usage: `basename $0` [source] [batch1] [batch2] [n_runs]
where:
    -source   the X-ray source being modelled (e.g. gs1826)
    -batch1   first batch number being collected (e.g., 3 for gs1826_3)
    -batch2   last batch (all inbetween will be collected)
    -n_runs   number of runs/models in each batch"
  exit 0
fi
if [ $# -ne 4 ]; then
  echo "Error: must supply 4 arguments (use option -h for help):
1. source
2. batch1
3. batch2
4. n_runs"
  exit 1
fi
#=============================================

MODEL_DIR=${KEPLER_MODELS}
GRID_DIR=${KEPLER_GRIDS}
BASENAME='xrb'
GEN_NAME='xrb_g'
BURN_NAME='rpabg'

SRC=${1}
BATCH1=${2}
BATCH2=${3}
N_RUNS=${4}

for b in $(seq ${BATCH1} ${BATCH2})
do
  BATCH_NAME=${SRC}'_'${b}
  BATCH_DIR=${MODEL_DIR}'/'${BATCH_NAME}
  SAVE_DIR=${GRID_DIR}'/sources/'${SRC}'/generators/'${BATCH_NAME}

  echo '=============================='
  echo 'copying from: '${BATCH_DIR}
  echo '          to: '${SAVE_DIR}
  echo '=============================='

  mkdir ${SAVE_DIR}

  for i in $(seq 1 ${N_RUNS})
  do
    RUN_DIR=${BATCH_DIR}'/'${BASENAME}${i}
    RUN_NAME=${BATCH_NAME}'_R'${i}

    G_FILE=${RUN_DIR}'/'${GEN_NAME}
    B_FILE=${RUN_DIR}'/'${BURN_NAME}

    G_SAVE_NAME='gen_'${RUN_NAME}
    B_SAVE_NAME=${BURN_NAME}'_'${RUN_NAME}

    G_SAVE_PATH=${SAVE_DIR}'/'${G_SAVE_NAME}
    B_SAVE_PATH=${SAVE_DIR}'/'${B_SAVE_NAME}

    echo ${BASENAME}${i}'/{'${GEN_NAME}','${BURN_NAME}'}  --->  '${RUN_NAME}

    cp ${G_FILE} ${G_SAVE_PATH}
    cp ${B_FILE} ${B_SAVE_PATH}

  done
done
