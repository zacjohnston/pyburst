#!/bin/bash
#=============================================
# Renames concord chain files by changing step label
#=============================================
if [ "$1" == "-h" ]; then
  echo "usage: `basename $0` [source] [batch_label] [n_runs] [step0] [step1] [step_add]
where:
    -source         source object being modelled (e.g. gs1826)
    -batch_label    label list of batches, (e.g. 4-3-2)
    -n_runs         number of runs/models in each batch
    -step1          first step
    -step2          last step
    -step_size       step-size between files. Will be added when renaming"
  exit 0
fi
if [ $# -ne 6 ]; then
  echo "Error: must supply 6 arguments (use option -h for details):
1. source
2. batch_label
3. n_runs
4. step1
5. step2
6. step_size"
  exit 1
fi
#=============================================
DIR=${KEPLER_GRIDS}

source=${1}
batch_label=${2}
n_runs=${3}
step0=${4}
step1=${5}
step_add=${6}

CHAIN_DIR=${DIR}'/sources/'${source}'/concord'

for r in $(seq 1 ${n_runs})
do
  RUN_NAME=${source}'_'${batch_label}'_R'${r}

  for i in $(seq ${step0} ${step_add} ${step1})
  do
    new_step=$((${i}+${step_add}))
    STEP_NAME=${RUN_NAME}'_S'${i}
    NEW_NAME=${RUN_NAME}'_S'${new_step}

    mv ${CHAIN_DIR}'/chain_'${STEP_NAME}'.npy' ${CHAIN_DIR}'/chain_'${NEW_NAME}'.npy'
    mv ${CHAIN_DIR}'/lnprob_'${STEP_NAME}'.npy' ${CHAIN_DIR}'/lnprob_'${NEW_NAME}'.npy'
    mv ${CHAIN_DIR}'/rstate_'${STEP_NAME}'.pkl' ${CHAIN_DIR}'/rstate_'${NEW_NAME}'.pkl'

  done
done
