#!/bin/bash
#=============================================
# Deletes redundant concord chain files by only keeping most recent
#=============================================
#  Help
#---------------------------------------------
if [ "$1" == "-h" ]; then
  echo "usage: `basename $0` [source] [batch_label] [n_runs] [step1] [step2] [step_size]
where:
    -source         source object being modelled (e.g. gs1826)
    -batch_label    label list of batches, (e.g. 4-3-2)
    -n_runs         number of runs/models in each batch
    -con_ver        ID to distinguish different implementations of concord
    -step1          first step  (also taken to be the step-size)
    -step2          last step - this one will be kept"
  exit 0
fi

if [ $# -ne 6 ]; then
  echo "Error: must supply 6 arguments (use option -h for help):
1. source
2. batch_label
3. n_runs
4. con_version
5. step1 (also taken as step size)
6. step2"
  exit 1
fi

#=============================================
DIR=${KEPLER_GRIDS}

source=${1}
batch_label=${2}
n_runs=${3}
con_ver=${4}
step1=${5}
step2=${6}
step_size=${step1}

if [ $con_ver == 0 ]; then
  con_str=''
else
  con_str='_C0'${con_ver}
fi

last_delete=$((${step2}-${step_size}))
DIR=${DIR}'/sources/'${source}'/concord'

echo '=============================='
echo 'Keeping step: ' ${step2}
echo '=============================='
echo 'Deleting:'

# ==== iterate over runs ====
for r in $(seq 1 ${n_runs}); do
  RUN_NAME=${source}'_'${batch_label}'_R'${r}

  # ==== iterate over steps ====
  for i in $(seq ${step1} ${step_size} ${last_delete}); do
    STEP_NAME=${RUN_NAME}'_S'${i}${con_str}
    rm -v ${DIR}'/chain_'${STEP_NAME}'.npy'
    # rm -v ${DIR}'/lnprob_'${STEP_NAME}'.npy'
    # rm -v ${DIR}'/rstate_'${STEP_NAME}'.pkl'
  done
done
