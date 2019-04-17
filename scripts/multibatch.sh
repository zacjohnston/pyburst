#!/bin/bash
#========================================================
# Submits multiple batch jobs (via batch scripts)
# input : first grid ID, last grid ID, restart(y/n)
#========================================================
if [ $# -ne 6 ]; then
  if [ $# -ne 4 ]; then
    echo "Error: must supply 4 or 6 arguments:
    1. source
    2. batch1
    3. batch2
    (4. run0)
    (5. run1)
    6. restart"
    exit 1
  fi
fi
#============================================
# Environment variables:
LOC_DIR=${KEPLER_PATH}/runs
CLUSTER=${CLUSTER}

source=$1
batch1=$2
batch2=$3

if [ $# == 6 ]; then
  run0=$4
  run1=$5
  restart=$6
  filestring=${run0}-${run1}.qsub
fi

if [ $# == 4 ]; then
   restart=$4
   filestring=*
fi

if [ "${restart}" == 'y' ]
then
  echo 'RESTARTING MODELS'
  prefix="restart_${source}"
else
  echo 'STARTING MODELS'
  prefix="${source}"
fi

for i in $(seq ${batch1} ${batch2})
do
  cd ${LOC_DIR}/${source}/${source}_${i}/logs
  sbatch ${CLUSTER}_${prefix}_${i}_${filestring}
done

exit 0
