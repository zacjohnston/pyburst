#!/bin/bash
#=============================================
# Pulls kepler models from remote monarch (deletes ascii dumps first)
# NOTE: !!!DOESN'T WORK YET!!!
#=============================================
if [ "$1" == "-h" ]; then
  echo "usage: `basename $0` [source] [batch_label] [n_runs] [step0] [step1] [step_add]
where:
    -source         source object being modelled (e.g. gs1826)
    -batch_label    label list of batches, (e.g. 4-3-2)
    -n_runs         number of runs/models in each batch
    -con_ver        concord version id
    -step1          first step
    -step2          last step
    -step_size       step-size between files. Will be added when renaming"
  exit 0
fi
if [ $# -ne 7 ]; then
  echo "Error: must supply 7 arguments (use option -h for details):
1. source
2. batch_label
3. n_runs
4. con_ver
5. step1
6. step2
7. step_size"
  exit 1
fi

#=============================================
source=${1}
batch_label=${2}
nrunss=${3}
con_ver=${4}
step1=${5}
step2=${6}
step_size=${7}

command='rmchains ${source} ${batch_label} ${nruns} ${con_ver} ${step1} ${step2} ${step_size}'
echo $command
# ssh m ${command}
# syncm
