#!/bin/bash
#=============================================
# Pulls kepler models from remote monarch (deletes ascii dumps first)
# NOTE: !!!DOESN'T WORK YET!!!
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
source=${1}
batch1=${2}
batch2=${3}
nrunss=${4}

# command="'rmascii ${source} ${batch1} ${batch2} 1 ${nruns}'"
command="'ls'"

ssh m ${command}
exit
syncm
