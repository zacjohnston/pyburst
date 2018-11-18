#!/bin/bash
#=============================================
# Removes kepler executable from a set of models
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
SOURCE=${1}
B0=${2}
B1=${3}
NRUNS=${4}

for g in $(seq ${B0} ${B1})
do
    for r in $(seq 1 ${NRUNS})
    do
        rm $KEPLER_MODELS/${SOURCE}_${g}/xrb${r}/k
    done
done
