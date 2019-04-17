#!/bin/bash

if [ "$1" == "-h" ]; then
  echo "usage: `basename $0` [basename] [first grid] [last grid]
where:
1. grid/batch basename (e.g. 'grid', 'gs1826')
2. first grid
3. last grid"
  exit 0
fi

if [ $# -ne 3 ]; then
  echo "Error: must supply 3 arguments (use option -h for help):
1. grid/batch basename (e.g. 'grid', 'gs1826')
2. first grid
3. last grid"
  exit 1
fi

GRID_DIR=$KEPLER_MODELS
GRID_NAME=$1
GRID_FIRST=$2
GRID_LAST=$3
RUN_NAME='xrb'

echo "Deleting ascii dumps:"
for g in $(seq ${GRID_FIRST} ${GRID_LAST})
do
    gpath="${GRID_DIR}/${GRID_NAME}/${GRID_NAME}_${g}"
    nruns=$(wc -l < ${gpath}/MODELS.txt)
    for r in $(seq ${nruns})
    do
        echo -ne "${GRID_NAME}_${g}  ${RUN_NAME}${r}/${nruns}"'\r'
        rm -f ${gpath}/${RUN_NAME}${r}/${RUN_NAME}${r}_*
    done
    echo
done
