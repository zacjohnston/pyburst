#!/bin/bash
#=============================================
# Pulls kepler models from remote clusters
#=============================================
if [[ "$1" == "-h" ]]; then
  echo "usage: `basename $0` [source] [batch0] [batch1]
where:
    -cluster  cluster to push to [monarch, icer, oz]
    -source   the name of the grid (eg grid5)
    -batch0 (optional) first batch number being collected (e.g., 3 for grig5_3)
    -batch1 (optional) last batch (all inbetween will be collected)

Note: if only cluster and source are provided, will pull all available batches
    "
  exit 0
fi
if [[ $# -ne 4 ]]; then
    if [[ $# -ne 2 ]]; then
    echo "Error: must supply 2 or 4 arguments (use option -h for help):
        1. cluster
        2. source
       (3. batch0)
       (4. batch1)"
    exit 1
    fi
fi

#=============================================
cluster=${1}
source=${2}
TARGET_DIR=${KEPLER_MODELS}/${source}

if [[ "${cluster}" == "monarch" ]]; then
  SERVER='m2'
  KPATH='/home/zacpetej/id43/kepler/runs'
elif [[ "${cluster}" == "icer" ]]; then
  SERVER='icer'
  KPATH='/mnt/home/f0003004/kepler/runs'
elif [[ "${cluster}" == "oz" ]]; then
  SERVER='oz'
  KPATH='/fred/oz011/zac/kepler/runs'
else
  echo "Must choose one of (icer, monarch, oz)"
  exit 1
fi

MODELS_DIR=${SERVER}:${KPATH}/${source}

if [[ $# == 4 ]]; then
    batch0=${3}
    batch1=${4}

    for batch in $(seq ${batch0} ${batch1}); do
      PULL_DIR="${PULL_DIR} ${MODELS_DIR}/${source}_${batch}"
    done
else
    PULL_DIR="${MODELS_DIR}/*"
fi

rsync -av ${PULL_DIR} ${TARGET_DIR}