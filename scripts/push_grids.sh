#!/bin/bash
#=============================================
# Pushes kepler models to remote monarch (assumes set of 3 batches)
#=============================================
if [ "$1" == "-h" ]; then
  echo "usage: `basename $0` [source] [batch0] [batch1]
where:
    -source   the X-ray source being modelled (e.g. gs1826)
    -batch0   first batch number being collected (e.g., 3 for gs1826_3)
    -batch1   last batch (all inbetween will be collected)
    -cluster  cluster to push to [monarch, icer]"
  exit 0
fi
if [ $# -ne 4 ]; then
  echo "Error: must supply 4 arguments (use option -h for help):
1. source
2. batch0
3. batch1
4. cluster"
  exit 1
fi
#=============================================
DIR=$KEPLER_MODELS
source=${1}
batch0=${2}
batch1=${3}
cluster=${4}

if [ "${cluster}" == "monarch" ]; then
  TARGET='m2:/home/zacpetej/id43/kepler/runs'
elif [ "${cluster}" == "icer" ]; then
  TARGET='icer:/mnt/home/f0003004/kepler/runs'
else
  echo "Must choose one of (icer, monarch)"
  exit 1
fi

for batch in $(seq ${batch0} ${batch1}); do
  batches="${batches} ${source}_${batch}"
done

cd ${DIR}
rsync -av ${batches} ${TARGET}
