#!/bin/bash
#=============================================
# Pushes kepler models to remote monarch
#=============================================
if [[ "$1" == "-h" ]]; then
  echo "usage: `basename $0` [source] [batch0] [batch1]
where:
    -source   the X-ray source being modelled (e.g. gs1826)
    -batch0   first batch number being collected (e.g., 3 for gs1826_3)
    -batch1   last batch (all inbetween will be collected)
    -cluster  cluster to push to [monarch, icer]"
  exit 0
fi
if [[ $# -ne 4 ]]; then
  echo "Error: must supply 4 arguments (use option -h for help):
1. source
2. batch0
3. batch1
4. cluster"
  exit 1
fi
#=============================================
DIR=${KEPLER_MODELS}
source=${1}
batch0=${2}
batch1=${3}
cluster=${4}

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
TARGET=${SERVER}:${KPATH}/${source}

for batch in $(seq ${batch0} ${batch1}); do
  batches="${batches} ${source}_${batch}"
done

cd ${DIR}/${source}

ssh ${SERVER} mkdir -p ${TARGET}
rsync -av ${batches} ${TARGET}
