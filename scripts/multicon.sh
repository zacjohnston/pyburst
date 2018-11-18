#!/bin/bash
#=============================================
#Submits multiple concord job scripts
#=============================================
if [ $# -ne 4 ]; then
  echo "Error: must supply 4 arguments:
  1. source
  2. batch0
  3. batch1
  4. con_ver"
  exit 1
fi
#=============================================
SOURCE=${1}
B1=${2}
B2=${3}
CON_VER=${4}

if [ "$SOURCE" == 'gs1826' ]; then
  n=3
fi
if [ "$SOURCE" == '4u1820' ]; then
  n=2
fi

LOG_DIR=${KEPLER_GRIDS}/sources/${SOURCE}/logs

for b in $(seq ${B1} ${n} ${B2}); do
  cd ${LOG_DIR}
	qsub icer_con${CON_VER}_${SOURCE}_${b}-*
done
