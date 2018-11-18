#!/bin/bash
#---------------------------------------------
#       help
#---------------------------------------------
if [ $# -ne 2 ]; then
  echo "Error: must supply 2 arguments:
1. source
2. cluster"
  exit 1
fi
#---------------------------------------------
# relies on ssh alias m=zacpetej@monarch.erc.monash.edu
SOURCE=${1}
CLUSTER=${2}
TARGET_MCMC=${KEPLER_GRIDS}/sources/${SOURCE}/mcmc
TARGET_LOGS=${KEPLER_GRIDS}/sources/${SOURCE}/logs

if [ "${CLUSTER}" == "monarch" ]; then
  PULL_DIR_MCMC="m:/home/zacpetej/id43/kepler_grids/sources/${SOURCE}/mcmc"
  PULL_DIR_LOGS="m:/home/zacpetej/id43/kepler_grids/sources/${SOURCE}/logs"
elif [ "${CLUSTER}" == "icer" ]; then
  PULL_DIR_MCMC="isync:/mnt/home/f0003004/kepler_grids/sources/${SOURCE}/mcmc"
  PULL_DIR_LOGS="isync:/mnt/home/f0003004/kepler_grids/sources/${SOURCE}/logs"
fi

rsync -av --info=progress2 ${PULL_DIR_MCMC}/ ${TARGET_MCMC}/
rsync -av --info=progress2 ${PULL_DIR_LOGS}/*.o* ${TARGET_LOGS}/
