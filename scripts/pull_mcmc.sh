#!/bin/bash


if [ $# -ne 2 ]; then
  echo "Error: must supply 2 arguments:
1. source
2. cluster"
  exit 1
fi

SOURCE=${1}
CLUSTER=${2}
TARGET_MCMC=${KEPLER_GRIDS}/sources/${SOURCE}/mcmc

if [ "${CLUSTER}" == "monarch" ]; then
  PULL_DIR_MCMC="m:/home/zacpetej/id43/kepler_grids/sources/${SOURCE}/mcmc"
elif [ "${CLUSTER}" == "icer" ]; then
  PULL_DIR_MCMC="isync:/mnt/home/f0003004/kepler_grids/sources/${SOURCE}/mcmc"
elif [ "${CLUSTER}" == "c" ]; then
  PULL_DIR_MCMC="c:/home/zac/projects/kepler_grids/sources/${SOURCE}/mcmc"
fi

rsync -av --info=progress2 ${PULL_DIR_MCMC}/ ${TARGET_MCMC}/
