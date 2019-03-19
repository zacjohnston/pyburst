#!/usr/bin/env bash
#---------------------------------------------
#       usage
#---------------------------------------------
if [ $# -ne 2 ]; then
  echo "Must supply 2 arguments:
1. source
2. cluster"
  exit 0
fi
#---------------------------------------------
# relies on ssh alias m=zacpetej@monarch.erc.monash.edu
SOURCE=${1}
CLUSTER=${2}
FILE_SOURCE=${KEPLER_GRIDS}/sources/${SOURCE}/interpolator/


if [ "${CLUSTER}" == "monarch" ]; then
    FILE_TARGET=m:/home/zacpetej/id43/kepler_grids/sources/${SOURCE}/interpolator/
elif [ "${CLUSTER}" == "icer" ]; then
    FILE_TARGET=isync:/mnt/home/f0003004/kepler_grids/sources/${SOURCE}/interpolator/
elif [ "${CLUSTER}" == "oz" ]; then
    FILE_TARGET=oz:/fred/oz011/zac/kepler_grids/sources/${SOURCE}/interpolator/
fi

rsync -av --info=progress2 ${FILE_SOURCE}/ ${FILE_TARGET}
