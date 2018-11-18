#!/bin/bash
#---------------------------------------------
#       help
#---------------------------------------------
if [ "$1" == "-h" ]; then
  echo "usage: `basename $0` [source] [cluster]
  where:
    -source   the X-ray source being modelled (e.g. gs1826)
    -cluster  the cluster to pull from (monarch, icer)"
  exit 0
fi

if [ $# -ne 2 ]; then
  echo "Error: must supply 2 arguments (use option -h for help):
1. source
2. cluster"
  exit 1
fi
#---------------------------------------------
# relies on ssh alias m=zacpetej@monarch.erc.monash.edu
SOURCE=${1}
CLUSTER=${2}
TARGET=${KEPLER_GRIDS}/sources/${SOURCE}/concord

if [ "${CLUSTER}" == "monarch" ]; then
  PULL_DIR="m:/home/zacpetej/id43/kepler_grids/sources/${SOURCE}/concord"
elif [ "${CLUSTER}" == "icer" ]; then
  PULL_DIR="icer:/mnt/home/f0003004/kepler_grids/sources/${SOURCE}/concord"
fi

rsync -av --info=progress2 ${PULL_DIR}/ ${TARGET}/
