#!/usr/bin/env bash

#----------------------------------------------------
# Moves large MCMC data files into archive for backup
#----------------------------------------------------

if [ $# -ne 2 ]; then
  if [ $# -ne 3 ]; then
  echo "Must supply 2 or 3 arguments:
          1. source
          2. version1
          3. version2 (optional, to iterate over multiple batches)"
  exit 0
  fi
fi


srce=${1}
version1=${2}
MCMC_PATH=${KEPLER_GRIDS}/sources/${srce}/mcmc/
ARCHIVE_PATH=/c/zac/backups/mcmc/${srce}/


if [ $# == 2 ]; then
  version2=${version1}
fi
if [ $# == 3 ]; then
  version2=${3}
fi

for version_i in $(seq ${version1} ${version2}); do
#    echo "*${srce}_V${version_i}_* ==> ${ARCHIVE_PATH}"
    mv -v ${MCMC_PATH}*${srce}_V${version_i}_* ${ARCHIVE_PATH}
done
