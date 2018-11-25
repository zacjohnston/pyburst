#!/bin/bash
#============================================
# Pipeline setps of grid analysis:
#       1. collect generator files
#       2. setup kepler-analyser
#       3. run kepler-analyser
#       4. collect results from kepler-analyser
#       5. update kepler_grids git repo
#============================================
if [ $# -ne 2 ]; then
  if [ $# -ne 3 ]; then
  echo "Must supply 2 or 3 arguments:
          1. source
          2. batch1
          3. batch2 (optional, to iterate multiple batches)"
  exit 1
  fi
fi
#============================================
GRID_PATH=${KEPLER_GRIDS}
MODELS_PATH=${KEPLER_MODELS}
GEN_SCRIPT=${PYBURST_PATH}'/scripts/collgen.sh'
MANAGER_SCRIPT=${PYBURST_PATH}'/scripts/manage_analyser.py'
ANALYSE_SCRIPT=${PYBURST_PATH}'/scripts/run_analyser.py'
GIT_SCRIPT=${PYBURST_PATH}'/scripts/update_git.sh'

SRC=${1}
BATCH1=${2}

if [ $# == 2 ]; then
  BATCH2=${BATCH1}
fi
if [ $# == 3 ]; then
  BATCH2=${3}
fi

# get number of batches based on source
n=0
if [ "$SRC" == "gs1826" ]; then
  n=2
else
  if [ "$SRC" == "4u1820" ]; then
    n=1
  fi
fi
step=$((n+1))

for BATCH_i in $(seq ${BATCH1} ${step} ${BATCH2}); do
  BATCH_i_plus=$((${BATCH_i}+${n}))
  NRUNS=$(ls -d ${MODELS_PATH}/${SRC}_${BATCH_i}/xrb* | wc -l)

  # 1. Collect generator files
#  bash ${GEN_SCRIPT} ${SRC} ${BATCH_i} ${BATCH_i_plus} ${NRUNS}

  # 2. Setup analyser
#   python ${MANAGER_SCRIPT} 'setup' ${SRC} ${BATCH_i} ${BATCH_i_plus} ${NRUNS}

  # 3. Run analyser
  source activate python2.7
  python ${ANALYSE_SCRIPT} ${SRC} ${BATCH_i} ${BATCH_i_plus}
  source deactivate
done

# 4. Collect results
python ${MANAGER_SCRIPT} 'collect' ${SRC} ${BATCH1} ${BATCH_i_plus} ${NRUNS}

# 4. Combine tables
# python ${MANAGER_SCRIPT} 'combine' ${SRC} ${BATCH1} ${BATCH_i_plus} ${NRUNS}

# 5. Update git repo
# bash ${GIT_SCRIPT}
