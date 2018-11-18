#!/bin/bash
#=============================================
# update git repo with new files from grids
#=============================================

DIR=${KEPLER_GRIDS}

echo '=============================='
echo 'Updating kepler_grids git repo'
echo '=============================='
cd $DIR
git add sources
