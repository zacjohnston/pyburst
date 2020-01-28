import numpy as np
import sys

# kepler grids
from pyburst.lampe_analyser import lampe_analyser_tools
from pyburst.grids import grid_tools, grid_strings

# =================================================================
# Script callable from terminal to setup/collect kepler-analyser models
# (used in /kepler_grids/scripts/grid_pipeline.sh)
# =================================================================
if len(sys.argv) != 5 + 1:
    print('Parameters:'
          + '\n1. option (setup, collect, combine)'
          + '\n2. source'
          + '\n3. first batch'
          + '\n4. last batch'
          + '\n5. n_runs')
    sys.exit(0)

option = sys.argv[1]
source = sys.argv[2]
batch_first = int(sys.argv[3])
batch_last = int(sys.argv[4])
nruns = int(sys.argv[5])

con_vers = [6]  # con_vers to write submision scripts for
basename = 'xrb'

if (source == 'gs1826') or (source == '4u1820'):
    batch0 = 2
else:
    batch0 = 1

batches = np.arange(batch_first, batch_last + 1)
source = grid_strings.source_shorthand(source=source)

if option == 'setup':
    lampe_analyser_tools.multi_setup_analyser(batches=batches, source=source,
                                              basename=basename)
elif option == 'collect':
    lampe_analyser_tools.collect_output(runs=nruns, batches=batches, source=source,
                                        basename=basename)

    grid_tools.copy_paramfiles(batches=batches, source=source)

    # for con_ver in con_vers:
    #     for cluster in ['monarch', 'icer']:
    #         cjob_submission.write_submission_script(cluster=cluster,
    #                                         batches=batches, source=source,
    #                                         con_ver=con_ver, threads=8)
elif option == 'combine':
    for table in ['params', 'summ']:
        grid_tools.combine_grid_tables(batches=np.arange(batch0, batch_last + 1),
                                       source=source, table_basename=table)
else:
    print('option parameter must be one of [setup, collect]')
