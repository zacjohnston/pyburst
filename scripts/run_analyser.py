import numpy as np
import sys
import os
import kepler_analyser


# =================================================
# Executable script to run kepler-analyser on a grid of models
# -------------------------------------------------
# Requires a set of models ready for analysis:
#           - use mdot/analyser_tools.setup_analyser()
#
# Usage: python run_analyser.py [source] [batch_first] [batch_last]
#           e.g. python run_analyser.py gs1826 5 6
# =================================================
# NOTE: Requires python 2.7 (bash: 'source activate python2.7')
# =================================================

def run(**kwargs):
    grid = kepler_analyser.ModelGrid(**kwargs)
    grid.analyse_all()
    filepath = os.path.join(output_dir, 'summ.csv')
    out = np.loadtxt(filepath, skiprows=1, usecols=[1, 18], delimiter=',')
    for x in out:
        dt = x[1] / 3600
        n = x[0]
        print('tDel (hr): {dt:.2f},  N_bursts: {n:.0f}'.format(dt=dt, n=n))


source = sys.argv[1]
batch_first = int(sys.argv[2])
batch_last = int(sys.argv[3])

parameter_filename = 'MODELS.txt'
base_name = 'xrb'
GRIDS_PATH = os.environ['KEPLER_GRIDS']
analyser_path = os.path.join(GRIDS_PATH, 'analyser')

for batch in range(batch_first, batch_last + 1):
    batch_str = '{source}_{batch}'.format(source=source, batch=batch)
    batch_path = os.path.join(analyser_path, source)

    input_str = batch_str + '_input'
    output_str = batch_str + '_output'
    input_dir = os.path.join(batch_path, input_str)
    output_dir = os.path.join(batch_path, output_str)

    run(base_dir=input_dir, base_name=base_name, parameter_filename=parameter_filename, output_dir=output_dir)
