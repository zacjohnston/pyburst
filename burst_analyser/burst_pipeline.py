"""
Wrapper for sequential burst analysis routines, such as:
    - copying params table files
    - loading/saving lightcurve files
    - analysing models
    - collecting the results
"""
import numpy as np
import os
import sys

# kepler_grids
from . import burst_analyser
from . import burst_tools
from ..grids import grid_tools
from ..misc.pyprint import printv

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


def run_analysis(batches, source, copy_params=True, reload=True, multithread=True,
                 analyse=True, collect=True, verbose=True):
    """Run sequential analysis steps for burst models
    """
    # 1.
    if copy_params:
        printv('Copying parameter tables', verbose)
        grid_tools.copy_paramfiles(batches, source)

    # 2.
    if reload:
        printv('Loading lightcurve files', verbose)
        burst_tools.multi_batch_save(batches, source, multithread=multithread)

    # 3.
    if analyse:
        printv('Extracting burst properties from models', verbose)
        if multithread:
            burst_analyser.multithread_extract(batches, source)
        else:
            burst_analyser.extract_bursts(batches, source)

    # 4.
    if collect:
        printv('Collecting results', verbose)
        last_batch = batches[-1]
        burst_tools.combine_extracts(np.arange(1, last_batch+1), source)


def check_n_bursts(batches, source, kgrid):
    """Compares n_bursts detected with kepler_analyser against burstfit_1808
    """
    mismatch = np.zeros(4)
    filename = f'mismatch_{source}_{batches[0]}-{batches[-1]}.txt'
    filepath = os.path.join(GRIDS_PATH, filename)

    for batch in batches:
        summ = kgrid.get_summ(batch)
        n_runs = len(summ)

        for i in range(n_runs):
            run = i + 1
            n_bursts1 = summ.iloc[i]['num']
            sys.stdout.write(f'\r{source}_{batch} xrb{run:02}')

            burstfit = burst_analyser.BurstRun(run, batch, source, verbose=False)
            burstfit.analyse()
            n_bursts2 = burstfit.n_bursts

            if n_bursts1 != n_bursts2:
                m_new = np.array((batch, run, n_bursts1, n_bursts2))
                mismatch = np.vstack((mismatch, m_new))

        np.savetxt(filepath, mismatch)
    return mismatch
