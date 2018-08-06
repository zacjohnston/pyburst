"""
Wrapper for sequential burst analysis routines, such as:
    - copying params table files
    - loading/saving lightcurve files
    - analysing models
    - collecting the results
"""
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import sys
import time

# kepler_grids
from . import burst_analyser
from . import burst_tools
from ..grids import grid_tools, grid_strings
from ..misc.pyprint import print_title

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


def run_analysis(batches, source, copy_params=True, reload=True, multithread=True,
                 analyse=True, collect=True):
    """Run sequential analysis steps for burst models
    """
    # TODO: multithread by runs (for large batches)

    # 1.
    if copy_params:
        print_title('Copying parameter tables')
        grid_tools.copy_paramfiles(batches, source)
        # TODO combine paramfiles (grid_tools)

    # 2.
    if reload:
        print_title('Loading lightcurve files')
        burst_tools.multi_batch_save(batches, source, multithread=multithread)

    # 3.
    if analyse:
        print_title('Extracting burst properties from models')
        if multithread:
            multithread_extract(batches, source)
        else:
            extract_bursts(batches, source)

    # 4.
    if collect:
        print_title('Collecting results')
        last_batch = batches[-1]
        burst_tools.combine_extracts(np.arange(1, last_batch + 1), source)


# def multithread_extract(batches, source, plot_model=True, plot_convergence=True,
#                         plot_linregress=True):
#     for batch in batches:
#         args = []
#         n_runs = grid_tools.get_nruns(batch, source)
#         runs = np.arange(1, n_runs+1)
#
#         for run in runs:
#             # args.append((run, batch, source, basename, True))
#             args.append([batch, source, plot_model, plot_convergence, plot_linregress])
#         # with mp.Pool(processes=8) as pool:
#         #     pool.starmap(load, args)
#
#     t0 = time.time()
#     with mp.Pool(processes=8) as pool:
#         pool.starmap(extract_batches, args)
#     t1 = time.time()
#     dt = t1 - t0
#     print(f'Time taken: {dt:.1f} s ({dt/60:.2f} min)')


def extract_batches(batches, source, save_plots=True, multithread=True):
    """Do burst analysis on arbitrary number of batches"""
    analysis_path = grid_strings.get_source_subdir(source, 'burst_analysis')
    batches = grid_tools.ensure_np_list(batches)

    flags = ('converged',)
    b_ints = ('batch', 'run', 'num', 'discard')
    bprops = ('dt', 'fluence', 'length', 'peak', 'rate')
    col_order = ['batch', 'run', 'num', 'converged', 'discard', 'dt', 'u_dt', 'rate', 'u_rate',
                 'fluence', 'u_fluence', 'length', 'u_length', 'peak', 'u_peak']

    for batch in batches:
        print_title(f'Batch {batch}')
        batch_str = f'{source}_{batch}'
        batch_path = os.path.join(analysis_path, batch_str)
        grid_tools.try_mkdir(analysis_path, skip=True)

        filename = f'burst_analysis_{batch_str}.txt'
        filepath = os.path.join(analysis_path, filename)

        n_runs = grid_tools.get_nruns(batch, source)
        for run in range(1, n_runs + 1):
            extract_runs(run, batch, source, save_plots=save_plots)

        # table = table[col_order]
        # table_str = table.to_string(index=False, justify='left',
        #                             formatters={'discard': '{:.0f}'.format})
        # with open(filepath, 'w') as f:
        #     f.write(table_str)


def extract_runs(runs, batch, source, save_plots=True):
    """Do burst analysis on run(s) from a single batch and save results
    """
    runs = grid_tools.ensure_np_list(runs)
    for run in runs:
        print_title(f'Run {run}')
        model = burst_analyser.BurstRun(run, batch, source, analyse=True)
        model.save_burst_table()
        model.save_summary_table()

        if save_plots:
            model.plot(display=False, save=True)
            model.plot_convergence(display=False, save=True)
            model.plot_linregress(display=False, save=True)

