"""
Wrapper for sequential burst analysis routines, such as:
    - copying params table files
    - loading/saving lightcurve files
    - analysing models
    - collecting the results
"""
import numpy as np
import multiprocessing as mp
import os
import time

# kepler_grids
from . import burst_analyser
from . import burst_tools
from pyburst.grids import grid_tools, grid_strings
from pyburst.misc.pyprint import print_title

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


def run_analysis(batches, source, copy_params=False, reload=True, multithread=True,
                 analyse=True, save_plots=True, collect=True, load_bursts=False,
                 load_summary=False, auto_last_batch=True, basename='xrb',
                 new_models=False):
    """Run all analysis steps for burst models
    """
    if new_models:
        print('Adding new models. '
              'Overriding options: reload, copy_params, auto_last_batch')
        reload = False
        copy_params = True
        auto_last_batch = False

    all_batches = np.arange(batches[-1]) + 1  # assumes batches[-1] is final batch of grid
    if copy_params:
        print_title('Copying parameter tables')
        grid_tools.copy_paramfiles(batches, source)
        grid_tools.combine_grid_tables(all_batches, 'params', source=source)

    if analyse:
        print_title('Extracting burst properties from models')
        extract_batches(batches=batches, source=source, save_plots=save_plots,
                        load_bursts=load_bursts, multithread=multithread, reload=reload,
                        basename=basename, load_summary=load_summary)

    if collect:
        print_title('Collecting results')
        if auto_last_batch:
            grid_table = grid_tools.load_grid_table('params', source=source,
                                                    burst_analyser=True)
            last_batch = grid_table.batch.iloc[-1]
        else:
            last_batch = batches[-1]  # Assumes last batch is the last for whole grid

        burst_tools.combine_batch_summaries(np.arange(last_batch) + 1, source=source,
                                            table_name='burst_analysis')


def extract_batches(source, batches=None, save_plots=True, multithread=True,
                    reload=False, load_bursts=False, load_summary=False, basename='xrb',
                    param_table=None):
    """Do burst analysis on arbitrary number of batches"""
    t0 = time.time()
    if param_table is not None:
        print('Using models from table provided')
        batches = np.unique(param_table['batch'])
    else:
        batches = grid_tools.ensure_np_list(batches)

    for batch in batches:
        print_title(f'Batch {batch}')

        analysis_path = grid_strings.batch_analysis_path(batch, source)
        for folder in ['input', 'output']:
            path = os.path.join(analysis_path, folder)
            grid_tools.try_mkdir(path, skip=True)

        if param_table is not None:
            subset = grid_tools.reduce_table(param_table, params={'batch': batch})
            runs = np.array(subset['run'])
        else:
            n_runs = grid_tools.get_nruns(batch, source)
            runs = np.arange(n_runs) + 1

        if multithread:
            args = []
            for run in runs:
                args.append((run, batch, source, save_plots, reload, load_bursts,
                             load_summary, basename))
            with mp.Pool(processes=8) as pool:
                pool.starmap(extract_runs, args)
        else:
            extract_runs(runs, batch, source, reload=reload, save_plots=save_plots,
                         load_bursts=load_bursts, load_summary=load_summary,
                         basename=basename)

        burst_tools.combine_run_summaries(batch, source, table_name='summary')

    t1 = time.time()
    dt = t1 - t0
    print_title(f'Time taken: {dt:.1f} s ({dt/60:.2f} min)')


def extract_runs(runs, batch, source, save_plots=True, reload=False, load_bursts=False,
                 load_summary=False, basename='xrb'):
    """Do burst analysis on run(s) from a single batch and save results
    """
    runs = grid_tools.ensure_np_list(runs)
    for run in runs:
        print_title(f'Run {run}')
        model = burst_analyser.BurstRun(run, batch, source, analyse=True,
                                        reload=reload, load_bursts=load_bursts,
                                        basename=basename, load_summary=load_summary)
        model.save_burst_table()
        model.save_summary_table()

        if save_plots:
            model.plot(display=False, save=True, log=False)
            model.plot_convergence(display=False, save=True)
            model.plot_lightcurves(display=False, save=True)
