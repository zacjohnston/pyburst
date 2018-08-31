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
from pygrids.grids import grid_tools, grid_strings, grid_analyser
from pygrids.misc.pyprint import print_title

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


def run_analysis(batches, source, copy_params=True, reload=True, multithread=True,
                 analyse=True, save_plots=True, collect=True, load_bursts=False,
                 auto_last_batch=True):
    """Run all analysis steps for burst models
    """
    # TODO: copy generators
    all_batches = np.arange(batches[-1]) + 1  # assumes batches[-1] is final batch of grid
    if copy_params:
        print_title('Copying parameter tables')
        grid_tools.copy_paramfiles(batches, source)
        grid_tools.combine_grid_tables(all_batches, 'params', source=source)

    if analyse:
        print_title('Extracting burst properties from models')
        extract_batches(batches, source, save_plots=save_plots, load_bursts=load_bursts,
                        multithread=multithread, reload=reload)

    if collect:
        print_title('Collecting results')
        if auto_last_batch:
            kgrid = grid_analyser.Kgrid(source, exclude_defaults=True,
                                        powerfits=False, burst_analyser=True)
            last_batch = int(kgrid.params.iloc[-1]['batch'])
        else:
            last_batch = batches[-1]

        burst_tools.combine_batch_summaries(np.arange(last_batch) + 1, source)


def extract_batches(batches, source, save_plots=True, multithread=True,
                    reload=False, load_bursts=False):
    """Do burst analysis on arbitrary number of batches"""
    t0 = time.time()
    batches = grid_tools.ensure_np_list(batches)

    for batch in batches:
        print_title(f'Batch {batch}')

        analysis_path = grid_strings.batch_analysis_path(batch, source)
        output_path = os.path.join(analysis_path, 'output')
        grid_tools.try_mkdir(output_path, skip=True)

        n_runs = grid_tools.get_nruns(batch, source)
        runs = np.arange(n_runs) + 1

        if multithread:
            args = []
            for run in runs:
                args.append((run, batch, source, save_plots, reload, load_bursts))
            with mp.Pool(processes=8) as pool:
                pool.starmap(extract_runs, args)
        else:
            extract_runs(runs, batch, source, reload=reload, save_plots=save_plots,
                         load_bursts=load_bursts)

        burst_tools.combine_run_summaries(batch, source)

    t1 = time.time()
    dt = t1 - t0
    print_title(f'Time taken: {dt:.1f} s ({dt/60:.2f} min)')


def extract_runs(runs, batch, source, save_plots=True, reload=False, load_bursts=False):
    """Do burst analysis on run(s) from a single batch and save results
    """
    runs = grid_tools.ensure_np_list(runs)
    for run in runs:
        print_title(f'Run {run}')
        model = burst_analyser.BurstRun(run, batch, source, analyse=True,
                                        reload=reload, load_bursts=load_bursts)
        model.save_burst_table()
        model.save_summary_table()

        if save_plots:
            model.plot(display=False, save=True)
            model.plot_convergence(display=False, save=True)
            model.plot_linregress(display=False, save=True)

