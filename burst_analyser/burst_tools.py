import numpy as np
import pandas as pd
import subprocess
import os
import multiprocessing as mp
import time

# kepler
import lcdata

# pygrids
from ..misc import pyprint
from ..grids.grid_tools import try_mkdir
from ..grids import grid_strings, grid_tools

MODELS_PATH = os.environ['KEPLER_MODELS']
GRIDS_PATH = os.environ['KEPLER_GRIDS']


def load(run, batch, source, basename='xrb', reload=False, save=True,
         silent=False):
    """Attempts to load pre-saved lumfile, or load binary. Returns luminosity [t, lum]
    """
    batch_str = grid_strings.get_batch_string(batch, source)

    analysis_path = grid_strings.get_source_subdir(source, 'burst_analysis')
    input_path = os.path.join(analysis_path, batch_str, 'input')
    try_mkdir(input_path, skip=True)

    presaved_file = f'{batch_str}_{run}.txt'
    run_str = grid_strings.get_run_string(run, basename)
    presaved_filepath = os.path.join(input_path, presaved_file)

    # ===== Force reload =====
    if reload:
        print('Force-reloading binary file: ')
        try:
            print('Deleting old presaved file')
            subprocess.run(['rm', presaved_filepath])
        except:
            pass

    # ===== Try loading pre-saved data =====
    try:
        print(f'Looking for pre-saved luminosity file: {presaved_filepath}')
        lum = np.loadtxt(presaved_filepath, skiprows=1)
        print('Pre-saved data found, loaded.')

    except FileNotFoundError:
        print('No presaved file found. Reloading binary')
        pyprint.print_dashes()
        model_path = grid_strings.get_model_path(run, batch, source, basename)
        lc_filename = f'{run_str}.lc'
        lc_filepath = os.path.join(model_path, lc_filename)

        if os.path.exists(lc_filepath):
            lum_temp = lcdata.load(lc_filepath, silent=silent)
            n = len(lum_temp.time)

            lum = np.full((n, 2), np.nan)
            lum[:, 0] = lum_temp.time
            lum[:, 1] = lum_temp.xlum

            pyprint.print_dashes()
            if save:
                print(f'Saving data for faster loading in: {presaved_filepath}')
                header = 'time (s),             luminosity (erg/s)'
                np.savetxt(presaved_filepath, lum, header=header)
        else:
            print(f'File not found: {lc_filepath}')
            lum = np.array([np.nan])

    pyprint.print_dashes()
    return lum


def batch_save(batch, source, runs=None, basename='xrb', reload=True, **kwargs):
    """Loads a collection of models and saves their lightcurves
    """
    if runs is None:
        runs = grid_tools.get_nruns(batch, source)
    runs = grid_tools.expand_runs(runs)

    for run in runs:
        load(run, batch, source, basename=basename, reload=reload, **kwargs)


def multi_batch_save(batches, source, multithread=True, **kwargs):
    """Loads multiple batches of models and saves lightcurves
    """
    batches = grid_tools.expand_batches(batches, source)
    t0 = time.time()
    if multithread:
        args = []
        for batch in batches:
            args.append((batch, source))

        with mp.Pool(processes=8) as pool:
            pool.starmap(batch_save, args)
    else:
        for batch in batches:
            batch_save(batch, source, **kwargs)

    t1 = time.time()
    dt = t1 - t0
    print(f'Time taken: {dt:.1f} s ({dt/60:.2f} min)')


def combine_extracts(batches, source):
    """Combines extracted burst property summary tables
    """
    source_path = grid_strings.get_source_path(source)
    big_table = pd.DataFrame()

    for batch in batches:
        batch_str = f'{source}_{batch}'
        analysis_path = os.path.join(source_path, 'burst_analysis', batch_str)

        filename = f'burst_analysis_{batch_str}.txt'
        filepath = os.path.join(analysis_path, filename)
        batch_table = pd.read_csv(filepath, delim_whitespace=True)

        big_table = pd.concat((big_table, batch_table), ignore_index=True)

    table_str = big_table.to_string(index=False, justify='left', col_space=12)

    filename = f'burst_analysis_{source}.txt'
    filepath = os.path.join(source_path, 'burst_analysis', filename)

    with open(filepath, 'w') as f:
        f.write(table_str)

    return big_table


def copy_sample_plots(batches, source):
    """Collect the last plot of each batch into a folder, for quick examination
    """
    source_path = grid_strings.get_source_path(source)
    target_path = os.path.join(source_path, 'plots', 'quickview')

    for batch in batches:
        n_runs = grid_tools.get_nruns(batch, source)
        model_str = grid_strings.get_model_string(n_runs, batch, source)

        filename = f'model_{model_str}.png'
        filepath = os.path.join(source_path, 'plots', 'burst_analysis', filename)
        target_filepath = os.path.join(target_path, filename)
        subprocess.run(['cp', filepath, target_filepath])