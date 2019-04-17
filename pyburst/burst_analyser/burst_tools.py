import numpy as np
import pandas as pd
import subprocess
import os
import sys
import multiprocessing as mp
import time

# kepler
try:
    import lcdata
except ModuleNotFoundError:
    print('Kepler python module "lcdata" not found. Some functionality disabled.')

# pyburst
from pyburst.misc import pyprint
from pyburst.grids import grid_strings, grid_tools


# TODO: Move to kepler_tools.py?
def load_lum(run, batch, source, basename='xrb', reload=False, save=True,
             silent=True, check_monotonic=True):
    """Attempts to load pre-extracted luminosity data, or load raw binary.
    Returns [time (s), luminosity (erg/s)]
    """
    def load_save(load_filepath, save_filepath):
        lum_loaded = extract_lcdata(filepath=load_filepath, silent=silent)
        if save:
            try:
                save_ascii(lum=lum_loaded, filepath=save_filepath)
            except FileNotFoundError:
                print("Can't save preloaded luminosity file, path not found")
        return lum_loaded

    batch_str = grid_strings.get_batch_string(batch, source)
    analysis_path = grid_strings.get_source_subdir(source, 'burst_analysis')
    input_path = os.path.join(analysis_path, batch_str, 'input')

    presaved_filepath = os.path.join(input_path, f'{batch_str}_{run}.txt')
    run_str = grid_strings.get_run_string(run, basename)
    model_path = grid_strings.get_model_path(run, batch, source, basename)
    binary_filepath = os.path.join(model_path, f'{run_str}.lc')
    print(binary_filepath)
    if reload:
        print('Deleting preloaded file, reloading binary file')
        subprocess.run(['rm', '-f', presaved_filepath])
        try:
            lum = load_save(binary_filepath, presaved_filepath)
        except FileNotFoundError:
            print('XXXXXXX lumfile not found. Skipping XXXXXXXX')
            return
    else:
        try:
            lum = load_ascii(presaved_filepath)
        except (FileNotFoundError, OSError):
            print('No preloaded file found. Reloading binary')
            try:
                lum = load_save(binary_filepath, presaved_filepath)
            except FileNotFoundError:
                print('XXXXXXX lumfile not found. Skipping XXXXXXX')
                return

    if check_monotonic:
        dt = np.diff(lum[:, 0])
        if True in (dt < 0):
            pyprint.print_warning('Lightcurve timesteps are not in order. '
                                  + 'Something has gone horribly wrong!', n=80)
            raise RuntimeError('Lightcurve timesteps are not in order')
    return lum


def load_ascii(filepath):
    """Loads pre-extracted .txt file of [time, lum]
    """
    print(f'Loading preloaded luminosity file: {filepath}')
    return np.loadtxt(filepath, skiprows=1)


def save_ascii(lum, filepath):
    """Saves extracted [time, lum]
    """
    print(f'Saving data for faster loading in: {filepath}')
    header = 'time (s),             luminosity (erg/s)'
    np.savetxt(filepath, lum, header=header)


def extract_lcdata(filepath, silent=True):
    """Extracts luminosity versus time from kepler binary file (.lc)
    """
    lumfile = lcdata.load(filepath, silent=silent, graphical=False)

    if lumfile is None:
        raise FileNotFoundError("lumfile doesn't exist")
    else:
        n = len(lumfile.time)
        lum = np.full((n, 2), np.nan)
        lum[:, 0] = lumfile.time
        lum[:, 1] = lumfile.xlum
        return lum


def batch_save(batch, source, runs=None, basename='xrb', reload=True, **kwargs):
    """Loads a collection of models and saves their lightcurves
    """
    if runs is None:
        runs = grid_tools.get_nruns(batch, source)
    runs = grid_tools.expand_runs(runs)

    for run in runs:
        load_lum(run, batch, source, basename=basename, reload=reload, **kwargs)


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


def multi_save(table, source, basename='xrb'):
    """Extract models from table of arbitrary batches/runs
    """
    batches = np.unique(table['batch'])
    t0 = time.time()

    for batch in batches:
        subset = grid_tools.reduce_table(table, params={'batch': batch})
        runs = np.array(subset['run'])
        args = []

        for run in runs:
            args.append((run, batch, source, basename, True))
        with mp.Pool(processes=8) as pool:
            pool.starmap(load_lum, args)

    t1 = time.time()
    dt = t1 - t0
    print(f'Time taken: {dt:.1f} s ({dt/60:.2f} min)')


def combine_batch_tables(batches, source, table_name):
    """Combines summary files of given batches into single table
    """
    print('Combining batch summary tables:')
    source_path = grid_strings.get_source_path(source)
    table_list = []

    for batch in batches:
        sys.stdout.write(f'\r{source} {batch}/{batches[-1]}')
        batch_table = load_batch_table(batch, source=source, table_name=table_name)
        table_list += [batch_table]
    sys.stdout.write('\n')

    combined_table = pd.concat(table_list, ignore_index=True)
    table_str = combined_table.to_string(index=False, justify='left')

    filename = f'{table_name}_{source}.txt'
    filepath = os.path.join(source_path, 'burst_analysis', filename)
    print(f'Saving: {filepath}')
    with open(filepath, 'w') as f:
        f.write(table_str)

    return combined_table

# TODO: Combine these two functions (diff: load_run_table, load_batch_table)

def combine_all_run_tables(batches, source, table_name):
    """Combine the runs from multiple batches into batch tables
    """
    print('Combining the run tables of multiple batches')
    for batch in batches:
        combine_run_tables(batch, source=source, table_name=table_name)


def combine_run_tables(batch, source, table_name):
    """Combine summary files of runs into a batch table
    """
    # TODO: rename all instances of burst_analysis to summary (in directory tree)
    print(f'Combining model summary tables:')
    n_runs = grid_tools.get_nruns(batch, source)
    runs = np.arange(n_runs) + 1
    table_list = []

    for run in runs:
        sys.stdout.write(f'\r{source}{batch} {run}/{runs[-1]}')
        run_table = load_run_table(run, batch, source=source, table=table_name)
        table_list += [run_table]
    sys.stdout.write('\n')

    combined_table = pd.concat(table_list, ignore_index=True)
    table_str = combined_table.to_string(index=False, justify='left')

    if table_name == 'summary':  # hack fix for now TODO: !!!
        table_name = 'burst_analysis'

    filepath = grid_strings.batch_table_filepath(batch, source, table_name=table_name)
    print(f'Saving: {filepath}')
    with open(filepath, 'w') as f:
        f.write(table_str)

    return combined_table


def load_run_table(run, batch, source, table):
    """Loads file of given run (either summary or burst table)
    """
    if table not in ['summary', 'bursts']:
        raise ValueError("table must be on of ['summary', 'bursts']")

    analysis_path = grid_strings.batch_analysis_path(batch, source)
    filename = grid_strings.get_batch_filename(table, batch, source,
                                               run=run, extension='.txt')
    filepath = os.path.join(analysis_path, 'output', filename)
    return pd.read_csv(filepath, delim_whitespace=True)


def load_batch_table(batch, source, table_name):
    """Loads summary table of batch from file and returns as pd table
    """
    filepath = grid_strings.batch_table_filepath(batch, source, table_name=table_name)
    return pd.read_csv(filepath, delim_whitespace=True)


def get_burst_cycles(run, batch, source):
    """Returns dump cycles that correspond to burst start times
    """
    burst_table = load_run_table(run, batch, source=source, table='bursts')
    mask = ~np.isnan(burst_table['dump_start'])
    return np.array(burst_table['dump_start'][mask].astype(int))


def get_quartiles(x, iqr_frac=1.5):
    """Returns quartile values for given array

    parameters
    ----------
    x : array
        array to calculate quartiles from
    iqr_frac : flt
        distance from Q1/Q3 to define outliers, as fraction of interquartile range (IQR)
    returns
    -------
    lower outlier limit, q1, q2, q3, upper outlier limit
    """
    q1 = np.percentile(x, 25)
    q2 = np.percentile(x, 50)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - iqr_frac*iqr
    upper = q3 + iqr_frac*iqr

    return lower, q1, q2, q3, upper


def get_outlier_idxs(x, percentiles):
    """Returns list of outlier indexes
    """
    low_idxs = np.where(x < percentiles[0])[0]
    high_idxs = np.where(x > percentiles[4])[0]
    return np.concatenate((low_idxs, high_idxs))


def snip_outliers(x, percentiles):
    """Returns array x, with outliers removed
    """
    idxs = get_outlier_idxs(x, percentiles)
    return np.delete(x, idxs)
