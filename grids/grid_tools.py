import numpy as np
import pandas as pd
import sys, os
import itertools
import subprocess
from astropy.io import ascii
from functools import reduce

# kepler_grids
from pygrids.misc.pyprint import print_dashes

# kepler
import kepdump

flt2 = '{:.2f}'.format
flt4 = '{:.4f}'.format
FORMATTERS = {'z': flt4, 'y': flt4, 'x': flt4, 'accrate': flt4,
              'tshift': flt2, 'qb': flt4, 'xi': flt2, 'qb_delay': flt2,
              'mass': flt2}

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


# TODO: rewrite docstrings


def load_grid_table(tablename, source, con_ver=None,
                    verbose=True, **kwargs):
    """=================================================
    Returns file of model parameters as pandas DataFrame
    =================================================
    tablename  = str   : table name (e.g. 'params', 'summ')
    source     = str   : name of source object
    ================================================="""
    source = source_shorthand(source=source)
    path = kwargs.get('path', GRIDS_PATH)
    param_path = os.path.join(path, 'sources', source, tablename)

    if tablename == 'concord_summ':
        if con_ver == None:
            raise ValueError('must provide con_ver (concord version)')
        else:
            cv = f'_C{con_ver:02}'  # concord_summ needs concord version
    else:
        cv = ''

    filename = f'{tablename}_{source}{cv}.txt'
    filepath = os.path.join(param_path, filename)

    printv(f'Loading {tablename} table: {filepath}', verbose)
    params = pd.read_table(filepath, delim_whitespace=True)
    return params


def expand_runs(runs):
    """========================================================
    Checks format of 'runs' parameter and returns relevant array
    ========================================================
    if runs is arraylike: keep
    if runs is integer N: assume there are N runs from 1 to N
    ========================================================"""
    if type(runs) == int:  # assume runs = n_runs
        runs_out = np.arange(1, runs + 1)
    elif type(runs) == list or type(runs) == np.ndarray:
        runs_out = runs
    else:
        raise TypeError(f"type(runs) ({type(runs)}) must be int, list, or nparray")

    return runs_out


def expand_batches(batches, source):
    """========================================================
    Checks format of 'batches' parameter and returns relevant array
    ========================================================
    if batches is arraylike: keep
    if batches is integer N: assume there a triplet batch from N to N+3
    ========================================================"""
    source = source_shorthand(source=source)
    N = {'gs1826': 3, '4u1820': 2}  # number of epochs
    special = {4, 7}  # special cases (reverse order)

    if type(batches) == int:  # assume batches gives first batch
        if batches in special and source == 'gs1826':
            batches_out = np.arange(batches, batches - 3, -1)
        else:
            batches_out = np.arange(batches, batches + N.get(source, 1))

    elif type(batches) == list or type(batches) == np.ndarray:
        batches_out = batches
    else:
        raise TypeError('type(batches) ({type(batches)}) must be int, list, or nparray')

    return batches_out


def get_nruns(batch, source, **kwargs):
    """========================================================
    Returns the number of runs in a batch
    ========================================================"""
    source = source_shorthand(source=source)
    modelfile = load_modelfile(batch=batch, source=source, **kwargs)
    return len(modelfile)


def source_shorthand(source):
    """========================================================
    Expands source shorthand to full string (e.g. 4u --> 4u1820)
    ========================================================"""
    if source == '4u':
        return '4u1820'
    elif source == 'gs':
        return 'gs1826'
    else:
        return source


def get_source_path(source):
    return os.path.join(GRIDS_PATH, 'sources', source)


def load_modelfile(batch, source, filename='MODELS.txt', **kwargs):
    """========================================================
    Returns the modelfile of a batch
    ========================================================
    batch  =  int  :
    (path  =  str  : path to location of model directories)
    ========================================================"""
    source = source_shorthand(source=source)
    path = kwargs.get('path', MODELS_PATH)
    batch_str = f'{source}_{batch}'
    filepath = os.path.join(path, batch_str, filename)

    modelfile = pd.read_table(filepath, delim_whitespace=True)
    return modelfile


def reduce_table(table, params, exclude={}, verbose=True):
    """=================================================
    Returns the subset of a table that satisfy the specified variables
    =================================================
    table   =  pd.DataFrame  : table to reduce (pandas object)
    params  =  {}            : params that must be satisfied
    exclude = {}             : params to exclude/blacklist completely
    ================================================="""
    subset_idxs = reduce_table_idx(table=table, params=params,
                                   exclude=exclude, verbose=verbose)
    return table.iloc[subset_idxs]


def reduce_table_idx(table, params, exclude={}, verbose=True):
    """=================================================
    Returns the subset of table indices that satisfy the specified variables
    =================================================
    table   =  pd.DataFrame  : table to reduce (pandas object)
    params  =  {}            : params that must be satisfied
    exclude =  {}            : params to exclude/blacklist completely
    ================================================="""
    subset_idxs = get_rows(table=table, params=params, verbose=verbose)

    exclude = ensure_np_list(exclude)
    for key, vals in exclude.items():
        for val in vals:
            idxs_exclude = np.where(table[key] == val)[0]
            to_delete = np.intersect1d(idxs_exclude, subset_idxs)

            for i_del in to_delete:
                subset_idxs = np.delete(subset_idxs, np.where(subset_idxs == i_del))

    return subset_idxs


def get_rows(table, params, verbose):
    """=================================================
    Returns indices of table rows that satify all given params
    ================================================="""
    idxs = {}
    for key, val in params.items():
        idxs[key] = np.where(table[key] == val)[0]

        if len(idxs[key]) == 0:
            printv(f'No row contains {key}={val}', verbose)

    return reduce(np.intersect1d, list(idxs.values()))


def exclude_rows(table, idxs):
    """=================================================
    Returns table with specified rows removed
        NOTE: uses pandas.dataframe indices, not raw indices
    =================================================
    idxs = [] : list of row indexes to exclude/remove from table
    ================================================="""
    mask = table.index.isin(idxs)
    return table[~mask]


def exclude_params(table, params):
    """=================================================
    Returns table with blacklisted parameters removed
        NOTE: only one excluded parameter must be satisfied to be removed
    =================================================
    params = {} : dict of parameter values to exclude/remove from table
    ================================================="""
    params = ensure_np_list(params)
    idxs_exclude = []
    for key, vals in params.items():
        for val in vals:
            idxs_exclude += list(np.where(table[key] == val)[0])

    return exclude_rows(table=table, idxs=idxs_exclude)


def enumerate_params(params_full):
    """=================================================
    Enumerates parameters into a set of all models
    =================================================
    params_full = {}   : specifies all unique values each param will take
    ================================================="""
    params = dict(params_full)
    all_models = dict.fromkeys(params)

    for k in all_models:
        all_models[k] = []

    # === Generate list of param dicts, each one representing a single model ===
    enumerated_params = list(dict(zip(params, x)) for x in itertools.product(*params.values()))

    for i, p in enumerate(enumerated_params):
        for k in all_models:
            all_models[k] = np.append(all_models[k], [p[k]])  # append each model to param lists

    return all_models


def copy_paramfiles(batches, source):
    """========================================================
    Copy MODELS/param table file to grids
    ========================================================"""
    source = source_shorthand(source=source)
    batches = ensure_np_list(variable=batches)

    for batch in batches:
        batch_str = f'{source}_{batch}'
        param_filename = f'params_{batch_str}.txt'
        param_filepath = os.path.join(GRIDS_PATH, 'sources', source, 'params', param_filename)
        models_filepath = os.path.join(MODELS_PATH, batch_str, 'MODELS.txt')

        subprocess.run(['cp', models_filepath, param_filepath])


def rewrite_column(batch, source):
    """========================================================
    Replaces column header 'id' with 'run' in MODELS.txt file
    ========================================================"""
    source = source_shorthand(source=source)
    batch_str = f'{source}_{batch}'
    models_filepath = os.path.join(MODELS_PATH, batch_str, 'MODELS.txt')

    with open(models_filepath) as f:
        lines = f.readlines()
        lines[0] = lines[0].replace('id ', 'run')

    with open(models_filepath, 'w') as f:
        for line in lines:
            f.write(line)


def ensure_np_list(variable):
    """Ensures contents of variable are in the form of list(s)/array(s)
        (Caution: not foolproof. Assumes data is number-like, e.g. no strings)

    input : may be of form dict, integer, float, or array-like
                (Will evaluate all items if variable is a dict)
    """

    def check_value(val):
        """Returns value as np.array if not already"""
        if type(val) in [int, float]:
            return np.array([val])
        else:
            return np.array(val)

    if type(variable) == dict:
        for key, val in variable.items():
            variable[key] = check_value(val=val)
    else:
        variable = check_value(val=variable)

    return variable


def combine_grid_tables(batches, table_basename, source, **kwargs):
    """Reads table files of batches and combines them into a single file
    """

    def get_filepath(base, source, batch, table_path):
        filename = f'{base}_{source}_{batch}.txt'
        return os.path.join(table_path, filename)

    source = source_shorthand(source=source)
    path = kwargs.get('path', GRIDS_PATH)
    table_path = os.path.join(path, 'sources', source, table_basename)

    print(f'Combining grid tables for: {table_basename}')

    # ===== writing column names =====
    filepath = get_filepath(base=table_basename, source=source, batch=batches[0], table_path=table_path)
    table_in = ascii.read(filepath)
    cols = np.concatenate([['batch'], table_in.colnames])
    table_out = pd.DataFrame(columns=cols)

    # ===== copying in batch table =====
    last = batches[-1]
    for batch in batches:
        sys.stdout.write(f'\r{source} {batch}/{last}')
        filepath = get_filepath(base=table_basename, source=source, batch=batch, table_path=table_path)

        table_in = ascii.read(filepath)
        data = table_in.to_pandas()
        data['batch'] = batch
        table_out = pd.concat([table_out, data])
    sys.stdout.write('\n')

    # ===== Ensure column order =====
    table_out = table_out[cols]
    table_str = table_out.to_string(index=False, justify='left', formatters=FORMATTERS,
                                    col_space=8)

    filename = f'{table_basename}_{source}.txt'
    filepath = os.path.join(table_path, filename)

    with open(filepath, 'w') as f:
        f.write(table_str)


def check_finished_multi(batch1, batch2, source, **kwargs):
    """Iterator of check_finished()
    """
    source = source_shorthand(source=source)
    n_epochs = {'gs1826': 3, '4u1820': 2}
    n = n_epochs.get(source, 1)

    for batch in range(batch1, batch2 + 1, n):
        check_finished(batches=batch, source=source, **kwargs)


def check_finished(batches, source, efficiency=True, show='all',
                   basename='xrb', extension='z1', **kwargs):
    """Checks which running models are finished

    t_end      =  flt  : end-time of the simulations
    basename   =  str  : prefix for individual model names
    extension  =  str  : suffix of kepler dump
    efficiency = bool  : print time per 1000 steps
    all        = str   : which models to show, based on their progress,
                    one of (all, finished, not_finished, started, not_started)
    (path      =  str  : path to location of model directories)

    Notes
    -----
    timeused gets reset when a model is resumed,
        resulting in unreliable values in efficiency
    """

    def progress_string(batch, basename, run, progress, elapsed, remaining,
                        eff_str, eff2_str):
        string = [f'{batch}    {basename}{run:02}  {progress:.0f}%   ' +
                  f'{elapsed:.0f}hrs     ~{remaining:.0f}hrs,    ' +
                  f'{eff_str},    {eff2_str}']
        return string

    def shorthand(string):
        map_ = {'a': 'all', 'ns': 'not_started',
                'nf': 'not_finished', 'f': 'finished'}
        if string not in map_:
            if string not in map_.values():
                raise ValueError("invalid 'show' parameter")
            return string
        else:
            return map_[string]

    source = source_shorthand(source=source)
    show = shorthand(show)
    path = kwargs.get('path', MODELS_PATH)
    batches = expand_batches(batches=batches, source=source)

    print_strings = []
    print_idx = {'finished': [], 'not_finished': [],
                 'started': [], 'not_started': []}

    for batch in batches:
        n_runs = get_nruns(batch=batch, source=source, **kwargs)
        print_strings += [f'===== Batch {batch} =====']

        batch_str = f'{source}_{batch}'
        batch_path = os.path.join(path, batch_str)

        for run in range(1, n_runs + 1):
            run_str = f'{basename}{run}'
            run_path = os.path.join(batch_path, run_str)
            string_idx = len(print_strings)

            filename = f'{run_str}{extension}'
            filepath = os.path.join(run_path, filename)

            # ===== get t_end from cmd file =====
            cmd_file = f'{run_str}.cmd'
            cmd_filepath = os.path.join(run_path, cmd_file)

            try:
                with open(cmd_filepath) as f:
                    lines = f.readlines()

                t_end_str = lines[-2].strip()
                t_end = float(t_end_str.strip('@time>'))

                kmodel = kepdump.load(filepath)
                progress = kmodel.time / t_end
                timeused = kmodel.timeused[0][-1]  # CPU time elapsed
                ncyc = kmodel.ncyc  # No. of time-steps
                remaining = (timeused / 3600) * (1 - progress) / progress

                if efficiency:
                    eff = (timeused / (ncyc / 1e4)) / 3600  # Time per 1e4 cyc
                    eff2 = timeused / kmodel.time
                    eff_str = f'{eff:.1f} hr/10Kcyc'
                    eff2_str = f'{eff2:.2f} walltime/modeltime'
                else:
                    eff_str = ''
                    eff2_str = ''

                # ===== Tracking model progress =====
                print_idx['started'] += [string_idx]

                if f'{remaining:.0f}' == '0':
                    print_idx['finished'] += [string_idx]
                else:
                    print_idx['not_finished'] += [string_idx]
            except:
                progress = 0
                timeused = 0
                remaining = 0
                eff_str = ''
                eff2_str = ''

                print_idx['not_started'] += [string_idx]

            progress *= 100
            elapsed = timeused / 3600
            print_strings += progress_string(batch=batch, basename=basename,
                                             run=run, progress=progress, elapsed=elapsed,
                                             remaining=remaining, eff_str=eff_str, eff2_str=eff2_str)

    print_idx['all'] = np.arange(len(print_strings))

    print_dashes()
    print('Batch  Model       elapsed  remaining')
    for i, string in enumerate(print_strings):
        if i in print_idx[show]:
            print(string)


def printv(string, verbose):
    if verbose:
        print(string)


def try_mkdir(path, skip=False):
    try:
        print(f'Creating directory  {path}')
        subprocess.run(['mkdir', path], check=True)
    except:
        if skip:
            print('Directory already exists - skipping')
        else:
            print('Directory exists')
            cont = input('Overwrite? (DESTROY) [y/n]: ')

            if cont == 'y' or cont == 'Y':
                subprocess.run(['rm', '-r', path])
                subprocess.run(['mkdir', path])
            elif cont == 'n' or cont == 'N':
                sys.exit()
