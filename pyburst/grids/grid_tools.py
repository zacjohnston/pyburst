import numpy as np
import pandas as pd
import sys
import os
import itertools
import subprocess
from astropy.io import ascii

# kepler_grids
from pyburst.misc.pyprint import print_dashes
from . import grid_strings

# kepler
import kepdump

flt2 = '{:.2f}'.format
flt4 = '{:.4f}'.format
exp2 = '{:.2e}'.format
FORMATTERS = {'z': flt4, 'y': flt4, 'x': flt4, 'accrate': flt4,
              'tshift': flt2, 'qb': flt4, 'acc_mult': flt2, 'qb_delay': flt2,
              'mass': flt2, 'accmass': exp2, 'accdepth': exp2}

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


# TODO: rewrite docstrings


def write_pandas_table(table, filepath):
    """Write a given pandas table to file
    """
    print(f'Writing: {filepath}')
    table_str = table.to_string(index=False, formatters=FORMATTERS)
    with open(filepath, 'w') as f:
        f.write(table_str)


def load_grid_table(tablename, source, verbose=True, burst_analyser=False):
    """Returns file of model parameters as pandas DataFrame
    
    tablename  = str   : table name (e.g. 'params', 'summ')
    source     = str   : name of source object
    """
    source = grid_strings.source_shorthand(source)

    if burst_analyser and tablename == 'summ':
        param_path = grid_strings.get_source_subdir(source, 'burst_analysis')
        filename = f'burst_analysis_{source}.txt'
        filepath = os.path.join(param_path, filename)
    else:
        param_path = grid_strings.get_source_subdir(source, tablename)
        filename = f'{tablename}_{source}.txt'
        filepath = os.path.join(param_path, filename)

    printv(f'Loading {tablename} table: {filepath}', verbose)
    params = pd.read_table(filepath, delim_whitespace=True)
    return params


def expand_runs(runs):
    """Checks format of 'runs' parameter and returns relevant array
    
    if runs is arraylike: keep
    if runs is integer N: assume there are N runs from 1 to N
    """
    if type(runs) == int:  # assume runs = n_runs
        runs_out = np.arange(1, runs + 1)
    elif type(runs) == list or type(runs) == np.ndarray:
        runs_out = runs
    else:
        raise TypeError(f"type(runs) ({type(runs)}) must be int, list, or nparray")

    return runs_out


def expand_batches(batches, source):
    """Checks format of 'batches' parameter and returns relevant array
    
    if batches is arraylike: keep
    if batches is integer N: assume there a triplet batch from N to N+3
    """
    source = grid_strings.source_shorthand(source=source)
    n = {'gs1826': 3, '4u1820': 2}  # number of epochs
    special = {4, 7}  # special cases (reverse order)
    b_type = type(batches)

    if b_type is int \
            or b_type is np.int64:  # assume batches gives first batch
        if batches in special and source == 'gs1826':
            batches_out = np.arange(batches, batches - 3, -1)
        else:
            batches_out = np.arange(batches, batches + n.get(source, 1))

    elif b_type is list or b_type is np.ndarray:
        batches_out = batches
    else:
        raise TypeError(f'type(batches) ({b_type}) must be int, list, or nparray')

    return batches_out


def get_nruns(batch, source):
    """Returns the number of runs in a batch
    """
    source = grid_strings.source_shorthand(source=source)
    model_table = load_model_table(batch=batch, source=source)
    return len(model_table)


def load_model_table(batch, source, filename='MODELS.txt'):
    """Returns the model_table of a batch
    """
    source = grid_strings.source_shorthand(source=source)
    filepath = grid_strings.get_model_table_filepath(batch, source, filename)
    print(f'Loading: {filepath}')
    model_table = pd.read_table(filepath, delim_whitespace=True)
    return model_table


def add_model_column(batches, source, col_name, col_value, filename='MODELS.txt'):
    """Adds a column to model table(s) file.
    Note: can also be used to rewrite column values

    parameters
    ----------
    batches : int|array
    source : str
    col_name : str
    col_value : int|float|array
        contents of new column. If a single value, will fill whole column.
        If array-like, must be correct length
    filename : str
    """
    print(f'Adding column: {col_name}')
    batches = ensure_np_list(batches)

    for batch in batches:
        table = load_model_table(batch, source=source, filename=filename)
        table[col_name] = col_value

        filepath = grid_strings.get_model_table_filepath(batch, source, filename=filename)
        write_pandas_table(table, filepath)


def reduce_table(table, params, exclude_any=None, exclude_all=None):
    """Returns the subset of a table that satisfy the specified variables

    table : pd.DataFrame
      table to reduce (pandas table)
    params : dict
        params that must all be satisfied (each value must be scalar)
    exclude : dict
        params to exclude/blacklist completely (can be arrays for multiple values)
    exclude_all : dict
        similar to exclude, but every parameter value must be satisfied to exclude
    """
    mask = param_mask_all(table, params)
    sub_table = table[mask].copy()
    sub_table = exclude_params(sub_table, params=exclude_any, logic='any')
    sub_table = exclude_params(sub_table, params=exclude_all, logic='all')
    return sub_table


def reduce_table_idx(table, params, exclude_any=None, exclude_all=None):
    """Returns the subset of table indices that satisfy the specified variables
        Same as reduce_table(), but returns indices instead of table
    
    table : pd.DataFrame
      table to reduce (pandas table)
    params : dict
        params that must all be satisfied (each value must be scalar)
    exclude : dict
        params to exclude/blacklist completely (can be arrays for multiple values)
    """
    table_copy = table.reset_index()
    sub_table = reduce_table(table_copy, params, exclude_any=exclude_any,
                             exclude_all=exclude_all)
    return np.array(sub_table.index)


def get_rows(table, params):
    """Returns indices of table rows that satify all given params
    """
    table_copy = table.reset_index()
    mask = param_mask_all(table_copy, params)
    return np.array(table_copy[mask].index)


def exclude_params(table, params, logic):
    """
    Returns table with blacklisted parameters excluded
        NOTE: only one excluded parameter must be satisfied to be removed
    
    params : dict
        parameters to exclude from table.
        Each key specifies parameter name, and its value can be a scalar or array
    logic : ['any', 'all']
        boolean logic to use:
            'any' - models will be excluded when ANY parameters match
            'all' - model will only be excluded if ALL parameters match (must be scalars)

        If 'all', params is a list of dicts specifying multiple sets
                    of parameter combinations
    """
    def check_type(params_, logic_):
        types = {'any': dict, 'all': list}
        if type(params_) is not types[logic_]:
            raise TypeError(f'for exclude_logic={logic_}, params must be type({types[logic_]})')

    mask_excluded = np.full(len(table), False)  # model to be excluded if True

    if logic == 'any':
        if params not in [None, {}]:
            check_type(params, logic)
            mask_excluded = param_mask_any(table, params)

    elif logic == 'all':
        if params not in [None, {}, [], [{}]]:
            check_type(params, logic)
            for param_set in params:
                mask_excluded = mask_excluded | param_mask_all(table, param_set)
    else:
        raise ValueError("'logic' must be one of ['any', 'all']")

    return table[~mask_excluded]


def param_mask_any(table, params):
    """Returns boolean mask of table where any of the specified params are satisfied

    table : pandas.DataFrame
    params : dict
    """
    params = ensure_np_list(params)
    mask = np.full(len(table), False)
    for param, values in params.items():
        mask = mask | table[param].isin(values)
    return mask


def param_mask_all(table, params):
    """Returns boolean mask of table where ALL of the specified params are satisfied
    """
    check_scalars(params)
    mask = np.full(len(table), True)
    for param, value in params.items():
        mask = mask & (table[param] == value)
    return mask


def check_scalars(params):
    """Check if all items in parameter dictionary are scalars and not arrays
    """
    for key, value in params.items():
        if hasattr(value, '__len__') and (not isinstance(value, str)):
            raise TypeError("values in params must be scalars")


def enumerate_params(params_full):
    """Enumerates parameters into a set of all models
    
    params_full = {}   : specifies all unique values each param will take
    """
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
    """Copy MODELS/param table file from kepler to grids
    """
    source = grid_strings.source_shorthand(source=source)
    batches = ensure_np_list(variable=batches)

    for batch in batches:
        params_filepath = grid_strings.get_table_filepath(source, 'params', batch=batch)
        model_table_filepath = grid_strings.get_model_table_filepath(batch, source)
        subprocess.run(['cp', model_table_filepath, params_filepath])


def rename_model_column(batch, source, col_old, col_new):
    """Renames a column in MODELS.txt file
    """
    source = grid_strings.source_shorthand(source=source)
    filepath = grid_strings.get_model_table_filepath(batch, source)
    table = load_model_table(batch, source)
    table = table.rename(index=str, columns={col_old: col_new})
    write_pandas_table(table, filepath)


def ensure_np_list(variable):
    """Ensures contents of variable are in the form of list(s)/array(s)
        (Caution: not foolproof. Assumes data is number-like, e.g. no strings)

    input : may be of form dict, integer, float, or array-like
                (Will evaluate all items if variable is a dict)
    """

    def check_value(var):
        """Returns value as np.array if not already"""
        if type(var) in [np.ndarray, list, tuple]:
            return np.array(var)
        else:
            return np.array([var])

    if type(variable) == dict:
        for key, val in variable.items():
            variable[key] = check_value(val)
    else:
        variable = check_value(variable)

    return variable


def get_unique_param(param, source):
    """Return unique values of given parameter
    """
    source = grid_strings.source_shorthand(source=source)
    params_filepath = grid_strings.get_table_filepath(source, 'params')
    param_table = pd.read_table(params_filepath, delim_whitespace=True)
    return np.unique(param_table[param])


def combine_grid_tables(batches, table_basename, source, **kwargs):
    """Reads table files of batches and combines them into a single file
    """
    def get_filepath(base, source, batch, table_path):
        filename = grid_strings.get_batch_filename(prefix=base, batch=batch, source=source,
                                                   extension='.txt')
        return os.path.join(table_path, filename)

    source = grid_strings.source_shorthand(source=source)
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
    filename = grid_strings.get_source_filename(source, table_basename, extension='.txt')
    filepath = os.path.join(table_path, filename)
    write_pandas_table(table_out, filepath)


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

    source = grid_strings.source_shorthand(source=source)
    show = shorthand(show)
    batches = expand_batches(batches=batches, source=source)

    print_strings = []
    print_idx = {'finished': [], 'not_finished': [],
                 'started': [], 'not_started': []}

    for batch in batches:
        n_runs = get_nruns(batch=batch, source=source)
        print_strings += [f'===== Batch {batch} =====']
        batch_path = grid_strings.get_batch_models_path(batch, source)

        for run in range(1, n_runs + 1):
            run_str = grid_strings.get_run_string(run, basename)
            run_path = grid_strings.get_model_path(run, batch, source, basename=basename)
            string_idx = len(print_strings)

            filename = f'{run_str}{extension}'
            filepath = os.path.join(run_path, filename)

            # ===== get t_end from cmd file =====
            cmd_file = f'{run_str}.cmd'
            cmd_filepath = os.path.join(run_path, cmd_file)

            try:
                with open(cmd_filepath) as f:
                    lines = f.readlines()

                marker = '@time>'
                for line in lines[-10:]:
                    if marker in line:
                        t_end = float(line.strip('@time>').strip())
                        break

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


def print_params_summary(table, show=None):
    """Print summary of unique params in a given table

    parameters
    ----------
    table : pandas.DataFrame
        table of models to summarise (subset of self.params)
    show : [str] (optional)
        specify parameters to show.
        defaults to ['accrate', 'x', 'z', 'qb', 'mass']
    """
    if type(table) != pd.core.frame.DataFrame:
        raise TypeError('table must be pandas.DataFrame')

    if show is None:
        show = ['accrate', 'x', 'z', 'qb', 'mass']

    for param in show:
        unique = np.unique(table[param])
        print(f'{param} = {unique}')


def printv(string, verbose):
    if verbose:
        print(string)


def try_mkdir(path, skip=False):
    print(f'Creating directory  {path}')
    if os.path.exists(path):
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
    else:
        subprocess.run(['mkdir', '-p', path], check=True)

