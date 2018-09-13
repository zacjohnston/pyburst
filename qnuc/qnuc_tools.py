import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import linregress

# pygrids
from pygrids.grids import grid_tools, grid_analyser, grid_strings
from pygrids.kepler import kepler_tools


def predict_qnuc(params, source, linr_table=None, grid_version=0):
    """Predict optimal Qnuc for given accrate and mass

    linr_table : pd.DataFrame (optional)
        provide linr_table directly (if linr_table=None, will load new table)
    """
    params = params.copy()
    accrate = params.pop('accrate')

    if linr_table is None:
        linr_table = linregress_qnuc(source, grid_version)

    row = grid_tools.reduce_table(linr_table, params=params)

    if len(row) == 0:
        raise ValueError(f'Qnuc not available for {params}')
    elif len(row) > 1:
        raise ValueError(f'Qnuc not uniquely defined for given params (underdefined)')

    return accrate * row.m.values[0] + row.y0.values[0]


def linregress_qnuc(source, grid_version=0):
    """Returns table of linear fits to optimal Qnuc's (versus accretion rate)
    """
    param_list = ['x', 'z', 'qb', 'accdepth', 'accmass', 'mass']
    full_table = load_qnuc_table(source, grid_version)
    accrates = np.unique(full_table['accrate'])  # assumes all parameter-sets have this accrate
    param_table = grid_tools.reduce_table(full_table, params={'accrate': accrates[0]})

    linr_table = param_table.reset_index()[param_list]

    for i in range(len(linr_table)):
        params = linr_table.loc[i][param_list].to_dict()
        sub_table = grid_tools.reduce_table(full_table, params=params)
        linr = linregress(sub_table['accrate'], sub_table['qnuc'])

        linr_table.loc[i, 'm'] = linr[0]
        linr_table.loc[i, 'y0'] = linr[1]
    return linr_table


def extract_qnuc_table(source, grid_version=0, param_batch=None, param_table=None,
                       cycles=None, temp_zone=20):
    """Extracts optimal Qnuc across all parameters

    param_batch : int (optional)
        batch that represents all unique parameters (x, z, accrate, mass)
    param_table : pd.DataFrame (optional)
        alternative to param_batch. A table containing each unique set of paraemeters
    """
    # TODO: make more automatic with grid_version (getting param_table)
    kgrid = grid_analyser.Kgrid(source, linregress_burst_rate=False,
                                grid_version=grid_version)
    if param_table is None:
        if param_batch is None:
            raise ValueError('Must specify one of "param_batch" or "param_table"')
        else:
            param_table = kgrid.get_params(batch=param_batch)
    elif param_batch is not None:
        raise ValueError('Can only specify one of "param_batch" and "param_table"')

    qnuc_table = iterate_solve_qnuc(source, param_table=param_table,
                                    cycles=cycles, kgrid=kgrid, grid_version=grid_version,
                                    temp_zone=temp_zone)
    save_qnuc_table(qnuc_table, source, grid_version)


def load_qnuc_table(source, grid_version=0):
    path = grid_strings.get_source_subdir(source, 'qnuc')
    prefix = f'qnuc_v{grid_version}'
    filename = grid_strings.get_source_filename(source, prefix=prefix, extension='.txt')
    filepath = os.path.join(path, filename)
    return pd.read_table(filepath, delim_whitespace=True)


def save_qnuc_table(table, source, grid_version=0):
    path = grid_strings.get_source_subdir(source, 'qnuc')
    prefix = f'qnuc_v{grid_version}'
    filename = grid_strings.get_source_filename(source, prefix=prefix, extension='.txt')
    filepath = os.path.join(path, filename)

    table_str = table.to_string(index=False)
    with open(filepath, 'w') as f:
        f.write(table_str)


def get_slopes(table, source, cycles=None, basename='xrb', temp_zone=20):
    """Returns slopes of base temperature evolution(K/s), for given model table
    """
    slopes = []
    for row in table.itertuples():
        temps = kepler_tools.extract_temps(row.run, row.batch, source,
                                           cycles=cycles, basename=basename,
                                           temp_zone=temp_zone)
        i0 = 2 if len(temps) > 2 else 1  # skip first dumps if possible
        linr = linregress(temps[i0:, 0], temps[i0:, 1])
        slopes += [linr[0]]

    return np.array(slopes)


def iterate_solve_qnuc(source, param_table, cycles=None, kgrid=None, grid_version=0,
                       temp_zone=20):
    """Iterates over solve_qnuc for a table of params
    """
    param_list = ['x', 'z', 'qb', 'accrate', 'accdepth', 'accmass', 'mass']
    param_table = param_table.reset_index()
    n = len(param_table)
    qnuc = np.zeros(n)

    for row in param_table.itertuples():
        sys.stdout.write(f'\rOptimising Qnuc for parameters: {row.Index+1}/{n}')
        params = {'x': row.x, 'z': row.z, 'accrate': row.accrate, 'mass': row.mass}
        qnuc[row.Index] = solve_qnuc(source=source, params=params,
                                     cycles=cycles, kgrid=kgrid,
                                     grid_version=grid_version,
                                     temp_zone=temp_zone)[0]
    sys.stdout.write('\n')
    qnuc_table = param_table.copy()[param_list]
    qnuc_table['qnuc'] = qnuc
    return qnuc_table


def solve_qnuc(source, params, cycles=None, kgrid=None, grid_version=0, temp_zone=20):
    """Returns predicted Qnuc that gives stable base temperature
    """
    # TODO: add other methods (e.g. bisection)
    param_list = ('x', 'z', 'accrate', 'mass')
    for p in param_list:
        if p not in params:
            raise ValueError(f'Missing "{p}" from "params"')

    if kgrid is None:
        kgrid = grid_analyser.Kgrid(source, linregress_burst_rate=False,
                                    grid_version=grid_version)

    subset = kgrid.get_params(params=params)
    slopes = get_slopes(table=subset, source=source, cycles=cycles, temp_zone=temp_zone)

    linr = linregress(subset['qnuc'], slopes)
    x0 = -linr[1]/linr[0]  # x0 = -y0/m
    u_x0 = (linr[4] / linr[0]) * x0
    return x0, u_x0


def add_qnuc_column(table, qnuc_source, grid_version=0):
    """Iterates over parameters in table, and adds a predicted qnuc column
    """
    param_list = ['x', 'z', 'qb', 'accdepth', 'accmass', 'mass']
    linr_table = linregress_qnuc(qnuc_source, grid_version)
    for i in range(len(table)):
        params = table[param_list].iloc[i].to_dict()
        table.loc[i, 'qnuc'] = predict_qnuc(params=params, source=qnuc_source,
                                            linr_table=linr_table,
                                            grid_version=grid_version)
    return table
