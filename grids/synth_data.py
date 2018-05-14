"""
Tools for handling synthetic observations, for the purpose of testing mcmc methods

General structure:
    A collection of synthetic data will be organised into the following hierarchy:
        1. batch:
            the collection of synthetic observations
        2. series:
            each batch may have multiple series, each representing a synthetic "source"
        3. epoch:
            each series may have multiple accretion epochs, differing only
            by accretion rate.

Note on nomenclature:
    "source" here refers to an expanded source string (e.g. 'sim10_2'),
    which also specifies the sim_batch (e.g. 10) and the series (e.g. 2).
    Methods in this module can break this into a smaller "source" string
    for the purposes of organising directories in kepler_grids, to avoid
    having to create/define a new "source" for every synthetic test.

    In other words: The source would be 'sim10_2' according to synth_data,
                    but the source according to kepler_grids in general
                    (particularly with regards to paths) would be 'sim10'.
"""
import numpy as np
import pandas as pd
import os

# kepler_grids
from ..grids import grid_strings, grid_tools

GRIDS_PATH = os.environ['KEPLER_GRIDS']
bprops = ['dt', 'u_dt', 'fper', 'u_fper', 'fluence', 'u_fluence', 'peak', 'u_peak']


def extract_obs_data(source):
    """Returns obs_data in dict form, e.g. for burstfit
    """
    _, series = get_batch_series(source)
    summary = load_summary(source)
    subset = grid_tools.reduce_table(summary, params={'series': series})

    obs_data = {}
    for bprop in bprops:
        obs_data[bprop] = np.array(subset[bprop])

    return obs_data


def load_summary(source):
    path = grid_strings.get_obs_data_path(source)
    trimmed_source = grid_strings.check_synth_source(source)
    filename = grid_strings.get_source_filename(trimmed_source,
                                                prefix='summary', extension='.txt')
    filepath = os.path.join(path, filename)
    return pd.read_csv(filepath, delim_whitespace=True)


def get_summary(source, save=True):
    table = load_info(source)
    new_cols = {'fluence': [], 'u_fluence': [],
                'peak': [], 'u_peak': []}

    n_entries = len(table)
    for i in range(n_entries):
        epoch = int(table.iloc[i]['number'])
        lightcurve = load_lightcurve(epoch, source)

        flu, u_flu = get_fluence(lightcurve)
        peak, u_peak = get_peak(lightcurve)

        new_cols['fluence'] += [flu]
        new_cols['u_fluence'] += [u_flu]
        new_cols['peak'] += [peak]
        new_cols['u_peak'] += [u_peak]
    for col in new_cols:
        table[col] = new_cols[col]

    table = table.rename(columns={'number': 'epoch'})
    if save:
        write_summary(table, source)

    return table


def write_summary(table, source):
    cols = ['series', 'epoch', 'dt', 'u_dt', 'fper', 'u_fper',
            'fluence', 'u_fluence', 'peak', 'u_peak', 'bol']
    table_out = table[cols]
    table_str = table_out.to_string(index=False, justify='left', col_space=12)

    path = grid_strings.get_obs_data_path(source)
    filename = grid_strings.get_source_filename(source, prefix='summary', extension='.txt')
    filepath = os.path.join(path, filename)

    with open(filepath, 'w') as f:
        f.write(table_str)


def get_batch_series(source):
    """Extracts sim_batch and series values from full source string

    e.g.: sim10_1 would return (10, 1)
    """
    stripped = source.strip('sim')
    sim_batch = int(stripped[:2])
    series = int(stripped[3:])
    return sim_batch, series


def load_info(source):
    """Load simulated data info

    Returns: pandas.DataFrame object"""
    columns = ['series', 'number', 'dt', 'u_dt', 'fper', 'u_fper', 'bol']

    filename = f'sim_info.csv'
    path = grid_strings.get_obs_data_path(source)
    filepath = os.path.join(path, filename)

    return pd.read_csv(filepath, delim_whitespace=True, skiprows=16, names=columns)


def load_lightcurve(epoch, source):
    columns = ['time', 'time_step', 'flux', 'u_flux']
    _, series = get_batch_series(source)

    filename = f'sim{series}_{epoch}.csv'
    path = grid_strings.get_obs_data_path(source)
    filepath = os.path.join(path, 'lightcurves', filename)

    return pd.read_csv(filepath, skiprows=16, names=columns)


def get_fluence(lightcurve):
    """Returns fluence and u_fluence, given a lightcurve table (from load_lightcurve)
    """
    time_bins = 2 * lightcurve['time_step']
    flux = lightcurve['flux']
    u_flux = lightcurve['u_flux']

    fluence = np.sum(time_bins * flux) * 1e-9

    u_fluence_i = time_bins * u_flux
    u_fluence = np.sqrt(np.sum(u_fluence_i**2)) * 1e-9

    return fluence, u_fluence


def get_peak(lightcurve):
    """Returns peak lfux and u_peak"""
    idx = np.argmax(lightcurve['flux'])

    peak = lightcurve.iloc[idx]['flux'] * 1e-9
    u_peak = lightcurve.iloc[idx]['u_flux'] * 1e-9

    return peak, u_peak
