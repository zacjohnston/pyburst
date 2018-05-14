import numpy as np
import pandas as pd
import os

# kepler_grids
from ..grids import grid_strings, grid_tools

GRIDS_PATH = os.environ['KEPLER_GRIDS']
bprops = ['dt', 'u_dt', 'fper', 'u_fper', 'fluence', 'u_fluence', 'peak', 'u_peak']


def extract_obs_data(series, sim_batch, sim_name='sim'):
    """Returns obs_data in dict form, e.g. for burstfit
    """
    summary = load_summary(sim_batch, sim_name)
    subset = grid_tools.reduce_table(summary, params={'series': series})

    obs_data = {}
    for bprop in bprops:
        obs_data[bprop] = np.array(subset[bprop])

    return obs_data


def load_summary(sim_batch, sim_name='sim'):
    source = get_source_string(sim_batch, sim_name)
    path = grid_strings.get_obs_data_path(source)
    filename = grid_strings.get_source_filename(source, prefix='summary', extension='.txt')
    filepath = os.path.join(path, filename)

    return pd.read_csv(filepath, delim_whitespace=True)


def get_summary(sim_batch, sim_name='sim', save=True):
    table = load_info(sim_batch, sim_name)
    new_cols = {'fluence': [], 'u_fluence': [],
                'peak': [], 'u_peak': []}

    n_entries = len(table)
    for i in range(n_entries):
        series = int(table.iloc[i]['series'])
        epoch = int(table.iloc[i]['number'])

        lightcurve = load_lightcurve(epoch, series, sim_batch, sim_name)
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
        write_summary(table, sim_batch, sim_name)

    return table


def write_summary(table, sim_batch, sim_name='sim'):
    cols = ['series', 'epoch', 'dt', 'u_dt', 'fper', 'u_fper',
            'fluence', 'u_fluence', 'peak', 'u_peak', 'bol']
    table_out = table[cols]
    table_str = table_out.to_string(index=False, justify='left',  # formatters=FORMATTERS,
                                    col_space=12)

    source = get_source_string(sim_batch, sim_name)
    path = grid_strings.get_obs_data_path(source)
    filename = grid_strings.get_source_filename(source, prefix='summary', extension='.txt')
    filepath = os.path.join(path, filename)

    with open(filepath, 'w') as f:
        f.write(table_str)


def get_batch_series(full_source):
    """Extracts sim_batch and series values from full source string

    e.g.: sim10_1 would return (10, 1)
    """
    stripped = full_source.strip('sim')
    sim_batch = int(stripped[:2])
    series = int(stripped[3:])
    return sim_batch, series


def get_source_string(sim_batch, sim_name):
    return f'{sim_name}{sim_batch}'


def reformat_source(full_source):
    """Reformats source string into one usable by other modules
    """
    idx = full_source.find('_')
    return full_source[:idx]


def load_info(sim_batch, sim_name='sim'):
    """Load simulated data info

    Returns: pandas.DataFrame object"""
    source = get_source_string(sim_batch, sim_name)
    columns = ['series', 'number', 'dt', 'u_dt', 'fper', 'u_fper', 'bol']

    filename = f'{sim_name}_info.csv'
    path = grid_strings.get_obs_data_path(source)
    filepath = os.path.join(path, filename)

    return pd.read_csv(filepath, delim_whitespace=True, skiprows=16, names=columns)


def load_lightcurve(epoch, series, sim_batch, sim_name='sim'):
    source = get_source_string(sim_batch, sim_name)
    columns = ['time', 'time_step', 'flux', 'u_flux']

    filename = f'{sim_name}{series}_{epoch}.csv'
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
