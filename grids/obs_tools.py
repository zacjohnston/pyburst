import numpy as np
import pandas as pd
import os

# kepler_grids
from ..grids import grid_strings

GRIDS_PATH = os.environ['KEPLER_GRIDS']


def get_source_string(sim_batch, sim_name):
    return f'{sim_name}{sim_batch}'


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
