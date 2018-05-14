import numpy as np
import pandas as pd
import os

# kepler_grids
from ..grids import grid_strings

GRIDS_PATH = os.environ['KEPLER_GRIDS']


def load_sim_info(sim_batch, sim_name='sim'):
    """Load simulated data info

    Returns: pandas.DataFrame object"""
    source = f'{sim_name}{sim_batch}'
    columns = ['series', 'number', 'dt', 'u_dt', 'fper', 'u_fper', 'bol']

    filename = f'{sim_name}_info.csv'
    path = grid_strings.get_obs_data_path(source)
    filepath = os.path.join(path, filename)

    table = pd.read_csv(filepath, delim_whitespace=True, skiprows=16, names=columns)
    return table
