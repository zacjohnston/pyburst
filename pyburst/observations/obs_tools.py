import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os


# pyburst
from . import obs_strings
from pyburst.grids.grid_tools import write_pandas_table

# TODO:
#   - load epoch
#   - add rate column
#   - add length column

def load_summary(source):
    """Loads summary of observed data

    parameters
    ----------
    source : str
    """
    filepath = obs_strings.summary_filepath(source)
    return pd.read_csv(filepath, delim_whitespace=True)


def save_summary(table, source):
    """Saves summary table to file

    parameters
    ----------
    table : pandas.DataFrame
    source : str
    """
    filepath = obs_strings.summary_filepath(source)
    write_pandas_table(table, filepath=filepath)


def load_epoch_lightcurve(epoch, source):
    """Returns table of epoch lightcurve data

    parameters
    ----------
    epoch : int
    source : str
    """
    columns = ('time', 'dt', 'flux', 'u_flux', 'kt', 'u_kt', 'normal', 'u_normal', 'chi2')
    factors = {'flux': 1e-9, 'u_flux': 1e-9}

    filepath = obs_strings.epoch_filepath(epoch=epoch, source=source)
    table = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None,
                        names=columns)

    for key, item in factors.items():
        table[key] *= item
    return table


def interpolate_lightcurve(lc_table):
    """Returns linear interpolator of epoch lightcurve
    """
    x = np.array(lc_table['time'] + 0.5*lc_table['dt'])
    y = np.array(lc_table['flux'])
    return interp1d(x=x, y=y)
