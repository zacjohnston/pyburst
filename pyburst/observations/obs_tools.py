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


def extract_lightcurve_array(lc_table):
    """Returns simple LC array with [time, flux, u_flux] columns
    """
    x = np.array(lc_table['time'] + 0.5 * lc_table['dt'])
    y = np.array(lc_table['flux'])
    u_y = np.array(lc_table['u_flux'])
    return np.stack([x, y, u_y], axis=1)


def interpolate_lightcurve(lc_table=None, lc_array=None):
    """Returns linear interpolator of epoch lightcurve
    """
    if lc_array is None:
        if lc_table is None:
            raise ValueError('Must provide one of [lc_table, lc_array]')
        else:
            lc_array = extract_lightcurve_array(lc_table)

    return interp1d(x=lc_array[:, 0], y=lc_array[:, 1])
