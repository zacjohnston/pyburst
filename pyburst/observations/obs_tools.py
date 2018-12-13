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


def get_peak_length(lc_table, peak_frac=0.75):
    """Returns peak length for given lightcurve table
    """
    lc_filled = fill_lightcurve(lc_table)
    peak = np.max(lc_table.flux)
    mask = lc_filled[:, 1] > peak_frac*peak
    time_slice = lc_filled[mask, 0]

    return time_slice[-1] - time_slice[0]


def fill_lightcurve(lc_table, n_x=1000):
    """Returns lightcurve [time, flux] with higher time-sampling (interpolated)
    """
    lc_array = extract_lightcurve_array(lc_table)
    interp = interpolate_lightcurve(lc_array=lc_array)

    t0 = lc_array[0, 0]
    t1 = lc_array[-1, 0]
    x = np.linspace(t0, t1, n_x)
    y = interp(x)

    return np.stack([x, y], axis=1)


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
