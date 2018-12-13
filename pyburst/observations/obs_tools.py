import numpy as np
import pandas as pd
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
    # Time [s], dt [s], average flux and error [10^-9 erg/cm^2/s], average
    # blackbody temperature and error [keV], average blackbody normalisation
    # and error [(km/d_10kpc)^2]
    columns = ('time', 'dt', 'flux', 'u_flux', 'kt', 'u_kt', 'normal', 'u_normal', 'chi2')
    filepath = obs_strings.epoch_filepath(epoch=epoch, source=source)
    table = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None,
                        names=columns)
    return table
