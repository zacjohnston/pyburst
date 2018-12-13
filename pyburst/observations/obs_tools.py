import numpy as np
import pandas as pd
import os


# pyburst
from . import obs_strings
from pyburst.grids.grid_tools import write_pandas_table

# TODO:
#   - load summary
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
