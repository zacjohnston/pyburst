import numpy as np
import pandas as pd
import os


# pyburst
from . import obs_strings

# TODO:
#   - load summary
#   - load epoch
#   - add rate column
#   - add length column

def load_summary(source):
    """Loads summary of observed data
    """
    filepath = obs_strings.summary_filepath(source)
    return pd.read_csv(filepath, delim_whitespace=True)
