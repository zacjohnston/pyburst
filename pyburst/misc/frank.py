import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pyburst
from pyburst import kepler
from pyburst.grids import grid_strings


def profile_path(run, batch, source='frank', basename='xrb'):
    """Return path to directory containing profile data
    """
    path = grid_strings.get_source_subdir(source, 'profiles')
    run_str = grid_strings.get_run_string(run=run, basename=basename)
    return os.path.join(path, run_str)
