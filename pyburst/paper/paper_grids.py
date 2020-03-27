import pandas as pd

# pyburst
from pyburst.grids import grid_analyser
from pyburst.physics import gravity

source = 'grid5'


def pipeline():
    """Creates single pandas table for paper model grid
    """
    param_cols = {
        'mdot': 'accrate',
        'qb': 'qb',
        'x': 'x',
        'z': 'z',
        'g': 'mass',
    }

    summ_cols = {
    }

    new_grid = pd.DataFrame()
    kgrid = load_grid()

    for col_new, col_old in param_cols.items():
        new_grid[col_new] = kgrid.params[col_old]

    return new_grid


def load_grid():
    """Load kepler model grid
    """
    return grid_analyser.Kgrid(source=source)
