import pandas as pd

# pyburst
from pyburst.grids import grid_analyser
from pyburst.physics import gravity

path = '/home/zac/projects/papers/mcmc/data'
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
        'n_bursts': 'n_used',
        'rate': 'rate',
        'dt': 'dt',
        'energy': 'fluence',
        'peak': 'peak',
        'u_rate': 'u_rate',
        'u_dt': 'u_dt',
        'u_energy': 'u_fluence',
        'u_peak': 'u_peak',
    }

    new_grid = pd.DataFrame()
    kgrid = load_grid()

    # copy parameters
    for col_new, col_old in param_cols.items():
        new_grid[col_new] = kgrid.params[col_old]

    # copy burst output
    for col_new, col_old in summ_cols.items():
        new_grid[col_new] = kgrid.summ[col_old]

    replace_mass_with_g(new_grid)

    return new_grid


def replace_mass_with_g(grid):
    grav_factor = gravity.get_acceleration_newtonian(r=10, m=1.0)
    grav_factor = grav_factor.value / 1e14

    grid['g'] *= grav_factor


def load_grid():
    """Load kepler model grid
    """
    return grid_analyser.Kgrid(source=source)
