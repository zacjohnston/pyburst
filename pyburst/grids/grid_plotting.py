import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# kepler_grids
from . import grid_analyser

GRIDS_PATH = os.environ['KEPLER_GRIDS']


def show_plot(fig, save, savepath, savename):
    if save:
        filepath = os.path.join(savepath, savename)
        print(f'Saving: {filepath}')
        plt.savefig(filepath)
        plt.close(fig)
    else:
        plt.show(block=False)


def plot_grid(kgrid, fixed_vals=(2.0, 1.1), fixed_param='mass',
              x_param='accrate', y_param='qb'):
    """Plot grid points for which a model exists
        Useful for diagnosing missing parts of grid.

    fixed : {}
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f'{fixed_param}={fixed_vals}')

    for i, fixed_val in enumerate(fixed_vals):
        for x_val in kgrid.unique_params[x_param]:
            params = {fixed_param: fixed_val, x_param: x_val}
            y = np.unique(kgrid.get_params(params=params)[y_param])
            x = np.full_like(y, x_val)

            color = {0: 'black', 1: 'C3'}.get(i)
            ax.plot(x, y, marker='o', ls='none', color=color, markersize=i*2+5)

    plt.show(block=False)


def plot_flags(kgrid, fixed=None, flag='short_waits'):
    """Map out parameters where short-wait bursts occur
    """
    if fixed is None:
        fixed = {'z': 0.005, 'mass': 1.4}

    fig, ax = plt.subplots()
    sub_p = kgrid.get_params(params=fixed)
    sub_s = kgrid.get_summ(params=fixed)

    short_map = sub_s[flag]
    shorts = sub_p[short_map]
    not_shorts = sub_p[np.invert(short_map)]

    ax.plot(not_shorts['accrate'], not_shorts['x'], marker='o', ls='none')
    ax.plot(shorts['accrate'], shorts['x'], marker='o', ls='none', color='C3')
    ax.set_title(fixed)
    ax.set_xlabel('accrate')
    ax.set_ylabel('X')
    plt.show(block=False)


def get_subset(kgrid, sub_params=None):
    if sub_params is None:
        sub_params = {'x': [0.65, 0.7, 0.75]}

    subsets = []
    for x in sub_params['x']:
        subsets += kgrid.get_params(params={'x': x})

    return pd.concat(subsets, ignore_index=True)