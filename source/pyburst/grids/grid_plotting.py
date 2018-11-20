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


def save_grid_params(source, param_list=('x', 'z', 'qb', 'mass')):
    """Iterates over grid parameters and plots grid points
    """
    kgrid = grid_analyser.Kgrid(source=source, exclude_test_batches=False,
                                powerfits=False, verbose=False)
    unique = kgrid.unique_params
    savepath = os.path.join(GRIDS_PATH, 'sources', source, 'plots', 'grid')

    for param in param_list:
        not_param = [x for x in param_list if x != param]
        fixed = {}

        # Just use the second of the other params (to avoid border values)
        for not_p in not_param:
            fixed[not_p] = unique[not_p][1]

        fig, ax = kgrid.plot_grid_params(var=['accrate', param], fixed=fixed,
                                         show=False)
        filename = f'grid_{source}_{param}.pdf'
        show_plot(fig, save=True, savepath=savepath, savename=filename)


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