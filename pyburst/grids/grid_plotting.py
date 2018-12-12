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