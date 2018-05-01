import numpy as np
import matplotlib.pyplot as plt
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
    kgrid = grid_analyser.Kgrid(source=source, load_concord_summ=False,
                    exclude_test_batches=False, powerfits=False, verbose=False)
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

