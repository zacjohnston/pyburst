import numpy as np
import pandas as pd

from pygrids.grids import grid_analyser, grid_tools


# TODO
#   - generate synthetic data (load, add noise, save, tables)
#   - load models (groups)
#   - add noise
#   - write table (input, output): files/synth


def setup_table(kgrid, batches):
    """Sets up table of synthetic data, including input/output values

    parameters
    ----------
    kgrid : Kgrid object
    batches : array
        list of batches, each corresponding to an epoch. Assumes the runs in each
        batch correspond to each other.
    """
    param_list = ('x', 'z', 'accrate', 'qb', 'mass')
    summ_list = ('rate', 'dt', 'fluence', 'peak')

    sub = grid_tools.reduce_table(kgrid.params, params={'batch': batches[0]})
    groups = np.array(sub['run'])

    n_groups = len(groups)
    n_epochs = len(batches)
    n_models = n_epochs * n_groups
    epochs = np.arange(n_epochs) + 1

    table = pd.DataFrame()

    for group in groups:
        group_params = kgrid.get_params(run=group).set_index(['batch']).loc[batches]
        group_summ = kgrid.get_summ(run=group).set_index(['batch']).loc[batches]

        group_table = pd.DataFrame()
        group_table['group'] = np.full(n_epochs, group)
        group_table['epoch'] = epochs
        group_table['batch'] = batches

        for var in param_list:
            group_table[var] = np.array(group_params[var])
        for var in summ_list:
            u_var = f'u_{var}'
            group_table[var] = np.array(group_summ[var])
            group_table[u_var] = np.array(group_summ[u_var])

        # TODO:
        #   - Randomly choose conversion factors (redshift, d_b, xi_ratio)
        #   - Calculate observables (from summ values and conversion factors)
        #   - Calculate f_per (from accrate and conversion factors)
        table = pd.concat([table, group_table], ignore_index=True)

    return table
