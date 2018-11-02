import numpy as np
import pandas as pd

from pygrids.grids import grid_tools
from pygrids.mcmc import mcmc_versions, mcmc_tools

# TODO
#   - generate synthetic data (load, add noise, save, tables)
#   - load models (groups)
#   - add noise
#   - write table (input, output): files/synth


def setup_table(kgrid, batches, source, mc_version):
    """Sets up table of synthetic data, including input/output values

    parameters
    ----------
    kgrid : Kgrid object
    batches : array
        list of batches, each corresponding to an epoch. Assumes the runs in each
        batch correspond to each other.
    source : str
    mc_version : int
    """
    mcv = mcmc_versions.McmcVersion(source=source, version=mc_version)
    sub = grid_tools.reduce_table(kgrid.params, params={'batch': batches[0]})
    groups = np.array(sub['run'])

    table = pd.DataFrame()

    for group in groups:
        group_table = initialise_group_table(group, batches)
        set_param_cols(group_table, batches=batches, kgrid=kgrid)
        set_summ_cols(group_table, batches=batches, kgrid=kgrid)
        set_free_params(group_table, mcv=mcv)

        # TODO:
        #   - Calculate observables (from summ values and conversion factors)
        #   - Calculate f_per (from accrate and conversion factors)
        table = pd.concat([table, group_table], ignore_index=True)

    return table


def initialise_group_table(group, batches):
    """Initialises a table for a single group of epochs

    parameters
    ----------
    group : int
        group ID number
    batches : sequence
        list of batch ID numbers that correspond to epochs
    """
    n_epochs = len(batches)
    epochs = np.arange(n_epochs) + 1

    group_table = pd.DataFrame()
    group_table['group'] = np.full(n_epochs, group)
    group_table['epoch'] = epochs
    group_table['batch'] = batches

    return group_table


def set_param_cols(group_table, batches, kgrid,
                   params=('x', 'z', 'accrate', 'qb', 'mass')):
    """Sets model parameter columns in table for a single group of epochs
    """
    group_params = kgrid.get_params(run=group_table.group[0]
                                    ).set_index(['batch']).loc[batches]
    for var in params:
        group_table[var] = np.array(group_params[var])


def set_summ_cols(group_table, batches, kgrid,
                  summ_list=('rate', 'dt', 'fluence', 'peak')):
    """Sets summ value columns in table for a single group of epochs
    """
    group_summ = kgrid.get_summ(run=group_table.group[0]
                                ).set_index(['batch']).loc[batches]

    for var in summ_list:
        u_var = f'u_{var}'
        group_table[var] = np.array(group_summ[var])
        group_table[u_var] = np.array(group_summ[u_var])


def set_free_params(group_table, mcv,
                    free_params=('redshift', 'd_b', 'xi_ratio')):
    """Chooses random free parameters for a given group of epochs

    parameters
    ----------
    group_table : pd.DataFrame
        table of a single group of epochs
    mcv : McmcVersion
        Object of mcmc constraints (for boundaries of free parameters)
    free_params : sequence
        free parameters to generate
    """
    for var in free_params:
        rand_x = mcmc_tools.get_random_params(var, n_models=1, mv=mcv)[0]
        group_table[var] = rand_x

