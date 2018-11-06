import numpy as np
import pandas as pd
import os
import astropy.constants as const
from astropy import units

# kepler_grids
from pygrids.grids import grid_tools, grid_strings
from pygrids.mcmc import mcmc_versions, mcmc_tools

# TODO
#   - generate synthetic data (load, add noise, save, tables)
#   - add noise
#   - write table (input, output): files/synth

c = const.c.to(units.cm / units.s)
msunyer_to_gramsec = (units.M_sun / units.year).to(units.g / units.s)
mdot_edd = 1.75e-8 * msunyer_to_gramsec
kpc_to_cm = units.kpc.to(units.cm)
mass_ref = 1.4


def setup_table(kgrid, batches, synth_source, mc_source, mc_version, synth_version,
                params=('x', 'z', 'accrate', 'qb', 'mass'),
                summ_list=('rate', 'dt', 'fluence', 'peak'),
                free_params=('redshift', 'd_b', 'xi_ratio'),
                observables=('rate', 'peak', 'fluence', 'fper'),
                ):
    """Sets up table of synthetic data, including input/output values

    parameters
    ----------
    kgrid : Kgrid object
    batches : array
        list of batches, each corresponding to an epoch. Assumes the runs in each
        batch correspond to each other.
    synth_source : str
    mc_source : str
    mc_version : int
    synth_version : int
    params : sequence(str)
        parameters to extract from kgrid and add to the table
    summ_list : sequence(str)
        burst properties to extract from kgrid and add to the table
    free_params : sequence(str)
        free parameters to randomly choose
    observables : sequence(str)
        names of observables to calculate from burst properties
    """
    mcv = mcmc_versions.McmcVersion(source=mc_source, version=mc_version)
    sub = grid_tools.reduce_table(kgrid.params, params={'batch': batches[0]})
    groups = np.array(sub['run'])

    table = pd.DataFrame()

    # ===== For each group of epochs, setup sub-table of inputs/outputs =====
    for group in groups:
        group_table = initialise_group_table(group, batches)

        set_param_cols(group_table, batches=batches, kgrid=kgrid, params=params)
        set_summ_cols(group_table, batches=batches, kgrid=kgrid, summ_list=summ_list)
        set_rand_free_params(group_table, mcv=mcv, free_params=free_params)
        set_observables(group_table, observables=observables)

        table = pd.concat([table, group_table], ignore_index=True)

    save_table(table, source=synth_source, version=synth_version)
    return table


def save_table(table, source, version):
    """Save synth table to file
    """
    filepath = get_table_filepath(source, version, try_mkdir=True)
    grid_tools.write_pandas_table(table, filepath)


def load_table(source, version):
    """Load synth table from file
    """
    filepath = get_table_filepath(source, version=version)
    return pd.read_table(filepath, delim_whitespace=True)


def load_group_table(source, version, group):
    """Returns synth table of single group of epochs
    """
    table = load_table(source, version)
    return grid_tools.reduce_table(table, params={'group': group})


def get_true_values(source, group, version,
                    params=('accrate', 'x', 'z', 'qb', 'mass', 'redshift', 'd_b', 'xi_ratio')):
    """Returns the "True" synthetic values to compare with posteriors
    """
    truth = np.array([])
    group_table = load_group_table(source, version, group)
    multiplier = {'mass': 1/mass_ref}

    for param in params:
        mult = multiplier.get(param, 1.0)

        values = np.array(group_table[param]) * mult
        if len(np.unique(values)) == 1:
            values = np.array([values[0]])
        truth = np.concatenate((truth, values))

    return truth

def get_table_filepath(source, version, try_mkdir=False):
    """Returns filepath of synth table
    """
    path = grid_strings.get_obs_data_path(source)
    if try_mkdir:
        grid_tools.try_mkdir(path, skip=True)

    filename = f'synth_{source}_{version}.txt'
    return os.path.join(path, filename)


def extract_obs_data(source, version, group,
                     bprops=('rate', 'fluence', 'peak', 'fper')):
    """Returns summary "observed" data as dictionary, for Burstfit
    """
    table = load_table(source, version)
    group_table = grid_tools.reduce_table(table, params={'group': group})

    obs_data = {}
    for bprop in bprops:
        for var in (bprop, f'u_{bprop}'):
            obs_data[var] = np.array(group_table[f'{var}_obs'])

    return obs_data


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


def set_param_cols(group_table, batches, kgrid, params):
    """Sets model parameter columns in table for a single group of epochs

    parameters
    ----------
    group_table : pd.DataFrame
        table of a single group of epochs to add columns to
    batches : sequence
        list of batches that correspond to the group's epochs
    kgrid : Kgrid object
        contains the model paramters for each batch of runs
    params : sequence(str)
        parameters to extract from kgrid and add to the table
    """
    group_params = kgrid.get_params(run=group_table.group[0]
                                    ).set_index(['batch']).loc[batches]
    for var in params:
        group_table[var] = np.array(group_params[var])


def set_summ_cols(group_table, batches, kgrid, summ_list):
    """Sets summ value columns in table for a single group of epochs

    parameters
    ----------
    group_table : pd.DataFrame
        table of a single group of epochs to add columns to
    batches : sequence
        list of batches that correspond to the group's epochs
    kgrid : Kgrid object
        contains the model paramters for each batch of runs
    summ_list : sequence(str)
        names of quantities to extract from kgrid and add to the table
    """
    group_summ = kgrid.get_summ(run=group_table.group[0]
                                ).set_index(['batch']).loc[batches]

    for var in summ_list:
        u_var = f'u_{var}'
        group_table[var] = np.array(group_summ[var])
        group_table[u_var] = np.array(group_summ[u_var])


def set_rand_free_params(group_table, mcv, free_params):
    """Chooses random free parameters for a given group of epochs

    parameters
    ----------
    group_table : pd.DataFrame
        table of a single group of epochs to add columns to
    mcv : McmcVersion
        Object of mcmc constraints (for boundaries of free parameters)
    free_params : sequence(str)
        free parameters to generate
    """
    for var in free_params:
        rand_x = mcmc_tools.get_random_params(var, n_models=1, mv=mcv)[0]
        group_table[var] = rand_x


def set_observables(group_table, observables):
    """Calculate observables from model results and parameters

    parameters
    ----------
    group_table : pd.Dataframe
        table of a single group of epochs to add columns to
    observables : sequence(str)
        names of observables to calculate
    """
    local_keys = {'fper': 'accrate', 'u_fper': 'accrate'}
    redshift = group_table.redshift[0]
    d_b = group_table.d_b[0] * kpc_to_cm
    xi_ratio = group_table.xi_ratio[0]

    for var in observables:
        for key in (var, f'u_{var}'):
            local_key = local_keys.get(key, key)
            key_obs = f'{key}_obs'
            local = group_table[local_key]
            observed = None

            if 'rate' in key:
                observed = observe_rate(local, redshift=redshift)
            elif 'dt' in key:
                observed = observe_dt(local, redshift=redshift)
            elif 'peak' in key:
                d_star = d_b**2
                observed = observe_lum(local, d_star=d_star, redshift=redshift)
            elif 'fluence' in key:
                d_star = d_b**2
                observed = observe_fluence(local, d_star=d_star)
            elif 'fper' in key:
                d_star = xi_ratio * d_b**2
                acc_lum = get_lacc(local, redshift=redshift)
                observed = observe_lum(acc_lum, d_star=d_star, redshift=redshift)
                if key == 'u_fper':
                    observed *= 0.03    # hack fix uncertainty

            group_table[key_obs] = observed


def observe_rate(rate, redshift):
    """Returns observable burst rate (per day) from given local rate
    """
    return rate / redshift

def observe_dt(dt, redshift):
    """Returns observable burst rate (per day) from given local rate
        """
    return dt * redshift

def observe_lum(lum, d_star, redshift):
    """Returns observable flux from given local luminosity
    """
    return observe_fluence(lum, d_star) / redshift

def observe_fluence(fluence, d_star):
    """Returns observable fluence from given local burst energy
    """
    return fluence / (4*np.pi * d_star)

def get_lacc(accrate, redshift):
    """Returns accretion luminosity
    """
    phi = (redshift - 1) * c.value**2 / redshift  # gravitational potential
    return accrate * mdot_edd * phi

