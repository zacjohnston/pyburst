# standard
import numpy as np
import os
import sys
import pickle
import time
import emcee
from scipy.optimize import minimize

# kepler_grids
from ..grids import grid_tools
from . import burstfit
from . import mcmc_versions
from . import mcmc_plotting
from . import mcmc_tools
from ..misc.pyprint import check_params_length

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


def setup_sampler(source, version, pos=None, n_walkers=None, n_threads=1,
                  **kwargs):
    """Initialises and returns EnsembleSampler object

    NOTE: Only uses pos to get n_walkers and n_dimensions
    """
    if pos is None:
        if n_walkers is None:
            print('ERROR: must provide either pos or n_walkers')
        pos = setup_positions(source=source, version=version, n_walkers=n_walkers)

    n_walkers = len(pos)
    n_dimensions = len(pos[0])

    bfit = burstfit.BurstFit(source=source, version=version, verbose=False,
                             **kwargs)

    sampler = emcee.EnsembleSampler(n_walkers, n_dimensions, bfit.lhood,
                                    threads=n_threads)
    return sampler


def setup_positions(source, version, n_walkers, params0=None, mag=1e-3):
    """Sets up and returns posititons of walkers

    Parameters
    ----------
    n_walkers: int, number of mcmc walkers to use
    params0: array, initial guess (mdot1, x, z, qb, g, redshift, d, inc)
    mag: flt, magnitude of random seeds to use for initial mcmc 'ball'
    """
    if type(params0) == type(None):
        mcmc_version = mcmc_versions.McmcVersion(source=source, version=version)
        params0 = mcmc_version.initial_position

    n_dimensions = len(params0)
    pos = [params0 * (1 + mag * np.random.randn(n_dimensions)) for i in range(n_walkers)]
    return np.array(pos)


def run_sampler(sampler, pos, n_steps, verbose=True):
    """Runs emcee chain for n_steps
    """
    t0 = time.time()

    for i, result in enumerate(sampler.sample(pos, iterations=n_steps)):
        if verbose:
            progress = (float(i + 1) / n_steps) * 100
            sys.stdout.write(f"\r{progress:.1f}%")
    sys.stdout.write("\n")

    t1 = time.time()
    dtime = t1 - t0
    time_per_step = dtime / n_steps

    n_walkers = pos.shape[0]
    n_samples = n_walkers * n_steps
    time_per_sample = dtime / n_samples

    if verbose:
        print(f'Compute time: {dtime:.1f} s')
        print(f'Time per step: {time_per_step:.1f} s')
        print(f'Time per sample: {time_per_sample:.4f} s')
    return result


def get_acceptance_fraction(source=None, version=None, n_walkers=None,
                            n_steps=None, sampler=None):
    """Returns acceptance fraction averaged over all walkers for all steps

    Must provide either:
        1. a sampler object (as returned from load_sampler_state)
        2. source, version, n_walkers, and n_steps
    """
    if sampler is None:
        if None in (source, version, n_walkers, n_steps):
            raise ValueError('Must provide source, version, n_steps, '
                             + 'and n_walkers (or directly provide sampler object)')
        sampler = load_sampler_state(source=source, version=version,
                                     n_steps=n_steps, n_walkers=n_walkers)
    else:
        n_steps = sampler['iterations']

    return np.average(sampler['naccepted'] / n_steps)


def get_max_lhood(source, version, n_walkers, n_steps,
                  verbose=True, plot=True):
    """Returns the point with the highest likelihood
    """
    sampler_state = load_sampler_state(source=source, version=version,
                                       n_steps=n_steps, n_walkers=n_walkers)

    chain = sampler_state['_chain']
    lnprob = sampler_state['_lnprob']

    max_idx = np.argmax(lnprob)
    max_lhood = lnprob.flatten()[max_idx]

    n_dimensions = chain.shape[2]
    flat_chain = chain.reshape((-1, n_dimensions))
    max_params = flat_chain[max_idx]

    if plot:
        bfit = burstfit.BurstFit(source=source, version=version, verbose=False)
        bfit.lhood(max_params, plot=True)

    if verbose:
        print(f'max_lhood = {max_lhood:.2f}')
        print('-' * 30)
        print('Best params:')
        mcmc_tools.print_params(max_params, source=source, version=version)

    return max_params


def optimise(source, version, params0):
    """Optimise the starting position of the mcmc walkers using standard
        minimisation methods

    Parameters
    ----------
    params0 : ndarray
        Initial guess of parameters
    """
    bfit = burstfit.BurstFit(source=source, version=version, verbose=False,
                             lhood_factor=-1)
    bnds = ((0.09, 0.23), (0.61, 0.79), (0.00251, 0.0174), (0.0251, 0.124),
            (0.81 / 1.4, 3.19 / 1.4), (1.01, 2.), (0.1, None), (0.1, 89.9))

    return minimize(bfit.lhood, x0=params0, bounds=bnds)


def save_sampler_state(sampler, source, version, n_steps, n_walkers):
    """Saves sampler state as dict
    """
    sampler_state = get_sampler_state(sampler=sampler)
    chain_id = mcmc_tools.get_mcmc_string(source=source, version=version,
                                          n_steps=n_steps, n_walkers=n_walkers)

    mcmc_path = get_mcmc_path(source)
    filename = f'sampler_{chain_id}.p'
    filepath = os.path.join(mcmc_path, filename)

    print(f'Saving: {filepath}')
    pickle.dump(sampler_state, open(filepath, 'wb'))


def load_sampler_state(source, version, n_steps, n_walkers):
    """Loads sampler state from file
    """
    chain_id = mcmc_tools.get_mcmc_string(source=source, version=version,
                                          n_steps=n_steps, n_walkers=n_walkers)

    filename = f'sampler_{chain_id}.p'
    mcmc_path = get_mcmc_path(source)
    filepath = os.path.join(mcmc_path, filename)
    sampler_state = pickle.load(open(filepath, 'rb'))

    return sampler_state


def get_sampler_state(sampler):
    """Returns sampler as a dictionary so its properties can be saved
    """
    sampler_dict = sampler.__dict__.copy()
    del sampler_dict['pool']
    return sampler_dict


def get_mcmc_path(source):
    return os.path.join(GRIDS_PATH, 'sources', source, 'mcmc')


def convert_params(params, source, version):
    """Converts params from dict to raw list format, and vice versa (ensures order)
    """
    pkeys = mcmc_versions.get_param_keys(source=source, version=version)
    check_params_length(params=params, n=len(pkeys))
    ptype = type(params)

    if ptype == dict:
        params_out = []
        for key in pkeys:
            params_out += [params[key]]

    elif (ptype == list) or (ptype == tuple) or (ptype == np.ndarray):
        params_out = {}
        for i, key in enumerate(pkeys):
            params_out[key] = params[i]

    return params_out


def save_all_plots(source, versions, n_steps, n_walkers,
                   discard, cap=None, display=False):
    """Saves all primary plots for given source/version

    version : int|list
        if type(list), iterates over each version
    """
    versions = grid_tools.ensure_np_list(versions)

    for version in versions:
        pkeys = mcmc_versions.get_param_keys(source=source, version=version)

        chain = mcmc_tools.load_chain(source, version=version, n_steps=n_steps,
                                      n_walkers=n_walkers)

        mcmc_plotting.plot_posteriors(chain, source=source, version=version,
                                      discard=discard, cap=cap, save=True, display=display)

        mcmc_plotting.plot_mass_radius(chain, source=source, version=version,
                                       discard=discard, cap=cap, save=True, display=display)

        mcmc_plotting.plot_walkers(chain, source=source, version=version,
                                   params=pkeys[:5], save=True, display=display)
