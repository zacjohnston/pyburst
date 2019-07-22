# standard
import numpy as np
import os
import sys
import time
import emcee
from scipy.optimize import fmin

# kepler_grids
from . import burstfit
from . import mcmc_versions
from . import mcmc_plot
from . import mcmc_tools
from pyburst.grids import grid_tools
from pyburst.misc.pyprint import check_params_length

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
                             re_interp=False, **kwargs)

    sampler = emcee.EnsembleSampler(n_walkers, n_dimensions, bfit.lhood,
                                    threads=n_threads)
    return sampler


def setup_positions(source, version, n_walkers, params0=None, mag=1e-3):
    """Sets up and returns posititons of walkers

    Parameters
    ----------
    source : str
    version : int
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
    result = None

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
        print(f'Compute time: {dtime:.1f} s ({dtime/3600:.2f} hr)')
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
        sampler = mcmc_tools.load_sampler_state(source=source, version=version,
                                     n_steps=n_steps, n_walkers=n_walkers)
    else:
        n_steps = sampler['iterations']

    return np.average(sampler['naccepted'] / n_steps)


def optimise(source, version, x0=None):
    """Optimise the starting position of the mcmc walkers using standard
        minimisation methods

    Parameters
    ----------
    params0 : ndarray
        Initial guess of parameters
    """
    bfit = burstfit.BurstFit(source=source, version=version, verbose=False,
                             lhood_factor=-1, zero_lhood=-1e9)
    if x0 is None:
        x0 = bfit.mcmc_version.initial_position

    return fmin(bfit.lhood, x0=x0, maxfun=10000)


def convert_params(params, source, version):
    """Converts params from dict to raw list format, and vice versa (ensures order)
    """
    pkeys = mcmc_versions.get_parameter(source, version, parameter='param_keys')
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
    else:
        raise TypeError('type(params) must be dict or array-like')

    return params_out
