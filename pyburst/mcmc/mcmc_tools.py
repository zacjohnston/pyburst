import numpy as np
import os
import sys
import pickle
import gzip
import chainconsumer

# kepler_grids
from pyburst.misc import pyprint
from pyburst.grids import grid_strings
from pyburst.plotting import plot_tools
from . import mcmc_versions, mcmc_params, burstfit


GRIDS_PATH = os.environ['KEPLER_GRIDS']


def slice_chain(chain, discard=None, cap=None, flatten=False):
    """Return a subset of a chain

    parameters
    ----------
    chain : nparray
    discard : int
        number of steps to discard (from start)
    cap : int, optional
         step number of endpoint
    flatten : bool
        return chain with the steps and walkers dimensions flattened
    """
    cap = {None: chain.shape[1]}.get(cap, cap)  # default to final step
    discard = {None: 0}.get(discard, discard)  # default to discard 0

    if discard >= cap:
        raise ValueError(f'discard ({discard}) must be less than cap ({cap})')

    for name, val in {'discard': discard, 'cap': cap}.items():
        if val > chain.shape[1]:
            raise ValueError(f'{name} is larger than the number of steps')
        if discard < 0:
            print("LTZ")
            raise ValueError(f"{name} ({val}) can't be negative")
    if flatten:
        n_dimensions = chain.shape[2]
        return chain[:, discard:cap, :].reshape((-1, n_dimensions))
    else:
        return chain[:, discard:cap, :]


def load_chain(source, version, n_steps, n_walkers, compressed=False, verbose=True):
    """Loads from file and returns np array of chain
    """
    extension = {True: '.npy.gz', False: '.npy'}[compressed]
    filename = get_mcmc_string(source=source, version=version,
                               n_steps=n_steps, n_walkers=n_walkers,
                               prefix='chain', extension=extension)

    mcmc_path = get_mcmc_path(source)
    filepath = os.path.join(mcmc_path, filename)
    pyprint.printv(f'Loading chain: {filepath}', verbose=verbose)

    if compressed:
        f = gzip.GzipFile(filepath, 'r')
        chain = np.load(f)
    else:
        chain = np.load(filepath)

    return chain


def save_compressed_chain(chain, source, version, verbose=True):
    """Saves a chain as a compressed zip
    """
    n_walkers, n_steps, n_dim = chain.shape
    filename = get_mcmc_string(source=source, version=version,
                               n_steps=n_steps, n_walkers=n_walkers,
                               prefix='chain', extension='.npy.gz')
    mcmc_path = get_mcmc_path(source)
    filepath = os.path.join(mcmc_path, filename)
    pyprint.printv(f'Saving compressed chain: {filepath}', verbose=verbose)

    with gzip.GzipFile(filepath, 'w') as f:
        np.save(f, chain)


def setup_chainconsumer(chain, discard, cap=None, param_labels=None, cloud=False,
                        source=None, version=None, sigmas=np.linspace(0, 2, 5),
                        summary=False):
    """Return ChainConsumer object set up with given chain and pkeys
    """
    if param_labels is None:
        if (source is None) or (version is None):
            raise ValueError('If param_labels not provided, must give source, version')
        param_keys = mcmc_versions.get_parameter(source, version, 'param_keys')
        param_labels = plot_tools.convert_mcmc_labels(param_keys)

    n_walkers = chain.shape[0]
    chain_flat = slice_chain(chain, discard=discard, cap=cap, flatten=True)

    cc = chainconsumer.ChainConsumer()
    cc.add_chain(chain_flat, parameters=param_labels, walkers=n_walkers)
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0, summary=summary)
    return cc


def setup_custom_chainconsumer(flat_chain, parameters, cloud=False, unit_labels=True,
                               sigmas=np.linspace(0, 2, 5), summary=False, fontsize=12):
    """Returns ChainConsumer, with derived parameters

        Note: chain must already be flattened and  discarded/capped
    """
    param_labels = plot_tools.convert_mcmc_labels(parameters, unit_labels=unit_labels)

    cc = chainconsumer.ChainConsumer()
    cc.add_chain(flat_chain, parameters=param_labels)
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0, summary=summary,
                 label_font_size=fontsize, tick_font_size=fontsize-2)
    return cc


def setup_master_chainconsumer(source, master_version, epoch_versions, n_steps, discard,
                               n_walkers=1000, epoch_discard=None, epoch_n_steps=None,
                               epoch_n_walkers=None, cap=None, sigmas=None, cloud=None,
                               compressed=False):
    """Setup multiple MCMC chains, including multi-epoch and single-epochs
    """
    if epoch_discard is None:
        epoch_discard = discard
    if epoch_n_steps is None:
        epoch_n_steps = n_steps
    if epoch_n_walkers is None:
        epoch_n_walkers = n_walkers

    cc = setup_epochs_chainconsumer(source, versions=epoch_versions, n_steps=epoch_n_steps,
                                    discard=epoch_discard, n_walkers=epoch_n_walkers,
                                    cap=cap, sigmas=sigmas, cloud=cloud, compressed=False)

    # ===== Setup master chain =====
    master_mc_v = mcmc_versions.McmcVersion(source, version=master_version)

    master_chain = load_chain(source, version=master_version, n_steps=n_steps,
                              n_walkers=n_walkers, compressed=compressed)
    master_chain_sliced = slice_chain(master_chain, discard=discard, cap=cap,
                                      flatten=True)

    formatted_params = plot_tools.convert_mcmc_labels(master_mc_v.param_keys)
    cc.add_chain(master_chain_sliced, parameters=formatted_params, color='black',
                 name='Multi-epoch')
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=False)

    return cc


def setup_epochs_chainconsumer(source, versions, n_steps, discard, n_walkers=1000,
                               cap=None, sigmas=None, cloud=None, compressed=False):
    """Setup multiple MCMC chains fit to individual epochs

    chains : [n_epochs]
        list of raw numpy chains
    param_keys : [n_epochs]
        list of parameters for each epoch chain
    discard : int
    cap : int (optional)
    sigmas : [] (optional)
    cloud : bool (optional)
    """
    param_keys = load_multi_param_keys(source, versions=versions)
    chains = load_multi_chains(source, versions=versions, n_steps=n_steps,
                               n_walkers=n_walkers, compressed=compressed)
    chains_flat = []
    for chain in chains:
        sliced_flat = slice_chain(chain, discard=discard, cap=cap, flatten=True)
        chains_flat += [sliced_flat]

    cc = chainconsumer.ChainConsumer()

    for i, chain_flat in enumerate(chains_flat):
        epoch = mcmc_versions.get_parameter(source, version=versions[i], parameter='epoch')
        param_labels = plot_tools.convert_mcmc_labels(param_keys[i])
        cc.add_chain(chain_flat, parameters=param_labels, name=str(epoch))

    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0)
    return cc


def setup_gravitational_chainconsumer(chain, discard, source, version, cap=None,
                                      r_nw=10, summary=False, unit_labels=True,
                                      sigmas=np.linspace(0, 2, 5)):
    """Returns ChainConsumer of gravitational parameters
    """
    grav_chain = mcmc_params.get_gravitational_chain(chain=chain, discard=discard,
                                                     source=source, version=version,
                                                     cap=cap, r_nw=r_nw)

    cc = setup_custom_chainconsumer(grav_chain, parameters=['R', 'M', 'g', '1+z'],
                                    sigmas=sigmas, summary=summary,
                                    unit_labels=unit_labels)
    return cc


def get_summary(chain, discard, source, version, cap=None):
    """Return summary values from MCMC chain (mean, uncertainties)
    """
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
    n_dimensions = chain.shape[2]
    summary = np.full((n_dimensions, 3), np.nan)
    cc = setup_chainconsumer(chain=chain, param_labels=pkeys, discard=discard, cap=cap)
    summary_dict = cc.analysis.get_summary()

    for i, key in enumerate(pkeys):
        summary[i, :] = summary_dict[key]
    return summary


def load_multi_chains(source, versions, n_steps, n_walkers=1000, compressed=False):
    """Loads multiple chains of MCMC runs

    parameters
    ----------
    source : str
    versions : [int]
        list of each mcmc chain version
    n_steps : [int], int
        number of steps for each chain (if scalar, assume all chains identical)
    n_walkers : [int], int (optional)
        number of walkers for each chain (if scalar, assume all chains identical)
    compressed : bool (optional)
    """
    if type(versions) not in (np.ndarray, list, tuple):
        raise TypeError("'versions' must be array-like")

    n_chains = len(versions)
    n_steps = check_array(n_steps, n=n_chains)
    n_walkers = check_array(n_walkers, n=n_chains)
    chains = []

    for i in range(n_chains):
        chains += [load_chain(source, n_walkers=n_walkers[i],
                              n_steps=n_steps[i], version=versions[i],
                              compressed=compressed)]
    return chains


def load_multi_param_keys(source, versions):
    """Returns list of epoch-specific param_keys for multiple chains
    """
    if type(versions) not in (np.ndarray, list, tuple):
        raise TypeError("'versions' must be array-like")

    n_chains = len(versions)
    param_keys = []

    for i in range(n_chains):
        param_keys += [mcmc_params.epoch_param_keys(source, version=versions[i])]

    return param_keys


def get_mcmc_string(source, version, n_walkers=None, n_steps=None,
                    n_threads=None, prefix=None, label=None, extension=''):
    """Return standardised string for mcmc labelling
    """

    def get_segment(var, tag='', delimiter_front='_', delimiter_back=''):
        """Return str segment, if provided
        """
        return {None: ''}.get(var, f'{delimiter_front}{tag}{var}{delimiter_back}')

    # TODO: Probably a smarter/more robust way to do this
    prefix_str = get_segment(prefix, delimiter_front='', delimiter_back='_')
    walker_str = get_segment(n_walkers, tag='W')
    thread_str = get_segment(n_threads, tag='T')
    step_str = get_segment(n_steps, tag='S')
    label_str = get_segment(label)

    return (f'{prefix_str}{source}_V{version}{walker_str}'
            + f'{thread_str}{step_str}{label_str}{extension}')


def get_max_lhood_params(source, version, n_walkers, n_steps, verbose=True,
                         return_lhood=False):
    """Returns the point with the highest likelihood
    """
    sampler_state = load_sampler_state(source=source, version=version,
                                       n_steps=n_steps, n_walkers=n_walkers)

    chain = sampler_state['_chain']
    lnprob = sampler_state['_lnprob']

    max_idx = np.argmax(lnprob)
    max_lhood = lnprob.flatten()[max_idx]

    n_dimensions = sampler_state['dim']
    flat_chain = chain.reshape((-1, n_dimensions))
    max_params = flat_chain[max_idx]

    if verbose:
        print(f'max_lhood = {max_lhood:.2f}')
        print('-' * 30)
        print('Best params:')
        print_params(max_params, source=source, version=version)

    if return_lhood:
        return max_params, max_lhood
    else:
        return max_params


def get_random_sample(chain, n, discard=None, cap=None):
    """Returns random sample of params from given MCMC chain
    """
    chain = slice_chain(chain, discard=discard, cap=cap)
    n_dim = chain.shape[-1]
    flat_chain = chain.reshape((-1, n_dim))
    idxs = np.random.randint(len(flat_chain), size=n)

    return flat_chain[idxs], idxs


def get_random_params(key, n_models, mv):
    """Returns random sample of length 'n_models', within mcmc boundaries
    """
    idx = mv.param_keys.index(key)

    bounds = mv.grid_bounds[idx]
    range_ = np.diff(bounds)
    rand = np.random.random_sample(n_models)
    return rand * range_ + bounds[0]


def save_sampler_state(sampler, source, version, n_steps, n_walkers):
    """Saves sampler state as dict
    """
    sampler_state = get_sampler_state(sampler=sampler)
    chain_id = get_mcmc_string(source=source, version=version,
                               n_steps=n_steps, n_walkers=n_walkers)

    mcmc_path = get_mcmc_path(source)
    filename = f'sampler_{chain_id}.p'
    filepath = os.path.join(mcmc_path, filename)

    print(f'Saving: {filepath}')
    pickle.dump(sampler_state, open(filepath, 'wb'))


def load_sampler_state(source, version, n_steps, n_walkers):
    """Loads sampler state from file
    """
    chain_id = get_mcmc_string(source=source, version=version,
                               n_steps=n_steps, n_walkers=n_walkers)

    filename = f'sampler_{chain_id}.p'
    mcmc_path = get_mcmc_path(source)
    filepath = os.path.join(mcmc_path, filename)
    sampler_state = pickle.load(open(filepath, 'rb'))

    return sampler_state


def bprop_sample(chain, n, source, version, discard, cap=None):
    """Get sample of burst properties from a chain
    """
    bfit = burstfit.BurstFit(source=source, version=version)
    p_sample, i_sample = get_random_sample(chain, n=n, discard=discard, cap=cap)
    bprops_full = np.full([bfit.n_epochs, bfit.n_bprops, n], np.nan)

    for i, x in enumerate(p_sample):
        sys.stdout.write(f'\r{i+1}/{n}')
        bprops_full[:, :, i] = bfit.bprop_sample(x)[:, ::2]

    sys.stdout.write('\n')
    return bprops_full


def get_sampler_state(sampler):
    """Returns sampler as a dictionary so its properties can be saved
    """
    sampler_dict = sampler.__dict__.copy()
    del sampler_dict['pool']
    return sampler_dict


def get_mcmc_path(source):
    source = grid_strings.check_synth_source(source)
    return os.path.join(GRIDS_PATH, 'sources', source, 'mcmc')


def print_params(params, source, version):
    """Pretty print parameters
    """
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
    for i, p in enumerate(params):
        print(f'{pkeys[i]:8}    {p:.4f}')


def check_array(x, n):
    """Checks if x is array-like, and returns one of length 'n' if not
    """
    if type(x) in [np.ndarray, list, tuple]:
        if len(x) == n:
            return x
        else:
            raise ValueError(f'A supplied array is not of length n={n}')
    else:
        return np.full(n, x)


# ===================================================
# The following functions adapted from: https://dfm.io/posts/autocorr/
# ===================================================
def autocorrelation(y, c=5.0):
    """Estimates integrated autocorrelation time
    """
    f = np.zeros(y.shape[1])

    # Calculate for each walker, then average
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)

    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_func_1d(x, norm=True):
    """Returns autocorrelation function for given time series
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n

    if norm:
        acf /= acf[0]
    return acf


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# ===================================================
