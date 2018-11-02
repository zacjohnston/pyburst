import numpy as np
import os
import pickle

# kepler_grids
from pygrids.misc import pyprint
from pygrids.grids import grid_strings
from . import mcmc_versions

GRIDS_PATH = os.environ['KEPLER_GRIDS']


def slice_chain(chain, discard=None, cap=None):
    """Return a subset of a chain

    parameters
    ----------
    discard : int
        number of steps to discard (from start)
    cap : int, optional
         step number of endpoint
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

    return chain[:, discard:cap, :]


def load_chain(source, version, n_steps, n_walkers, verbose=True):
    """Loads from file and returns np array of chain
    """
    filename = get_mcmc_string(source=source, version=version,
                               n_steps=n_steps, n_walkers=n_walkers,
                               prefix='chain', extension='.npy')

    mcmc_path = get_mcmc_path(source)
    filepath = os.path.join(mcmc_path, filename)
    pyprint.printv(f'Loading chain: {filepath}', verbose=verbose)

    return np.load(filepath)


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

    bounds = mv.prior_bounds[idx]
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
