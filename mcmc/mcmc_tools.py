import numpy as np
import os

# kepler_grids
from pygrids.misc import pyprint

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
    discard = {None: 0}.get(discard, discard)            # default to discard 0

    if discard >= cap:
        raise ValueError(f'discard ({discard}) must be less than cap ({cap})')

    for name, val in {'discard':discard, 'cap':cap}.items():
        if val > chain.shape[1]:
            raise ValueError(f'{name} is larger than the number of steps')
        if discard < 0:
            print("LTZ")
            raise ValueError(f"{name} ({val}) can't be negative")

    return chain[:, discard:cap, :]


def load_chain(source, version, n_steps, n_walkers, verbose=True):
    """Loads from file and returns np array of chain
    """
    filename = pyprint.get_mcmc_string(source=source, version=version,
                                       n_steps=n_steps, n_walkers=n_walkers,
                                       prefix='chain', extension='.npy')
    filepath = os.path.join(GRIDS_PATH, 'sources', source, 'mcmc', filename)
    pyprint.printv(f'Loading chain: {filepath}', verbose=verbose)
    return np.load(filepath)