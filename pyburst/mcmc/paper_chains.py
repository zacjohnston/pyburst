import numpy as np
import os
import gzip

from . import mcmc_tools, mcmc_params, mcmc_versions
from pyburst.physics import gravity

source = 'grid5'
path = '/home/zac/projects/papers/mcmc/data'


# ===============================================================
#                      Baseline chains
# ===============================================================
def pipeline(version, compressed=True):
    """Extract, reorder, and save chain
    """
    chain = load_chain(version)
    chain = replace_mass_with_grav(chain, version=version)
    chain = reorder_chain(chain, version=version)
    save_chain(chain, version=version, compressed=compressed)


def load_chain(version):
    """Load chain used in paper
    """
    n_steps = {6: 20000}.get(version, 10000)
    compressed = {6: True}.get(version, False)

    return mcmc_tools.load_chain(source=source, version=version, n_steps=n_steps,
                                 n_walkers=1000, compressed=compressed)


def replace_mass_with_grav(chain, version):
    """Convert m_nw from main chain to g (1e14 cm/s^2)
    """
    print('Converting m_nw to g')
    mv = get_mcmc_version(version)
    m_nw_idx = mv.param_keys.index('m_nw')

    grav_factor = gravity.get_acceleration_newtonian(r=10, m=1.0)
    grav_factor = grav_factor.value / 1e14

    chain_out = chain.copy()
    chain_out[:, :, m_nw_idx] *= grav_factor

    return chain_out


def get_mcmc_version(version):
    return mcmc_versions.McmcVersion(source=source, version=version)


def unflatten(flat_chain):
    """Unflatten chain into shape [walkers, steps, params]

    flat_chain: shape [samples, params]
    """
    n_params = flat_chain.shape[-1]
    return flat_chain.reshape((1000, -1, n_params))


def reorder_chain(chain, version):
    """Reorder param columns of chain, to be consistent with paper
        i.e. move qb_i before 'x'
    """
    print('Reordering chain params')
    mv = get_mcmc_version(version=version)
    params = mv.param_keys

    if version == 6:
        idx_map = {'qb1': 3,
                   'qb2': 4,
                   'qb3': 5,
                   'x': 6,
                   'z': 7}
    else:
        idx_map = {'qb1': 1,
                   'x': 2,
                   'z': 3}

    chain_out = chain.copy()
    for param, new_idx in idx_map.items():
        old_idx = params.index(param)
        chain_out[:, :, new_idx] = chain[:, :, old_idx]

    return chain_out


# ===============================================================
#                      Load/Save
# ===============================================================
def save_chain(chain, version, compressed=True):
    """Save chain to file
    """
    filepath = chain_filepath(version, compressed=compressed)
    print(f'Saving chain: {filepath}')

    if compressed:
        with gzip.GzipFile(filepath, 'w') as f:
            np.save(f, chain)
    else:
        np.save(filepath, chain)


def load_chain2(version, compressed=False):
    """Load saved chain from file
    """
    filepath = chain_filepath(version, compressed=compressed)
    print(f'Loading chain: {filepath}')

    if compressed:
        f = gzip.GzipFile(filepath, 'r')
        chain = np.load(f)
    else:
        chain = np.load(filepath)

    return chain

    
def chain_filepath(version, compressed):
    chain_id = version - 5
    filename = f'mcmc_chain_{chain_id}.npy'

    if compressed:
        filename += '.gz'

    return os.path.join(path, filename)


# ===============================================================
#                      Derived chains
# ===============================================================
def assemble_full_flat(version, discard):
    """Asseble full flat chains with additional params
    """
    chain = load_chain(version)
    flat = get_flat(chain, version, discard=discard)
    flat['R'] = get_radius(flat)
    flat['g'] = get_gravity(chain, version, discard=discard)
    flat['redshift'] = get_redshift(chain, flat, version, discard=discard)
    return flat


def setup_chainconsumer(flat):
    params = list(flat.keys())
    flat_chain = stack_flat(flat)
    return mcmc_tools.setup_custom_chainconsumer(flat_chain, parameters=params)


def stack_flat(flat):
    chains = []
    for key, chain in flat.items():
        chains += [chain]

    return np.column_stack(chains)


def get_radius(flat):
    """Get flattened Radius chain from full chain
    """
    print('getting radius')
    return mcmc_params.get_radius(mass_nw=flat['m_nw'], mass_gr=flat['m_gr'])


def get_flat(chain, version, discard):
    """Return flattened arrys of each param in full chain
    """
    print('flattening chain')
    param_keys = mcmc_versions.get_parameter(source, version=version, parameter='param_keys')

    flat = {}
    for key in param_keys:
        flat[key] = mcmc_params.get_param_chain(chain, param=key, discard=discard,
                                                source=source, version=version)
    return flat


def get_gravity(chain, version, discard):
    """Get flattened surface gravity (g, 10^14 cm/s^2) from full chain
    """
    print('getting gravity')
    return mcmc_params.get_gravity_chain(chain, discard=discard, source=source, version=version)


def get_redshift(chain, flat, version, discard):
    """Get redshift (z) from full chain
    """
    print('getting redshift')
    return mcmc_params.get_redshift_chain(chain, discard=discard, source=source,
                                          version=version, mass_nw=flat['m_nw'],
                                          mass_gr=flat['m_gr'])
