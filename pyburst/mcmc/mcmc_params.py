import numpy as np

# kepler_grids
from . import mcmc_versions
from . import mcmc_tools
from pyburst.physics import gravity
from pyburst.observations import obs_tools

# Concord
try:
    import anisotropy
except ModuleNotFoundError:
    print("pyburst/MCMC: Concord not installed, some functionality won't be available")

"""MCMC Parameters

Module for manipulating and calculating parameters derived from MCMC chains
"""


def get_constant_masses(source, version):
    """Returns constant values for mass_nw, mass_gr (if they exist, else return None)
    """
    constants = mcmc_versions.get_parameter_dict(source, version, 'constants')
    mass_nw = constants.get('m_nw', None)
    mass_gr = constants.get('m_gr', None)
    return mass_nw, mass_gr


def get_mass_from_chain(chain_flat, mass_nw, mass_gr, source, version):
    """Returns Newtonian and/or GR mass from MCMC chain
    """
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    if mass_nw is None:
        mass_nw = chain_flat[:, pkeys.index('m_nw')]
    if mass_gr is None:
        mass_gr = chain_flat[:, pkeys.index('m_gr')]

    return mass_nw, mass_gr


def get_mass_radius_chain(chain, discard, source, version, cap=None,
                          mass_nw=None, mass_gr=None):
    """Returns GR mass and radius given a chain containing gravity and redshift

    Returns ndarray of equivalent form to input chain (after slicing discard/cap)

    parameters
    ----------
    chain : np.array
    discard : int
    source : str
    version : int
    cap : int
    mass_nw : flt (optional)
        specify a constant mass_nw. If None, assume it is in chain
    mass_gr : flt (optional)
        specify a constant mass_gr. If None, assume it is in chain
    """
    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_walkers, n_steps, n_dimensions = chain.shape
    chain_flat = chain.reshape((-1, n_dimensions))

    mass_nw, mass_gr = get_mass_from_chain(chain_flat, mass_nw=mass_nw, mass_gr=mass_gr,
                                           source=source, version=version)
    radius_gr = get_radius(mass_nw=mass_nw, mass_gr=mass_gr)

    new_shape = (n_walkers, n_steps)
    mass_reshape = mass_gr.reshape(new_shape)
    radius_reshape = radius_gr.reshape(new_shape)

    return np.dstack((radius_reshape, mass_reshape))


def get_radius(mass_nw, mass_gr):
    """Returns GR radius for the given Newtonian mass and GR mass
    """
    ref_radius = 10
    m_ratio = mass_gr / mass_nw

    xi = gravity.gr_corrections(r=ref_radius, m=mass_nw, phi=m_ratio)[0]
    radius_gr = ref_radius * xi
    return radius_gr


def get_xedd_chain(chain, discard, source, version, cap=None):
    """Returns chain of X_edd, from a given chain with parameters xedd_ratio and x
    """
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_walkers, n_steps, n_dimensions = chain.shape
    chain_flat = chain.reshape((-1, n_dimensions))

    xedd_flat = chain_flat[:, pkeys.index('xedd_ratio')] * chain_flat[:, pkeys.index('x')]

    new_shape = (n_walkers, n_steps)
    xedd_chain = xedd_flat.reshape(new_shape)

    return xedd_chain


def get_redshift_chain(chain, discard, source, version, cap=None, r_nw=10,
                       mass_nw=None, mass_gr=None):
    """Returns chain of redshift samples for given MCMC chain
    """
    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_walkers, n_steps, n_dimensions = chain.shape
    chain_flat = chain.reshape((-1, n_dimensions))

    mass_nw, mass_gr = get_mass_from_chain(chain_flat, mass_nw=mass_nw, mass_gr=mass_gr,
                                           source=source, version=version)

    mass_ratio = mass_gr / mass_nw
    _, redshift = gravity.gr_corrections(r=r_nw, m=mass_nw, phi=mass_ratio)

    return redshift.reshape((n_walkers, n_steps))


def get_gravity_chain(chain, discard, source, version, cap=None, r_nw=10):
    """Returns flat chain of surface gravity (g) samples for a given MCMC chain
    """
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_walkers, n_steps, n_dimensions = chain.shape
    chain_flat = chain.reshape((-1, n_dimensions))

    mass_nw = chain_flat[:, pkeys.index('m_nw')]

    g = gravity.get_acceleration_newtonian(r=r_nw, m=mass_nw)

    return g.value


def get_inclination_chain(chain, discard, source, version, cap=None, disc_model='he16_a'):
    """returns inclination chain for given chain of xi_ratio, using simple disc model
    """
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_walkers, n_steps, n_dimensions = chain.shape
    chain_flat = chain.reshape((-1, n_dimensions))

    xi_ratio = chain_flat[:, pkeys.index('xi_ratio')]
    inc = anisotropy.inclination_ratio(xi_ratio, model=disc_model)

    return inc.reshape((n_walkers, n_steps))


def epoch_param_keys(source, version):
    """Returns param_keys corrected for epoch-specific parameters
    """
    mcmc_version = mcmc_versions.McmcVersion(source, version=version)
    epoch = mcmc_version.epoch
    param_keys = list(mcmc_version.param_keys)

    if epoch is None:   # if not a single-epoch chain
        return param_keys

    system_table = obs_tools.load_summary(mcmc_version.system)
    epochs = list(system_table.epoch)
    epoch_n = epochs.index(epoch) + 1

    for i, param in enumerate(param_keys):
        split = param.split('1')[0]  # assumes default index is 1

        if split in mcmc_version.epoch_unique:
            param_keys[i] = split + str(epoch_n)

    return param_keys

