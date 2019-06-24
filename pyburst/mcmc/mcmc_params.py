import numpy as np

# kepler_grids
from . import mcmc_versions
from . import mcmc_tools
from pyburst.physics import gravity

"""MCMC Parameters

Module for manipulating and calculating parameters derived from MCMC chains
"""

# TODO:
#       - get_inclination(xi_ratio)

def get_mass_radius_chain(chain, discard, source, version, cap=None):
    """Returns GR mass and radius given a chain containing gravity and redshift

    Returns ndarray of equivalent form to input chain (after slicing discard/cap)
    """
    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_walkers, n_steps, n_dimensions = chain.shape
    chain_flat = chain.reshape((-1, n_dimensions))
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    mass_nw = chain_flat[:, pkeys.index('m_nw')]
    mass_gr = chain_flat[:, pkeys.index('m_gr')]
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


def get_redshift(chain, discard, source, version, cap=None, r_nw=10):
    """Returns chain of redshift samples for given MCMC chain
    """
    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_walkers, n_steps, n_dimensions = chain.shape
    chain_flat = chain.reshape((-1, n_dimensions))
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    mass_nw = chain_flat[:, pkeys.index('m_nw')]
    mass_gr = chain_flat[:, pkeys.index('m_gr')]
    mass_ratio = mass_gr / mass_nw

    _, redshift = gravity.gr_corrections(r=r_nw, m=mass_nw, phi=mass_ratio)

    new_shape = (n_walkers, n_steps)
    redshift_reshape = redshift.reshape(new_shape)

    return redshift_reshape
