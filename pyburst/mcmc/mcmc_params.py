import numpy as np
from astropy import units

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
    mass_nw, mass_gr = get_masses(chain, discard=discard, source=source, cap=cap,
                                  version=version, mass_nw=mass_nw, mass_gr=mass_gr)

    radius_gr = get_radius(mass_nw=mass_nw, mass_gr=mass_gr)
    return np.column_stack((radius_gr, mass_gr))


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
    xedd_ratio_chain = get_param_chain(chain, param='xedd_ratio', discard=discard,
                                       source=source, version=version, cap=cap)
    x_chain = get_param_chain(chain, param='x', discard=discard,
                              source=source, version=version, cap=cap)

    return xedd_ratio_chain * x_chain


def get_redshift_chain(chain, discard, source, version, cap=None, r_nw=10,
                       mass_nw=None, mass_gr=None):
    """Returns chain of redshift samples for given MCMC chain
    """
    xi_chain, redshift_chain = get_xi_redshift_chain(chain, discard=discard,
                                                     source=source, version=version,
                                                     cap=cap, r_nw=r_nw, mass_nw=mass_nw,
                                                     mass_gr=mass_gr)
    return redshift_chain


def get_xi_chain(chain, discard, source, version, cap=None, r_nw=10,
                 mass_nw=None, mass_gr=None):
    """Returns chain of redshift samples for given MCMC chain
    """
    xi_chain, redshift_chain = get_xi_redshift_chain(chain, discard=discard,
                                                     source=source, version=version,
                                                     cap=cap, r_nw=r_nw, mass_nw=mass_nw,
                                                     mass_gr=mass_gr)
    return xi_chain


def get_xi_redshift_chain(chain, discard, source, version, cap=None, r_nw=10,
                          mass_nw=None, mass_gr=None):
    """Returns chain of the xi and redshift
        Note: xi is the radius ratio (R_gr / R_nw), not anisotropy
    """
    mass_nw, mass_gr = get_masses(chain, discard=discard, source=source, cap=cap,
                                  version=version, mass_nw=mass_nw, mass_gr=mass_gr)

    mass_ratio = mass_gr / mass_nw
    xi_chain, redshift_chain = gravity.gr_corrections(r=r_nw, m=mass_nw, phi=mass_ratio)

    return xi_chain, redshift_chain-1


def get_masses(chain, discard, source, version, cap=None, mass_nw=None, mass_gr=None):
    """Returns chain(s) of m_nw, m_gr from given chain

        Note: if either mass is None, assumes it's in the chain, otherwise return as is
    """
    if mass_nw is None:
        mass_nw = get_param_chain(chain, param='m_nw', discard=discard,
                                  source=source, version=version, cap=cap)
    if mass_gr is None:
        mass_gr = get_param_chain(chain, param='m_gr', discard=discard,
                                  source=source, version=version, cap=cap)
    return mass_nw, mass_gr


def get_gravity_chain(chain, discard, source, version, cap=None, r_nw=10):
    """Returns flat chain of surface gravity (g) samples for a given MCMC chain
        Note: returns in units of 1e14 cm/s^2
    """
    mass_nw_chain = get_param_chain(chain, param='m_nw', discard=discard,
                                    source=source, version=version, cap=cap)
    g = gravity.get_acceleration_newtonian(r=r_nw, m=mass_nw_chain)
    return g.value/1e14


def get_gravitational_chain(chain, discard, source, version, cap=None, r_nw=10,
                            fixed_grav=False):
    """Returns chain of gravitational parameters: [M, R, g, 1+z], or [M, R, 1+z]
    """
    chains = []
    mass_nw, mass_gr = get_constant_masses(source, version)

    chains += [get_mass_radius_chain(chain=chain, discard=discard,
                                     source=source, version=version,
                                     cap=cap, mass_nw=mass_nw,
                                     mass_gr=mass_gr)]

    if not fixed_grav:
        chains += [get_gravity_chain(chain=chain, discard=discard,
                                     source=source, version=version,
                                     cap=cap, r_nw=r_nw)]

    chains += [get_redshift_chain(chain=chain, discard=discard,
                                  source=source, version=version,
                                  cap=cap, mass_nw=mass_nw,
                                  mass_gr=mass_gr)]

    return np.column_stack(chains)


def get_gr_mdot_chain(chain, discard, source, version, n_epochs, cap=None, r_nw=10,
                      mass_nw=None, mass_gr=None):
    """Returns chain of GR-corrected accretion rate
    """
    xi_chain, redshift_chain = get_xi_redshift_chain(chain, discard=discard,
                                                     source=source, version=version,
                                                     cap=cap, r_nw=r_nw, mass_nw=mass_nw,
                                                     mass_gr=mass_gr)

    mdot_chain = get_mdot_chain(chain, discard, cap=None, n_epochs=n_epochs)
    return mdot_chain * xi_chain[:, np.newaxis]


def get_mdot_chain(chain, discard, n_epochs, cap=None):
    """Returns chain of mdots
        Note: currently assumes mdots are the first [n_epochs] parameters in chain
    """
    # TODO: generalise to other sources, auto get mdot indexes
    flat_chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap, flatten=True)
    return flat_chain[:, :n_epochs]


def get_disc_chain(chain, discard, source, version, cap=None, disc_model='he16_a'):
    """Returns chain of [inc, xi_b, xi_p, d] derived with a disc model
    """
    # TODO: reuse xi chains more efficiently
    inc_chain = get_inclination_chain(chain=chain, discard=discard,
                                      source=source, version=version,
                                      cap=cap, disc_model=disc_model)

    d_chain = get_distance_chain(chain=chain, discard=discard,
                                 source=source, version=version,
                                 cap=cap, disc_model=disc_model)

    return np.column_stack([d_chain, inc_chain])


def get_inclination_chain(chain, discard, source, version, cap=None, disc_model='he16_a'):
    """returns inclination chain for given chain of xi_ratio, using simple disc model
    """
    xi_ratio_chain = get_param_chain(chain, param='xi_ratio', discard=discard,
                                     source=source, version=version, cap=cap)

    return anisotropy.inclination_ratio(xi_ratio_chain, model=disc_model).value


def get_anisotropy_chains(chain, discard, source, version, cap=None, disc_model='he16_a'):
    """Returns chains for xi_b and xi_p, derived with a disc model
    """
    inc_chain = get_inclination_chain(chain=chain, discard=discard,
                                      source=source, version=version,
                                      cap=cap, disc_model=disc_model)

    xi_b_chain, xi_p_chain = anisotropy.anisotropy(inc_chain * units.deg,
                                                   model=disc_model)
    return xi_b_chain, xi_p_chain


def get_distance_chain(chain, discard, source, version, cap=None, disc_model='he16_a'):
    """Returns chain of absolute distance, derived from a disc anisotropy model
    """
    d_b_chain = get_param_chain(chain, param='d_b', discard=discard,
                                source=source, version=version, cap=cap)

    xi_b_chain, _ = get_anisotropy_chains(chain=chain, discard=discard,
                                          source=source, version=version,
                                          cap=cap, disc_model=disc_model)
    return d_b_chain / np.sqrt(xi_b_chain)


def get_param_chain(chain, param, discard, source, version, cap=None):
    """Returns flat chain of given parameter, from given multi-param chain
    """
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    return chain[:, :, pkeys.index(param)].reshape(-1)


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

