from . import mcmc_tools, mcmc_params

source = 'grid5'


def load_chain(version):
    """Load chain used in paper
    """
    n_steps = {6: 20000}.get(version, 10000)
    compressed = {6: True}.get(version, False)

    return mcmc_tools.load_chain(source=source, version=version, n_steps=n_steps,
                                 n_walkers=1000, compressed=compressed)


def get_radius(flat_masses):
    """Get flattened Radius chain from full chain
    """
    return mcmc_params.get_radius(mass_nw=flat_masses['m_nw'], mass_gr=flat_masses['m_gr'])


def get_flat_masses(chain, version):
    flat_masses = {}
    for key in ['m_nw', 'm_gr']:
        flat_masses[key] = mcmc_params.get_param_chain(chain, param=key, discard=0,
                                                       source=source, version=version)
    return flat_masses


def get_gravity(chain, version):
    """Get flattened surface gravity (g, 10^14 cm/s^2) from full chain
    """
    return mcmc_params.get_gravity_chain(chain, discard=0, source=source, version=version)


