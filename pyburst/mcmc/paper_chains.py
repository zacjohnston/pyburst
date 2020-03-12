from . import mcmc_tools, mcmc_params, mcmc_versions

source = 'grid5'


def load_chain(version):
    """Load chain used in paper
    """
    n_steps = {6: 20000}.get(version, 10000)
    compressed = {6: True}.get(version, False)

    return mcmc_tools.load_chain(source=source, version=version, n_steps=n_steps,
                                 n_walkers=1000, compressed=compressed)


def get_radius(flat):
    """Get flattened Radius chain from full chain
    """
    return mcmc_params.get_radius(mass_nw=flat['m_nw'], mass_gr=flat['m_gr'])


def get_flat(chain, version):
    """Return flattened arrys of each param in full chain
    """
    param_keys = mcmc_versions.get_parameter(source, version=version, parameter='param_keys')
    
    flat = {}
    for key in param_keys:
        flat[key] = mcmc_params.get_param_chain(chain, param=key, discard=0,
                                                source=source, version=version)
    return flat


def get_gravity(chain, version):
    """Get flattened surface gravity (g, 10^14 cm/s^2) from full chain
    """
    return mcmc_params.get_gravity_chain(chain, discard=0, source=source, version=version)


def get_redshift(chain, flat, version):
    """Get redshift (1+z) from full chain
    """
    return mcmc_params.get_redshift_chain(chain, discard=0, source=source, version=version,
                                          mass_nw=flat['m_nw'],
                                          mass_gr=flat['m_gr'])
