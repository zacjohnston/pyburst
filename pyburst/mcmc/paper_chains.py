from . import mcmc_tools, mcmc_params


def load_chain(version):
    """Load chain used in paper
    """
    n_steps = {6: 20000}.get(version, 10000)
    compressed = {6: True}.get(version, False)

    return mcmc_tools.load_chain(source='grid5', version=version, n_steps=n_steps,
                                 n_walkers=1000, compressed=compressed)


def get_radius(chain, version):
    """Get flattened Radius chain from full chain
    """
    flat = {}
    for key in ['m_nw', 'm_gr']:
        flat[key] = mcmc_params.get_param_chain(chain, param=key, discard=0,
                                                source='grid5', version=version)

    return mcmc_params.get_radius(mass_nw=flat['m_nw'], mass_gr=flat['m_gr'])

