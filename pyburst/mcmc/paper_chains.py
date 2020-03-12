from . import mcmc_tools


def load_chain(version):
    """Load chain used in paper
    """
    n_steps = {6: 20000}.get(version, 10000)
    compressed = {6: True}.get(version, False)

    return mcmc_tools.load_chain(source='grid5', version=version, n_steps=n_steps,
                                 n_walkers=1000, compressed=compressed)
