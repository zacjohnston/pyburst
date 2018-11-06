import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygrids.synth import synth
from pygrids.mcmc import mcmc_plot, mcmc_versions, mcmc_tools

def plot_posteriors(source, mc_version, discard, chain=None, n_walkers=None,
                    n_steps=None):
    """Plots mcmc posteriors for synthetic data
    """
    mcv = mcmc_versions.McmcVersion(source, mc_version)
    chain = check_chain(chain, n_walkers=n_walkers, n_steps=n_steps, source=source,
                        version=mc_version)
    truth = synth.get_true_values(source, version=mcv.synth_version,
                                  group=mcv.synth_group)
    mcmc_plot.plot_posteriors(chain, source=source, version=mcv.synth_version,
                              discard=discard, truth_values=truth)


def check_chain(chain, n_walkers, n_steps, source, version):
    """Checks if chain was provided or needs loading
    """
    if chain is None:
        if None in (n_walkers, n_steps):
            raise ValueError('Must provide either chain, or both n_walkers and n_steps')
        else:
            chain = mcmc_tools.load_chain(source, version=version, n_walkers=n_walkers,
                                          n_steps=n_steps)

    return chain
