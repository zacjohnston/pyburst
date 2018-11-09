import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygrids.synth import synth
from pygrids.mcmc import mcmc_plot, mcmc_versions, mcmc_tools

# TODO:
#   - plot contours
def plot_posteriors(source, mc_version, discard, chain=None, n_walkers=None,
                    n_steps=None, save=False, display=True):
    """Plots mcmc posteriors for synthetic data
    """
    plot_truth(plot_type='posteriors', source=source, mc_version=mc_version, discard=discard,
               chain=chain, n_walkers=n_walkers, n_steps=n_steps, save=save, display=display)


def plot_contours(source, mc_version, discard, chain=None, n_walkers=None,
                  n_steps=None, save=False, display=True):
    """Plots mcmc corner plot for synthetic data
    """
    plot_truth(plot_type='contours', source=source, mc_version=mc_version, discard=discard,
               chain=chain, n_walkers=n_walkers, n_steps=n_steps, save=save, display=display)


def plot_truth(plot_type, source, mc_version, discard, chain=None, n_walkers=None,
               n_steps=None, save=False, display=True):
    """Plots results of MCMC against true values of synthetic data
    """
    mcv = mcmc_versions.McmcVersion(source, mc_version)
    chain = check_chain(chain, n_walkers=n_walkers, n_steps=n_steps, source=source,
                        version=mc_version)
    truth = synth.get_true_values(source, version=mcv.synth_version,
                                  group=mcv.synth_group)

    if plot_type == 'posteriors':
        mcmc_plot.plot_posteriors(chain, source=source, version=mc_version,
                                  discard=discard, truth_values=truth, save=save,
                                  display=display)
    elif plot_type == 'contours':
        mcmc_plot.plot_contours(chain, discard=discard, source=source, truth=True,
                                version=mc_version, truth_values=truth, save=save,
                                display=display)
    else:
        raise ValueError('plot_type must be one of: (posteriors, corner)')


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
