import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyburst.grids import grid_analyser
from pyburst.synth import synth
from pyburst.plotting import plot_tools
from pyburst.mcmc import mcmc_plot, mcmc_versions, mcmc_tools, burstfit

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


def plot_interp_residuals(synth_source, batches, mc_source, mc_version,
                          fontsize=16):
    """Plot synthetic burst properties against interpolated predictions
        to test accuracy of interpolator
    """
    n_sigma = 1.96
    bfit = burstfit.BurstFit(source=mc_source, version=mc_version)
    bprops = bfit.mcmc_version.bprops

    kgrid = grid_analyser.Kgrid(source=synth_source)
    param_table = kgrid.get_combined_params(batches)

    interp_table = extract_interp_table(param_table, bfit=bfit)
    summ_table = kgrid.get_combined_summ(batches)

    fig, ax = plt.subplots(len(bprops), figsize=(6, 8))

    for i, bprop in enumerate(bprops):
        u_bprop = f'u_{bprop}'
        yscale = plot_tools.unit_scale(bprop)
        yunits = plot_tools.unit_label(bprop)

        model = np.array(summ_table[bprop]) / yscale
        interp = np.array(interp_table[bprop]) / yscale
        u_model = np.array(summ_table[u_bprop]) / yscale
        u_interp = np.array(interp_table[u_bprop]) / yscale

        residuals = interp - model
        u_residuals = n_sigma * np.sqrt(u_model**2 + u_interp**2)

        ax[i].errorbar(model, residuals, yerr=u_residuals, marker='o',
                       ls='none', capsize=3)

        x_max = np.max(model)
        x_min = np.min(model)
        ax[i].plot([0.9*x_min, 1.1*x_max], [0, 0], ls='--', color='black')
        ax[i].set_xlabel(f'{bprop} ({yunits})', fontsize=fontsize)

    ax[1].set_ylabel(f'Interpolated - model', fontsize=fontsize)
    plt.tight_layout()
    plt.show(block=False)


def extract_interp_params(param_table, mcv):
    """Returns np.array of params (n_models, n_params) ready for input to interpolator
    """
    aliases = {'mdot': 'accrate'}
    n_params = len(mcv.interp_keys)
    n_models = len(param_table)
    interp_params = np.full((n_models, n_params), np.nan)

    for i, key in enumerate(mcv.interp_keys):
        key = aliases.get(key, key)
        interp_params[:, i] = np.array(param_table[key])

    return interp_params


def extract_interp_table(param_table, bfit):
    """Returns pd.DataFrame of interpolated burst properties, from given param table

    parameters
    ----------
    param_table : pd.DataFrame
        subset of Kgrid.params to get interpolations for
    bfit : burstfit.BurstFit
        corresponding BurstFit object to use for interpolation
    """
    interp_params = extract_interp_params(param_table, mcv=bfit.mcmc_version)
    interp_bursts = bfit.interpolate(interp_params)

    interp_table = pd.DataFrame()
    for i, bprop in enumerate(bfit.mcmc_version.bprops):
        u_bprop = f'u_{bprop}'
        interp_table[bprop] = interp_bursts[:, 2*i]
        interp_table[u_bprop] = interp_bursts[:, 2*i + 1]

    return interp_table
