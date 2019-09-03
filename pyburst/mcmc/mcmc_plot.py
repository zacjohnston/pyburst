import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import chainconsumer
from math import ceil

# pyburst
from . import mcmc_versions
from . import mcmc_tools
from . import burstfit
from . import mcmc_params
from pyburst.observations import obs_tools
from pyburst.plotting import plot_tools
from pyburst.grids.grid_strings import get_source_path, print_warning
from pyburst.misc.pyprint import printv

GRIDS_PATH = os.environ['KEPLER_GRIDS']


def default_plt_options():
    """Initialise default plot parameters"""
    params = {'mathtext.default': 'regular',
              'font.family': 'serif',
              'text.usetex': False}
    plt.rcParams.update(params)


default_plt_options()


def save_plot(fig, prefix, save, source, version, display, chain=None, n_dimensions=None,
              n_walkers=None, n_steps=None, label=None, extension='.png',
              enforce_chain_info=True):
    """Handles saving/displaying of a figure passed to it
    """
    if enforce_chain_info and (None in (n_dimensions, n_walkers, n_steps)):
        if chain is None:
            raise ValueError('Must provide chain, or specify each of '
                             '(n_dimensions, n_walkers, n_steps)')
        else:
            n_walkers, n_steps, n_dimensions = chain.shape

    if save:
        filename = mcmc_tools.get_mcmc_string(source=source, version=version,
                                              n_walkers=n_walkers, n_steps=n_steps,
                                              prefix=prefix, label=label,
                                              extension=extension)
        source_path = get_source_path(source)
        filepath = os.path.join(source_path, 'plots', prefix, f'{filename}')
        fig.savefig(filepath)

    if display:
        plt.show(block=False)
    else:
        plt.close(fig)


def save_multiple_synth(series, source, version, n_steps, discard, n_walkers=960,
                        walkers=True, posteriors=True, contours=False,
                        display=False, mass_radius=True,
                        synth=True, compressed=False):
    """Save plots for multiple series in a synthetic data batch
    """
    # TODO reuse max_lhood point
    default_plt_options()
    for ser in series:
        if synth:
            full_source = f'{source}_{ser}'
        else:
            full_source = source

        chain = mcmc_tools.load_chain(full_source, n_walkers=n_walkers, n_steps=n_steps,
                                      version=version, compressed=compressed)

        if walkers:
            plot_walkers(chain, source=full_source, save=True,
                         display=display, version=version)

        if posteriors:
            plot_posteriors(chain, source=full_source, save=True, discard=discard,
                            display=display, version=version)

        if contours:
            plot_contours(chain, source=full_source, save=True, discard=discard,
                          display=display, version=version)

        if mass_radius:
            plot_mass_radius(chain, source=full_source, save=True, discard=discard,
                             display=display, version=version)


def save_all_plots(source, version, discard, n_steps, n_walkers=1000, display=False,
                   save=True, cap=None, posteriors=True, contours=True,
                   redshift=True, mass_radius=True, verbose=True, compressed=False):
    """Saves (and/or displays) main MCMC plots
    """
    chain = mcmc_tools.load_chain(source, version=version, n_steps=n_steps,
                                  n_walkers=n_walkers, verbose=verbose,
                                  compressed=compressed)
    if posteriors:
        printv('Plotting posteriors', verbose=verbose)
        plot_posteriors(chain, source=source, save=save, discard=discard, cap=cap,
                        display=display, version=version)

    if contours:
        printv('Plotting contours', verbose=verbose)
        plot_contours(chain, source=source, save=save, discard=discard, cap=cap,
                      display=display, version=version)

    if mass_radius:
        printv('Plotting mass-radius', verbose=verbose)
        plot_mass_radius(chain, source=source, save=save, discard=discard, cap=cap,
                         display=display, version=version)

    if redshift:
        printv('Plotting redshift', verbose=verbose)
        plot_redshift(chain, source=source, save=save, discard=discard, cap=cap,
                      display=display, version=version)


def plot_contours(chain, discard, source, version, cap=None,
                  display=True, save=False, truth_values=None, parameters=None,
                  sigmas=np.linspace(0, 2, 5), cc=None, summary=False):
    """Plots posterior contours of mcmc chain

    parameters : [str]
        specify which parameters to plot
    """
    default_plt_options()

    if cc is None:
        pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
        pkey_labels = plot_tools.convert_mcmc_labels(param_keys=pkeys)
        cc = setup_chainconsumer(chain=chain, param_labels=pkey_labels,
                                 discard=discard, cap=cap, sigmas=sigmas,
                                 summary=summary)
    if parameters is not None:
        parameters = plot_tools.convert_mcmc_labels(param_keys=parameters)

    if truth_values is not None:
        fig = cc.plotter.plot(truth=truth_values, parameters=parameters)
    else:
        fig = cc.plotter.plot(parameters=parameters)

    save_plot(fig, prefix='contours', chain=chain, save=save, source=source,
              version=version, display=display)
    return fig


def plot_posteriors(chain, discard, source, version, cap=None,
                    display=True, save=False, truth_values=None,
                    cc=None):
    """Plots posterior distributions of mcmc chain

    truth_values : list|dict
        Specify parameters of point (e.g. the true value) to draw on the distributions.
    """
    default_plt_options()
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
    pkey_labels = plot_tools.convert_mcmc_labels(param_keys=pkeys)
    if cc is None:
        cc = setup_chainconsumer(chain=chain, param_labels=pkey_labels, discard=discard,
                                 cap=cap)
    height = 3 * ceil(len(pkeys) / 4)

    if truth_values is not None:
        fig = cc.plotter.plot_distributions(figsize=[10, height],
                                            truth=truth_values)
    else:
        fig = cc.plotter.plot_distributions(figsize=[10, height])

    plt.tight_layout()
    save_plot(fig, prefix='posteriors', chain=chain, save=save, source=source,
              version=version, display=display)
    return fig


def plot_mass_radius(chain, discard, source, version, cap=None,
                     display=True, save=False, cloud=False,
                     sigmas=np.linspace(0, 2, 5), fontsize=18, figsize='column'):
    """Plots contours of mass versus radius from a given chain
    """
    # TODO: combine and generalise with plot_xedd()
    default_plt_options()
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    mass_nw, mass_gr = mcmc_params.get_constant_masses(source, version)
    mass_radius_chain = mcmc_params.get_mass_radius_chain(chain=chain, discard=discard,
                                                          source=source, version=version,
                                                          cap=cap, mass_nw=mass_nw,
                                                          mass_gr=mass_gr)
    cc = chainconsumer.ChainConsumer()
    cc.add_chain(mass_radius_chain, parameters=['R', 'M'])
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0)
    fig = cc.plotter.plot(figsize=figsize)

    # Manually set axis labels
    labels = plot_tools.convert_full_labels(['radius', 'mass'])
    fig.axes[2].set_xlabel(labels[0], fontsize=fontsize)
    fig.axes[2].set_ylabel(labels[1], fontsize=fontsize)

    for tick in fig.axes[2].xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in fig.axes[2].yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    fig.subplots_adjust(left=0.16, bottom=0.15)

    save_plot(fig, prefix='mass-radius', chain=chain, save=save, source=source,
              version=version, display=display)
    return fig


def plot_redshift(chain, discard, source, version, cap=None, display=True, save=False):
    """Plots posterior distribution of redshift given a chain
    """
    mass_nw, mass_gr = mcmc_params.get_constant_masses(source, version)
    redshift_chain = mcmc_params.get_redshift_chain(chain=chain, discard=discard,
                                                    source=source, version=version,
                                                    cap=cap, mass_nw=mass_nw,
                                                    mass_gr=mass_gr)

    cc = chainconsumer.ChainConsumer()
    cc.add_chain(redshift_chain, parameters=['(1+z)'])
    cc.configure(kde=False, smooth=0)

    fig = cc.plotter.plot_distributions(figsize=[5, 5])
    plt.tight_layout()

    save_plot(fig, prefix='redshift', chain=chain, save=save, source=source,
              version=version, display=display)
    return fig


def plot_gravitational_contours(chain, discard, source, version, cap=None, display=True,
                                save=False, r_nw=10, sigmas=np.linspace(0, 2, 5),
                                summary=False):
    """Plots contours of gravitational parameters
    """
    grav_chain = mcmc_params.get_gravitational_chain(chain=chain, discard=discard,
                                                     source=source, version=version,
                                                     cap=cap, r_nw=r_nw)
    # TODO: generalise new setup_chainconsumer for derived params
    cc = chainconsumer.ChainConsumer()
    cc.add_chain(grav_chain, parameters=['R', 'M', 'g', '1+z'])
    cc.configure(kde=False, smooth=0, sigmas=sigmas, summary=summary)

    fig = cc.plotter.plot()
    save_plot(fig, prefix='gravitational', chain=chain, save=save, source=source,
              version=version, display=display)
    return fig


def plot_disc_contours(chain, discard, source, version, cap=None, display=True,
                       save=False, disc_model='he16_a', sigmas=np.linspace(0, 2, 5),
                       summary=False):
    """Plots contours of parameters derived using disc model
    """
    disc_chain = mcmc_params.get_disc_chain(chain=chain, discard=discard, cap=cap,
                                            source=source, version=version,
                                            disc_model=disc_model)
    cc = chainconsumer.ChainConsumer()
    cc.add_chain(disc_chain, parameters=['i', 'xib', 'xip', 'd'])
    cc.configure(kde=False, smooth=0, sigmas=sigmas, summary=summary)

    fig = cc.plotter.plot()
    save_plot(fig, prefix='disc', chain=chain, save=save, source=source,
              version=version, display=display)
    return fig


def plot_distance_anisotropy(chain, discard, source, version, cap=None, display=True,
                             save=False, sigmas=np.linspace(0, 2, 5), summary=False,
                             figsize=(6, 6)):
    """Plots contours of MCMC parameters d_b, xi_ratio
    """
    d_b_chain = mcmc_params.get_param_chain(chain, param='d_b', discard=discard,
                                            source=source, version=version, cap=cap)
    xi_ratio_chain = mcmc_params.get_param_chain(chain, param='xi_ratio', discard=discard,
                                                 source=source, version=version, cap=cap)
    cc = chainconsumer.ChainConsumer()
    cc.add_chain(np.column_stack([d_b_chain, xi_ratio_chain]),
                 parameters=['db', 'xiratio'])
    cc.configure(kde=False, smooth=0, sigmas=sigmas, summary=summary)

    fig = cc.plotter.plot(figsize=figsize)
    plt.tight_layout()
    save_plot(fig, prefix='distance', chain=chain, save=save, source=source,
              version=version, display=display)
    return fig


def plot_xedd(chain, discard, source, version, cap=None, display=True,
              save=False, cloud=True, sigmas=np.linspace(0, 2, 10), figsize=(6, 6)):
    """Plots posterior for Eddington hydrogen composition (X_Edd)
    """
    default_plt_options()
    xedd_chain = mcmc_params.get_xedd_chain(chain=chain, discard=discard, source=source,
                                            version=version, cap=cap)

    cc = chainconsumer.ChainConsumer()
    label = plot_tools.quantity_label('xedd')
    cc.add_chain(xedd_chain, parameters=[label])
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0)

    fig = cc.plotter.plot(figsize=figsize)

    save_plot(fig, prefix='xedd', chain=chain, save=save, source=source,
              version=version, display=display)


def plot_walkers(chain, source, version, params=None, n_lines=30, xlim=-1,
                 display=True, save=False, label=''):
    """Plots walkers vs steps (i.e. "time")

    Parameters
    ----------
    source : str
    version : int
    chain : np.array
        chain as returned by load_chain()
    params : [str]
        parameter(s) of which to plot walkers.
    n_lines : int
        approx number of lines/walkers to plot on parameter
    xlim : int
        x-axis limit to plot (n_steps), i.e. ax.set_xlim((0, xlim))
    label : str
        optional label to add to filename when saving
    display : bool
    save : bool
    """
    default_plt_options()
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    # ===== Default to splitting all params into 2 plots  =====
    if params is None:
        half = int(len(pkeys) / 2)
        for i, param_split in enumerate((pkeys[:half], pkeys[half:])):
            plot_walkers(chain=chain, source=source, version=version,
                         params=param_split, n_lines=n_lines, xlim=xlim,
                         display=display, save=save, label=f'P{i+1}')
        return

    n_walkers, n_steps, n_dim = chain.shape
    n_params = len(params)

    jump_size = round(n_walkers / n_lines)
    steps = np.arange(n_steps)
    walker_idxs = np.arange(0, n_walkers, jump_size)

    # noinspection PyTypeChecker
    fig, ax = plt.subplots(n_params, 1, sharex=True, figsize=(10, 12))

    for i in range(n_params):
        p_idx = pkeys.index(params[i])

        for j in walker_idxs:
            walker = chain[j, :, p_idx]
            ax[i].plot(steps, walker, linewidth=0.5, color='black')
            ax[i].set_ylabel(params[i])

    if xlim == -1:
        xlim = n_steps

    ax[-1].set_xlabel('Step')
    ax[-1].set_xlim([0, xlim])
    plt.tight_layout()

    if display:
        plt.show(block=False)

    save_plot(fig, prefix='walkers', chain=chain, save=save, source=source,
              version=version, display=display,
              label=label, extension='.png')


def plot_qb_mdot(chain, source, version, discard, cap=None, display=True, save=False):
    """Plots 2D contours of Qb versus Mdot for each epoch (from multi-epoch chain)
    """
    mv = mcmc_versions.McmcVersion(source=source, version=version)
    chain_flat = mcmc_tools.slice_chain(chain, discard=discard, cap=cap, flatten=True)

    system_table = obs_tools.load_summary(mv.system)
    epochs = list(system_table.epoch)
    cc = chainconsumer.ChainConsumer()

    param_labels = [r'$\dot{M} / \dot{M}_\mathrm{Edd}$',
                    r'$Q_\mathrm{b}$ (MeV nucleon$^{-1}$)']

    for i, epoch in enumerate(epochs):
        mdot_idx = mv.param_keys.index(f'mdot{i + 1}')
        qb_idx = mv.param_keys.index(f'qb{i + 1}')
        param_idxs = [mdot_idx, qb_idx]

        cc.add_chain(chain_flat[:, param_idxs], parameters=param_labels,
                     name=str(epoch))

    cc.configure(kde=False, smooth=0)
    fig = cc.plotter.plot(display=False, figsize=(6, 6))
    plt.tight_layout()
    save_plot(fig, prefix='qb', save=save, source=source, version=version,
              display=display, chain=chain)
    return fig


def plot_epoch_posteriors(master_cc, source, version, display=True, save=False,
                          col_wrap=None):
    """Plot posteriors for multiiple epoch chains

    parameters
    ----------
    master_cc : ChainConsumer
        Contains the multi-epoch chain, created with setup_master_chainconsumer()
    source : str
    version : int
    display : bool (optional)
    save : bool (optional)
    col_wrap : int (optional)
    """
    param_order = {
        'grid5': ['mdot1', 'mdot2', 'mdot3', 'qb1', 'qb2', 'qb3', 'x', 'z', 'm_nw',
                  'm_gr', 'd_b', 'xi_ratio'],
        'he2': ['mdot1', 'mdot2', 'qb1', 'qb2', 'm_gr', 'd_b', 'xi_ratio'],
    }

    param_keys = param_order[source]
    formatted_params = plot_tools.convert_mcmc_labels(param_keys)
    n_epochs = len(master_cc.chains) - 1

    if col_wrap is None:
        col_wrap = n_epochs

    height = 3 * ceil(len(param_keys) / n_epochs)
    fig = master_cc.plotter.plot_distributions(parameters=formatted_params,
                                               col_wrap=col_wrap,
                                               figsize=[8, height],
                                               display=False)
    plt.tight_layout()

    save_plot(fig, prefix='multi_posteriors', save=save, source=source, version=version,
              display=display, enforce_chain_info=False)
    return fig


def setup_master_chainconsumer(source, master_version, epoch_versions, n_steps, discard,
                               n_walkers=1000, epoch_discard=None, epoch_n_steps=None,
                               epoch_n_walkers=None, cap=None, sigmas=None, cloud=None,
                               compressed=False):
    """Setup multiple MCMC chains, including multi-epoch and single-epochs
    """
    if epoch_discard is None:
        epoch_discard = discard
    if epoch_n_steps is None:
        epoch_n_steps = n_steps
    if epoch_n_walkers is None:
        epoch_n_walkers = n_walkers

    cc = setup_epochs_chainconsumer(source, versions=epoch_versions, n_steps=epoch_n_steps,
                                    discard=epoch_discard, n_walkers=epoch_n_walkers,
                                    cap=cap, sigmas=sigmas, cloud=cloud, compressed=False)

    # ===== Setup master chain =====
    master_mc_v = mcmc_versions.McmcVersion(source, version=master_version)

    master_chain = mcmc_tools.load_chain(source, version=master_version, n_steps=n_steps,
                                         n_walkers=n_walkers, compressed=compressed)
    master_chain_sliced = mcmc_tools.slice_chain(master_chain, discard=discard, cap=cap,
                                                 flatten=True)

    formatted_params = plot_tools.convert_mcmc_labels(master_mc_v.param_keys)
    cc.add_chain(master_chain_sliced, parameters=formatted_params, color='black',
                 name='Multi-epoch')
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=False)

    return cc


def setup_epochs_chainconsumer(source, versions, n_steps, discard, n_walkers=1000,
                               cap=None, sigmas=None, cloud=None, compressed=False):
    """Setup multiple MCMC chains fit to individual epochs

    chains : [n_epochs]
        list of raw numpy chains
    param_keys : [n_epochs]
        list of parameters for each epoch chain
    discard : int
    cap : int (optional)
    sigmas : [] (optional)
    cloud : bool (optional)
    """
    param_keys = mcmc_tools.load_multi_param_keys(source, versions=versions)
    chains = mcmc_tools.load_multi_chains(source, versions=versions, n_steps=n_steps,
                                          n_walkers=n_walkers, compressed=compressed)
    chains_flat = []
    for chain in chains:
        sliced_flat = mcmc_tools.slice_chain(chain, discard=discard, cap=cap, flatten=True)
        chains_flat += [sliced_flat]

    cc = chainconsumer.ChainConsumer()

    for i, chain_flat in enumerate(chains_flat):
        epoch = mcmc_versions.get_parameter(source, version=versions[i], parameter='epoch')
        param_labels = plot_tools.convert_mcmc_labels(param_keys[i])
        cc.add_chain(chain_flat, parameters=param_labels, name=str(epoch))

    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0)
    return cc


def get_summary(chain, discard, source, version, cap=None):
    """Return summary values from MCMC chain (mean, uncertainties)
    """
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
    n_dimensions = chain.shape[2]
    summary = np.full((n_dimensions, 3), np.nan)
    cc = setup_chainconsumer(chain=chain, param_labels=pkeys, discard=discard, cap=cap)
    summary_dict = cc.analysis.get_summary()

    for i, key in enumerate(pkeys):
        summary[i, :] = summary_dict[key]
    return summary


def setup_chainconsumer(chain, discard, cap=None, param_labels=None, cloud=False,
                        source=None, version=None, sigmas=np.linspace(0, 2, 5),
                        summary=False):
    """Return ChainConsumer object set up with given chain and pkeys
    """
    if param_labels is None:
        if (source is None) or (version is None):
            raise ValueError('If param_labels not provided, must give source, version')
        param_keys = mcmc_versions.get_parameter(source, version, 'param_keys')
        param_labels = plot_tools.convert_mcmc_labels(param_keys)

    n_walkers = chain.shape[0]
    chain_flat = mcmc_tools.slice_chain(chain, discard=discard, cap=cap, flatten=True)

    cc = chainconsumer.ChainConsumer()
    cc.add_chain(chain_flat, parameters=param_labels, walkers=n_walkers)
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0, summary=summary)
    return cc


def setup_custom_chainconsumer(chain, parameters, cloud=False,
                               sigmas=np.linspace(0, 2, 5), summary=False):
    """Returns ChainConsumer, with derived parameters

        Note: chain must already be flattened (and already discarded/capped)
    """
    param_labels = plot_tools.convert_mcmc_labels(parameters)

    cc = chainconsumer.ChainConsumer()
    cc.add_chain(chain, parameters=param_labels)
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0, summary=summary)

    return cc


def plot_max_lhood(source, version, n_walkers, n_steps, verbose=True, re_interp=False,
                   display=True, save=False):
    default_plt_options()
    max_params, max_lhood = mcmc_tools.get_max_lhood_params(source, version=version,
                                                            n_walkers=n_walkers,
                                                            n_steps=n_steps,
                                                            verbose=verbose,
                                                            return_lhood=True)
    bfit = burstfit.BurstFit(source=source, version=version, verbose=False, re_interp=re_interp)
    lhood, fig = bfit.lhood(max_params, plot=True)

    if lhood != max_lhood:
        print_warning(f'lhoods do not match (original={max_lhood:.2f}, current={lhood:.2f}). '
                      + 'BurstFit (e.g. lhood, lnhood) or interpolator may have changed')

    save_plot(fig, prefix='compare', n_dimensions=len(max_params),
              n_walkers=n_walkers, n_steps=n_steps, save=save, source=source,
              version=version, display=display)


def plot_bprop_sample(bprop_sample, source, version, subplot_figsize=(3, 2.5),
                      bfit=None):
    """Plot burst properties from large sample against observations
    """
    if bfit is None:
        bfit = burstfit.BurstFit(source=source, version=version, verbose=False)

    n_bprops = bprop_sample.shape[1]
    bprop_mean = np.mean(bprop_sample, axis=2)
    bprop_std = np.std(bprop_sample, axis=2)

    n_rows = int(np.ceil(n_bprops / 2))
    figsize = (2 * subplot_figsize[0], n_rows * subplot_figsize[1])
    fig, ax = plt.subplots(n_rows, 2, sharex=True, figsize=figsize)

    if n_bprops % 2 == 1:   # blank odd-numbered subplot
        ax[-1, -1].axis('off')

    for i, bprop in enumerate(bfit.mcmc_version.bprops):
        subplot_row = int(np.floor(i / 2))
        subplot_col = i % 2
        bfit.plot_compare(model=bprop_mean[:, i], u_model=bprop_std[:, i],
                          bprop=bfit.mcmc_version.bprops[i],
                          ax=ax[subplot_row, subplot_col], display=False,
                          legend=True if i == 0 else False,
                          xlabel=True if (i in [3, 4]) else False)

    plt.show(block=False)
    return fig


def plot_autocorrelation(chain, source, version, n_steps=10):
    """Plots estimated integrated autocorrelation time

        Note: Adapted from https://dfm.io/posts/autocorr/
    """
    # TODO: use save_plot()
    #   - save estimate values for re-use
    if n_steps < 2:
        raise ValueError('n_steps must be greater than 1')

    mv = mcmc_versions.McmcVersion(source=source, version=version)
    params_fmt = plot_tools.convert_mcmc_labels(mv.param_keys)

    sample_steps = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]),
                                      n_steps)).astype(int)
    fig, ax = plt.subplots()
    autoc = np.empty([len(mv.param_keys), n_steps])

    for i, param in enumerate(mv.param_keys):
        print(f'Calculating parameter: {param}')

        for j, n in enumerate(sample_steps):
            sys.stdout.write(f'\r{j+1}/{n_steps}  (step size={n})')
            autoc[i, j] = mcmc_tools.autocorrelation(chain[:, :n, i])

        ax.loglog(sample_steps, autoc[i], "o-", label=rf"{params_fmt[i]}")
        sys.stdout.write('\n')

    xlim = ax.get_xlim()
    ax.set_ylim([8, xlim[1] / 10])

    ax.plot(sample_steps, sample_steps / 10.0, "--k", label=r"$\tau = N/10$")
    ax.set_xlabel("number of samples, $N$")
    ax.set_ylabel(r"$\tau$ estimates")
    ax.legend(fontsize=14, ncol=2)

    plt.show(block=False)

    return sample_steps, autoc
