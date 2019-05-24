import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import chainconsumer
from math import ceil

# kepler_grids
from . import mcmc_versions
from . import mcmc_tools
from . import burstfit
from pyburst.physics import gravity
from pyburst.plotting import plot_tools
from pyburst.grids.grid_strings import get_source_path, print_warning

GRIDS_PATH = os.environ['KEPLER_GRIDS']


def default_plt_options():
    """Initialise default plot parameters"""
    params = {'mathtext.default': 'regular',
              'font.family': 'serif',
              'text.usetex': False}
    plt.rcParams.update(params)


default_plt_options()


def save_plot(fig, prefix, save, source, version, display, chain=None, n_dimensions=None,
              n_walkers=None, n_steps=None, label=None, extension='.png'):
    """Handles saving/displaying of a figure passed to it
    """
    if None in (n_dimensions, n_walkers, n_steps):
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
                        display=False, max_lhood=False, mass_radius=True,
                        synth=True):
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
                                      version=version)

        if walkers:
            plot_walkers(chain, source=full_source, save=True,
                         display=display, version=version)

        if posteriors:
            plot_posteriors(chain, source=full_source, save=True, discard=discard,
                            display=display, version=version, max_lhood=max_lhood)

        if contours:
            plot_contours(chain, source=full_source, save=True, discard=discard,
                          display=display, version=version, max_lhood=max_lhood)

        if mass_radius:
            plot_mass_radius(chain, source=full_source, save=True, discard=discard,
                             display=display, version=version, max_lhood=max_lhood)


def plot_contours(chain, discard, source, version, cap=None, truth=False, max_lhood=False,
                  display=True, save=False, truth_values=None, verbose=True,
                  smoothing=False):
    """Plots posterior contours of mcmc chain
    """
    default_plt_options()
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
    pkey_labels = plot_tools.convert_mcmc_labels(param_keys=pkeys)
    # TODO: re-use the loaded chainconsumer here instead of reloading
    cc = setup_chainconsumer(chain=chain, param_labels=pkey_labels, discard=discard,
                             cap=cap)

    if max_lhood:
        n_walkers, n_steps = chain[:, :, 0].shape
        max_params = mcmc_tools.get_max_lhood_params(source, version=version, n_walkers=n_walkers,
                                                     n_steps=n_steps, verbose=verbose)
        fig = cc.plotter.plot(truth=max_params, display=display)
    elif truth:
        if truth_values is None:
            truth_values = get_summary(chain, discard=discard, cap=cap,
                                       source=source, version=version)[:, 1]

        fig = cc.plotter.plot(truth=truth_values, display=display)
    else:
        fig = cc.plotter.plot(display=display)

    plt.tight_layout()
    save_plot(fig, prefix='contours', chain=chain, save=save, source=source,
              version=version, display=display)


def plot_posteriors(chain, discard, source, version, cap=None, max_lhood=False,
                    display=True, save=False, truth_values=None, verbose=True):
    """Plots posterior distributions of mcmc chain

    max_lhood : bool
        plot location of the maximum likelihood point. Overrides truth_values if True.
    truth_values : list|dict
        Specify parameters of point (e.g. the true value) to draw on the distributions.
        Will be overidden if max_lhood=True
    """
    default_plt_options()
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
    pkey_labels = plot_tools.convert_mcmc_labels(param_keys=pkeys)
    cc = setup_chainconsumer(chain=chain, param_labels=pkey_labels, discard=discard,
                             cap=cap)
    height = 3 * ceil(len(pkeys) / 4)
    if truth_values is not None:
        fig = cc.plotter.plot_distributions(display=display, figsize=[10, height],
                                            truth=truth_values)
    elif max_lhood:
        n_walkers, n_steps = chain[:, :, 0].shape
        max_params = mcmc_tools.get_max_lhood_params(source, version=version, n_walkers=n_walkers,
                                                     n_steps=n_steps, verbose=verbose)
        fig = cc.plotter.plot_distributions(display=display, figsize=[10, height],
                                            truth=max_params)
    else:
        fig = cc.plotter.plot_distributions(display=display, figsize=[10, height])

    plt.tight_layout()
    save_plot(fig, prefix='posteriors', chain=chain, save=save, source=source,
              version=version, display=display)


def plot_mass_radius(chain, discard, source, version, cap=None,
                     display=True, save=False, max_lhood=False, verbose=True,
                     cloud=True, sigmas=np.linspace(0, 2, 10)):
    """Plots contours of mass versus radius

    See: get_mass_radius()
    """
    default_plt_options()
    mass_radius_chain = get_mass_radius(chain=chain, discard=discard,
                                        source=source, version=version, cap=cap)

    cc = chainconsumer.ChainConsumer()
    cc.add_chain(mass_radius_chain.reshape(-1, 2), parameters=['R', 'M'])
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0)

    if max_lhood:
        n_walkers, n_steps = chain[:, :, 0].shape
        max_params = mcmc_tools.get_max_lhood_params(source, version=version, n_walkers=n_walkers,
                                                     n_steps=n_steps, verbose=verbose)
        mass, radius = get_mass_radius_point(max_params, source=source, version=version)
        fig = cc.plotter.plot(display=True, figsize=[6, 6], truth=[mass, radius])
    else:
        fig = cc.plotter.plot(display=True, figsize=[6, 6])

    save_plot(fig, prefix='mass-radius', chain=chain, save=save, source=source,
              version=version, display=display)


def plot_xedd(chain, discard, source, version, cap=None,
              display=True, save=False, max_lhood=False, verbose=True,
              cloud=True, sigmas=np.linspace(0, 2, 10)):
    """Plots posterior for Eddington hydrogen composition (X_Edd)
    """
    default_plt_options()
    xedd_chain = get_xedd_chain(chain=chain, discard=discard, source=source,
                                version=version, cap=cap)

    cc = chainconsumer.ChainConsumer()
    label = plot_tools.mcmc_label('xedd')
    cc.add_chain(xedd_chain.reshape(-1), parameters=[label])
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0)

    # TODO: plot max_lhood option
    fig = cc.plotter.plot(display=True, figsize=[6, 6])

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


def plot_qb(chain, discard, source, version, cap=None, summ=None, log=False):
    """Plot Qb versus accrate from MCMC run
    """
    fontsize = 14
    mc_version = mcmc_versions.McmcVersion(source, version=version)
    if 'qb' not in mc_version.epoch_unique:
        raise ValueError(f"Qb is not an epoch parameter in "
                         f"source '{source}', version '{version}'")
    if summ is None:
        summ = get_summary(chain, discard=discard, source=source, version=version, cap=cap)

    fig, ax = plt.subplots()

    n_epochs = 3
    mdot_i0 = mc_version.param_keys.index('mdot1')
    qb_i0 = mc_version.param_keys.index('qb1')

    mdot = summ[mdot_i0:mdot_i0 + n_epochs]
    qb = summ[qb_i0:qb_i0 + n_epochs]
    xerr = np.diff(mdot).transpose()
    yerr = np.diff(qb).transpose()

    x = mdot[:, 1]
    y = qb[:, 1]

    if log:
        xerr = xerr / (x * np.log(10))
        yerr = yerr / (y * np.log(10))
        x = np.log10(x)
        y = np.log10(y)

    ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr,
                marker='o', capsize=3, color='C0', ls='none')

    ax.set_ylabel('$Q_\mathrm{b}$ (MeV nucleon$^{-1}$)', fontsize=fontsize)
    ax.set_xlabel('$\dot{M} / \dot{M}_\mathrm{Edd}$', fontsize=fontsize)
    plt.tight_layout()
    plt.show(block=False)


def setup_epochs_chainconsumer(chains, param_keys, discard, cap=None, sigmas=None,
                               cloud=None):
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
    chains_flat = []
    for chain in chains:
        sliced = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
        _, _, n_dimensions = sliced.shape
        chains_flat += [sliced.reshape((-1, n_dimensions))]

    cc = chainconsumer.ChainConsumer()

    for i, chain_flat in enumerate(chains_flat):
        param_labels = plot_tools.convert_mcmc_labels(param_keys[i])
        cc.add_chain(chain_flat, parameters=param_labels)

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
                        source=None, version=None, sigmas=np.linspace(0, 2, 5)):
    """Return ChainConsumer object set up with given chain and pkeys
    """
    if param_labels is None:
        if (source is None) or (version is None):
            raise ValueError('If param_labels not provided, must give source, version')
        param_keys = mcmc_versions.get_parameter(source, version, 'param_keys')
        param_labels = plot_tools.convert_mcmc_labels(param_keys)

    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_dimensions = chain.shape[2]

    cc = chainconsumer.ChainConsumer()
    cc.add_chain(chain[:, :, :].reshape(-1, n_dimensions), parameters=param_labels)
    cc.configure(sigmas=sigmas, cloud=cloud, kde=False, smooth=0)
    return cc


def get_mass_radius(chain, discard, source, version, cap=None):
    """Returns GR mass and radius given a chain containing gravity and redshift

    Returns ndarray of equivalent form to input chain (after slicing discard/cap)
    """
    ref_radius = 10

    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_walkers, n_steps, n_dimensions = chain.shape
    chain_flat = chain.reshape((-1, n_dimensions))
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    mass_nw = chain_flat[:, pkeys.index('m_nw')]
    mass_gr = chain_flat[:, pkeys.index('m_gr')]
    m_ratio = mass_gr / mass_nw

    xi = gravity.gr_corrections(r=ref_radius, m=mass_nw, phi=m_ratio)[0]
    radius_gr = ref_radius * xi

    # reshape back into chain
    new_shape = (n_walkers, n_steps)
    mass_reshape = mass_gr.reshape(new_shape)
    radius_reshape = radius_gr.reshape(new_shape)

    return np.dstack((radius_reshape, mass_reshape))


def get_mass_radius_point(params, source, version):
    """Returns the mass, radius for a single walker point
    """
    ref_radius = 10
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    mass_nw = params[pkeys.index('m_nw')]
    mass_gr = params[pkeys.index('m_gr')]
    m_ratio = mass_gr / mass_nw

    xi = gravity.gr_corrections(r=ref_radius, m=mass_nw, phi=m_ratio)[0]
    radius_gr = ref_radius * xi

    return radius_gr, mass_gr


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
        print_warning(f'lhoods do not match ({max_lhood:.2f}, {lhood:.2f}). '
                      + 'BurstFit (lhood, lnhood) or interpolator may have changed')

    save_plot(fig, prefix='compare', n_dimensions=len(max_params),
              n_walkers=n_walkers, n_steps=n_steps, save=save, source=source,
              version=version, display=display)


def animate_contours(chain, source, version, dt=5, fps=20, ffmpeg=True):
    """Saves frames of contour evolution, to make an animation
    """
    default_plt_options()
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')

    n_walkers, n_steps, n_dimensions = chain.shape
    mtarget = os.path.join(GRIDS_PATH, 'sources', source, 'mcmc', 'animation')
    ftarget = os.path.join(mtarget, 'frames')

    cc = chainconsumer.ChainConsumer()

    for i in range(dt, n_steps, dt):
        print('frame  ', i)
        subchain = chain[:, :i, :].reshape((-1, n_dimensions))
        cc.add_chain(subchain, parameters=pkeys)

        fig = cc.plotter.plot()
        fig.set_size_inches(6, 6)
        cnt = round(i / dt)

        filename = f'{cnt:04d}.png'
        filepath = os.path.join(ftarget, filename)
        fig.savefig(filepath)

        plt.close(fig)
        cc.remove_chain()

    if ffmpeg:
        print('Creating movie')
        framefile = os.path.join(ftarget, f'%04d.png')
        savefile = os.path.join(mtarget, f'chain.mp4')
        subprocess.run(['ffmpeg', '-r', str(fps), '-i', framefile, savefile])
