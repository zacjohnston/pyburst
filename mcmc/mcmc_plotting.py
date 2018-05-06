import numpy as np
import sys
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import chainconsumer
from math import ceil

# kepler_grids
from ..physics import gparams
from ..grids.grid_strings import get_source_path
from . import mcmc_versions
from . import mcmc_tools

GRIDS_PATH = os.environ['KEPLER_GRIDS']


def save_plot(fig, prefix, chain, save, source, version,
              display, label=None, extension='.png'):
    """Handles saving/displaying of a figure passed to it
    """
    if save:
        n_walkers, n_steps, n_dimensions = chain.shape
        filename = mcmc_tools.get_mcmc_string(source=source, version=version,
                                              n_walkers=n_walkers, n_steps=n_steps,
                                              prefix=prefix, label=label,
                                              extension=extension)
        source_path = get_source_path(source)
        filepath = os.path.join(source_path, 'plots',
                                prefix, f'{filename}')
        fig.savefig(filepath)

    if not display:
        plt.close(fig)


def plot_contours(chain, discard, source, version, cap=None, truth=True,
                  display=True, save=False, truth_values=None):
    """Plots posterior contours of mcmc chain
    """
    pkeys = mcmc_versions.get_param_keys(source=source, version=version)
    # TODO: re-use the loaded chainconsumer here instead of reloading
    cc = setup_chainconsumer(chain=chain, param_labels=pkeys, discard=discard, cap=cap)

    if truth:
        if truth_values is None:
            truth_values = get_summary(chain, discard=discard, cap=cap,
                                       source=source, version=version)[:, 1]

        fig = cc.plotter.plot(truth=truth_values, display=display)
    else:
        fig = cc.plotter.plot(display=display)

    plt.tight_layout()
    save_plot(fig, prefix='contours', chain=chain, save=save, source=source,
              version=version, display=display)


def plot_posteriors(chain, discard, source, version, cap=None,
                    display=True, save=False, truth=False, truth_values=None):
    """Plots posterior distributions of mcmc chain
    """

    pkeys = mcmc_versions.get_param_keys(source=source, version=version)
    cc = setup_chainconsumer(chain=chain, param_labels=pkeys, discard=discard, cap=cap)
    height = 3 * ceil(len(pkeys) / 4)
    if truth:
        fig = cc.plotter.plot_distributions(display=display, figsize=[10, height],
                                            truth=truth_values)
    else:
        fig = cc.plotter.plot_distributions(display=display, figsize=[10, height])
    plt.tight_layout()
    save_plot(fig, prefix='posteriors', chain=chain, save=save, source=source,
              version=version, display=display)


def plot_mass_radius(chain, discard, source, version, cap=None,
                     display=True, save=False):
    """Plots contours of mass versus radius

    See: get_mass_radius()
    """
    mass_radius_chain = get_mass_radius(chain=chain, discard=discard,
                                        source=source, version=version, cap=cap)
    cc = chainconsumer.ChainConsumer()
    cc.add_chain(mass_radius_chain.reshape(-1, 2), parameters=['Mass', 'Radius'])
    fig = cc.plotter.plot(display=True, figsize=[6, 6])

    save_plot(fig, prefix='mass-radius', chain=chain, save=save, source=source,
              version=version, display=display)


def plot_walkers(chain, source, version, params=None, n_lines=100, xlim=-1,
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
    pkeys = mcmc_versions.get_param_keys(source=source, version=version)

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


def get_summary(chain, discard, source, version, cap=None):
    """Return summary values from MCMC chain (mean, uncertainties)
    """
    pkeys = mcmc_versions.get_param_keys(source=source, version=version)
    n_dimensions = chain.shape[2]
    summary = np.full((n_dimensions, 3), np.nan)
    cc = setup_chainconsumer(chain=chain, param_labels=pkeys, discard=discard, cap=cap)
    summary_dict = cc.analysis.get_summary()

    for i, key in enumerate(pkeys):
        summary[i, :] = summary_dict[key]
    return summary


def setup_chainconsumer(chain, discard, cap=None, param_labels=None,
                        source=None, version=None):
    """Return ChainConsumer object set up with given chain and pkeys
    """
    if type(param_labels) == type(None):
        if (source == None) or (version == None):
            raise ValueError('If param_labels not provided, must give source, version')
        param_labels = mcmc_versions.get_param_keys(source=source, version=version)

    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_dimensions = chain.shape[2]
    cc = chainconsumer.ChainConsumer()
    cc.add_chain(chain[:, :, :].reshape(-1, n_dimensions), parameters=param_labels)
    return cc


def get_mass_radius(chain, discard, source, version, cap=None):
    """Returns mass and radius given a chain containing gravity and redshift

    Returns ndarray of equivalent form to input chain (after slicing discard/cap)
    """
    pkeys = mcmc_versions.get_param_keys(source=source, version=version)
    mass_reference = 1.4
    radius_reference = 10
    g_reference = gparams.get_acceleration_newtonian(r=radius_reference,
                                                     m=mass_reference)
    g_idx = pkeys.index('g')
    red_idx = pkeys.index('redshift')

    chain = mcmc_tools.slice_chain(chain, discard=discard, cap=cap)
    n_walkers, n_steps, n_dimensions = chain.shape
    chain_flat = chain.reshape((-1, n_dimensions))

    redshift = chain_flat[:, red_idx]
    g = chain_flat[:, g_idx] * g_reference
    mass, radius = gparams.get_mass_radius(g=g, redshift=redshift)

    # reshape back into chain
    new_shape = (n_walkers, n_steps)
    mass_reshape = mass.value.reshape(new_shape)
    radius_reshape = radius.value.reshape(new_shape)

    return np.dstack((mass_reshape, radius_reshape))


def animate_contours(chain, source, version, dt=5, fps=20, ffmpeg=True):
    """Saves frames of contour evolution, to make an animation
    """
    pkeys = mcmc_versions.get_param_keys(source=source, version=version)
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


def animate_walkers(chain, source, version, stepsize=1, n_steps=100, bin=10, burn=100):
    mv = mcmc_versions.McmcVersion(source, version)
    g_idx = mv.param_keys.index('g')
    red_idx = mv.param_keys.index('redshift')
    save_path = os.path.join(GRIDS_PATH, 'sources', source, 'plots', 'misc', 'walker2')
    cc = chainconsumer.ChainConsumer()

    # ===== axis setup =====
    fig = plt.figure(1, figsize=(8, 8))

    nullfmt = NullFormatter()
    xlim = (0.6, 2.0)
    ylim = (1.08, 1.2)
    hist_ylim = (0, 1.1)

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)
    axScatter.set_xlabel('X', fontsize=20)
    axScatter.set_ylabel('Y', fontsize=20)

    axHistx.set_xlim(xlim)
    axHistx.set_ylim(hist_ylim)

    axHisty.set_ylim(ylim)
    axHisty.set_xlim(hist_ylim)

    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)

    # ===== Add data to axes =====
    for i in range(stepsize, stepsize * (n_steps + 1), stepsize):
        num = int(i / stepsize)
        sys.stdout.write(f'\r{num}/{n_steps}')

        # ===== walker scatter =====
        lines_scatter = axScatter.plot(chain[:, i, g_idx], chain[:, i, red_idx],
                                       marker='o', ls='none', markersize=2.5, color='C0')

        # ===== chainconsumer distributions =====
        # width1 = 10
        # burn= 100
        if i < bin:
            sub_chain = mcmc_tools.slice_chain(chain, discard=0, cap=i)
        elif i < burn:
            sub_chain = mcmc_tools.slice_chain(chain, discard=i - bin, cap=i)
        else:
            sub_chain = mcmc_tools.slice_chain(chain, discard=burn - bin, cap=i)

        cc.add_chain(sub_chain[:, :, [g_idx, red_idx]].reshape(-1, 2),
                     parameters=['g', 'redshift'])
        cc_fig = cc.plotter.plot_distributions(blind=True)

        x_x = cc_fig.axes[0].lines[0].get_data()[0]
        x_y = cc_fig.axes[0].lines[0].get_data()[1]

        y_x = cc_fig.axes[1].lines[0].get_data()[0]
        y_y = cc_fig.axes[1].lines[0].get_data()[1]

        x_ymax = np.max(x_y)
        y_ymax = np.max(y_y)

        plt.close(cc_fig)

        lines_x = axHistx.plot(x_x, x_y / x_ymax, color='C0')
        lines_y = axHisty.plot(y_y / y_ymax, y_x, color='C0')

        filename = f'walker2_biggrid2_V25_{num:04}.png'
        filepath = os.path.join(save_path, filename)
        fig.savefig(filepath)

        lines_scatter.pop(0).remove()
        lines_x.pop(0).remove()
        lines_y.pop(0).remove()
        cc.remove_chain()

        # fig.show()
        # return

    sys.stdout.write('')
    plt.close('all')
