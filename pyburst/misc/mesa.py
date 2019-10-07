import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# pyburst
from pyburst.grids import grid_analyser
from pyburst.burst_analyser import burst_analyser
from pyburst.physics import gravity
from pyburst.plotting import plot_tools

"""
Module for comparing Kepler and Mesa models

Currently very hacky and roughshod
"""

def plot(model_set, actual_mdot=True, qnuc=0.0, verbose=True,
         bprops=('rate', 'fluence', 'peak'), display=True, grid_version=0):
    """Plot predefined set of mesa model comparisons

    model_set : int
        ID for set of models (defined below)
    """
    mesa_info = get_mesa_set(model_set)

    if actual_mdot:
        mdots = mesa_info['mdots_actual']
    else:
        mdots = mesa_info['mdots']

    mesa_info['params']['qnuc'] = qnuc

    fig, ax = plot_compare(mesa_runs=mesa_info['runs'], display=display,
                           mesa_mdots=mdots, bprops=bprops,
                           params=mesa_info['params'], verbose=verbose,
                           grid_version=grid_version)

    return fig, ax


def plot_all_avg_lightcurves(mesa_set, grid_source='mesa',
                             display=True, verbose=True):
    """Plots all average lightcurve comparisons for a given set
    """
    mesa_info = get_mesa_set(mesa_set)
    kgrid = grid_analyser.Kgrid(source=grid_source, verbose=verbose)

    # ===== Setup plotting =====
    n_models = len(mesa_info['runs'])
    n_rows = int(np.ceil(n_models / 2))

    subplot_width = 4
    subplot_height = 2.5
    figsize = (2 * subplot_width, n_rows * subplot_height)

    fig, ax = plt.subplots(n_rows, 2, figsize=figsize)

    if n_models % 2 == 1:
        ax[-1, -1].axis('off')

    for i, run in enumerate(mesa_info['runs']):
        row_i = int(np.floor(i / 2))
        col_i = i % 2

        kep_run = mesa_info['kep_runs'][i]
        kep_batch = mesa_info['kep_batches'][i]

        plot_avg_lc(mesa_run=run, grid_run=kep_run,
                    grid_batch=kep_batch, grid_source='mesa',
                    ax=ax[row_i, col_i], display=False, kgrid=kgrid,
                    legend=True if i == 0 else False,
                    verbose=verbose)

    if display:
        plt.show(block=False)
    return fig, ax


def get_mesa_set(mesa_set):
    """Returns model info for a given mesa set
    """
    mesa_runs = {
        1: [1, 2, 4, 5, 6],
        2: [7, 8, 9, 10, 11],
        3: [16, 17, 18],
        4: [37, 38, 39, 40, 42],
        5: [73, 74, 78],
        6: np.arange(79, 85),
    }
    mesa_mdots = {
        1: [0.051, 0.069, 0.111, 0.15, 0.17],
        2: [0.051, 0.069, 0.079, 0.111, 0.15],
        3: [0.111, 0.15, 0.17],
        4: [0.051, 0.069, 0.079, 0.111, 0.17],
        5: [0.051, 0.069, 0.17],
        6: [0.051, 0.069, 0.079, 0.111, 0.15, 0.17],
    }
    mesa_mdots_actual = {
        1: [0.061, 0.079, 0.123, 0.164, 0.185],
        2: [0.065, 0.088, 0.09, 0.123, 0.164],
        3: [0.128, 0.171, 0.192],
        4: [0.066, 0.081, 0.092, 0.125, 0.202],
        5: [0.061, 0.083, 0.186],
        6: [0.078, 0.084, 0.098, 0.129, 0.17, 0.192],
    }
    params = {
        1: {'x': 0.7, 'z': 0.02, 'qb': 0.1},
        2: {'x': 0.7, 'z': 0.01, 'qb': 0.1},
        3: {'x': 0.7, 'z': 0.01, 'qb': 1.0},
        4: {'x': 0.7, 'z': 0.01, 'qb': 0.5},
        5: {'x': 0.75, 'z': 0.02, 'qb': 0.1},
        6: {'x': 0.75, 'z': 0.02, 'qb': 1.0},
    }
    kep_runs = {
        1: np.full(5, 4),
        2: np.full(5, 1),
        3: np.full(3, 3),
        4: np.full(5, 2),
        5: np.full(3, 10),
        6: np.full(6, 12),
    }
    kep_batches = {
        1: [2, 4, 8, 10, 12],
        2: [2, 4, 6, 8, 10],
        3: [8, 10, 12],
        4: [2, 4, 6, 8, 12],
        5: [2, 4, 12],
        6: [2, 4, 6, 8, 10, 12],
    }

    return {
        'runs': mesa_runs[mesa_set],
        'mdots': mesa_mdots[mesa_set],
        'mdots_actual': mesa_mdots_actual[mesa_set],
        'params': params[mesa_set],
        'kep_runs': kep_runs[mesa_set],
        'kep_batches': kep_batches[mesa_set],
    }


def plot_avg_lc(mesa_run, grid_run, grid_batch, grid_source='mesa',
                radius=10, mass=1.4, shaded=True, ax=None,
                display=True, legend=True, kgrid=None,
                verbose=True):
    """Plots comparison of average lightcurves from mesa
    """
    xi, redshift = gravity.gr_corrections(r=radius, m=mass, phi=1)
    lum_f = unit_factor('peak')

    if kgrid is None:
        kgrid = grid_analyser.Kgrid(source=grid_source, verbose=verbose)
    if ax is None:
        fig, ax = plt.subplots(figsize=[6, 4])

    mesa_model = setup_analyser(mesa_run, verbose=verbose)
    kgrid.load_mean_lightcurves(grid_batch)

    mesa_lc = mesa_model.average_lightcurve()
    kep_lc = kgrid.mean_lc[grid_batch][grid_run]

    mesa_time = mesa_lc[:, 0]
    mesa_lum = mesa_lc[:, 1] / lum_f
    mesa_u_lum = mesa_lc[:, 2] / lum_f

    kep_time = kep_lc[:, 0]
    kep_lum = kep_lc[:, 1] * xi**2 / lum_f
    kep_u_lum = kep_lc[:, 2] * xi**2 / lum_f

    if shaded:
        ax.fill_between(kep_time, y1=kep_lum + kep_u_lum,
                        y2=kep_lum - kep_u_lum,
                        color='C0', alpha=0.5)

        ax.fill_between(mesa_time, y1=mesa_lum + mesa_u_lum,
                        y2=mesa_lum - mesa_u_lum,
                        color='C1', alpha=0.5)

    ax.plot(kep_time, kep_lum, label='kepler', color='C0')
    ax.plot(mesa_time, mesa_lum, label='mesa', color='C1')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(plot_tools.full_label('lum'))
    if legend:
        ax.legend()

    plt.tight_layout()
    if display:
        plt.show(block=False)


def plot_compare(mesa_runs, mesa_mdots, grid_source='mesa',
                 params=None, bprops=('rate', 'fluence', 'peak'),
                 mass=1.4, radius=10, verbose=True, display=True,
                 grid_version=0):
    """Plot comparison of mesa and kepler models
    """
    if params is None:
        print('Using default params')
        params = {'x': 0.7, 'z': 0.02, 'qb': 0.1, 'qnuc': 0.0}

    kgrid = grid_analyser.Kgrid(source=grid_source, verbose=verbose,
                                grid_version=grid_version)
    grid_summ = kgrid.get_summ(params=params)
    grid_params = kgrid.get_params(params=params)

    xi, redshift = gravity.gr_corrections(r=radius, m=mass, phi=1)

    mesa_models = extract_bprops(runs=mesa_runs, mesa_mdots=mesa_mdots,
                                 verbose=verbose)

    n_bprops = len(bprops)
    figsize = (4.5, 2.5*n_bprops)
    fig, ax = plt.subplots(n_bprops, 1, figsize=figsize, sharex='all')

    for i, bprop in enumerate(bprops):
        u_bprop = f'u_{bprop}'

        unit_f = unit_factor(bprop)
        gr_f = gr_correction(bprop, xi=xi, redshift=redshift)

        # === kepler model ===
        ax[i].errorbar(grid_params['accrate']*xi**2,
                       grid_summ[bprop]*gr_f/unit_f,
                       yerr=grid_summ[u_bprop]*gr_f/unit_f, marker='o',
                       capsize=3, label='kepler')

        # === mesa model ===
        ax[i].errorbar(mesa_models['accrate'], mesa_models[bprop]/unit_f,
                       yerr=mesa_models[u_bprop]/unit_f, marker='o',
                       capsize=3, label='mesa')

        ylabel = plot_tools.full_label(bprop)
        ax[i].set_ylabel(ylabel)

    ax[0].legend()
    ax[-1].set_xlabel(plot_tools.full_label('mdot'))

    plt.tight_layout()
    if display:
        plt.show(block=False)

    return fig, ax


def unit_factor(bprop):
    """Returns scaling factor
    """
    if bprop == 'dt':
        return 3600
    else:
        return plot_tools.unit_scale(bprop)


def gr_correction(bprop, xi, redshift):
    """Returns gr correction factor to convert from kepler to mesa frame
    """
    corrections = {
        'rate': 1,  # mesa time in observer coordinate?
        'dt': 1,
        'fluence': xi**2,
        'peak': xi**2,
    }
    return corrections[bprop]


def model_filepath(run):
    """Returns str filepath to mesa model lightcurve
    """
    path = '/home/zac/projects/kepler_grids/obs_data/mesa/model_lc'
    filename = f'LCTrainVar{run}.txt'
    return os.path.join(path, filename)


def load_model_lc(run):
    """Loads and returns np.array of mesa model lightcurve
    """
    filepath = model_filepath(run)
    return np.loadtxt(filepath, skiprows=2)


def setup_analyser(run, verbose=True):
    """Sets up burst_analyser with haxx
    """
    lc = load_model_lc(run)
    lc[:, 0] *= 3600
    model = burst_analyser.BurstRun(run=run, batch=1, source='meisel',
                                    load_lum=False, analyse=False,
                                    truncate_edd=False,
                                    load_model_params=False,
                                    verbose=verbose)

    # inject lightcurve
    model.lum = lc
    model.setup_lum_interpolator()

    model.flags['lum_loaded'] = True
    model.options['load_model_params'] = False

    model.analyse()
    return model

def extract_bprops(runs, mesa_mdots, verbose=True,
                   bprops=('dt', 'rate', 'fluence', 'peak')):
    """Quick and dirty extraction of burst properties from Mesa models
    """
    n_models = len(runs)
    models = pd.DataFrame()
    models['accrate'] = mesa_mdots

    summ = dict.fromkeys(bprops)
    for key in bprops:
        u_key = f'u_{key}'
        summ[key] = np.full(n_models, np.nan)
        summ[u_key] = np.full(n_models, np.nan)

    for i, run in enumerate(runs):
        model = setup_analyser(run, verbose=verbose)

        for bprop in bprops:
            u_bprop = f'u_{bprop}'
            summ[bprop][i] = model.summary[bprop]
            summ[u_bprop][i] = model.summary[u_bprop]

    for key in summ:
        models[key] = summ[key]

    return models
