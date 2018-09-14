import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from astropy import units
import subprocess
from scipy.stats import linregress

# pygrids
from pygrids.grids import grid_strings, grid_tools
from pygrids.kepler import kepler_tools

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']
PROJECT_PATH = '/home/zacpetej/projects/oscillations/'

# TODO: implement module as proper pygrids module

def extract_cycles(cycles, run, batch, source, basename='xrb',
                   prefix=''):
    """Iterates over dump cycles and extracts profiles and tables
    """
    save_times(cycles, run, batch, source=source, basename=basename, prefix=prefix)
    copy_lightcurve(run, batch, source=source, basename=basename)

    dashes()
    print('Extracting profiles')
    for i, cycle in enumerate(cycles):
        dump = kepler_tools.load_dump(cycle, run, batch, source=source,
                                      basename=basename, prefix=prefix)

        table = get_profile(dump)
        save_table(table, table_name=f'{cycle}', run=run, batch=batch,
                   source=source, basename=basename, subdir='profiles')


def get_profile(dump):
    """Extracts key profile quantities of given dump
    """
    table = pd.DataFrame()
    table['y'] = dump.y[1:-2]
    table['rho'] = dump.dn[1:-2]
    table['T'] = dump.tn[1:-2]
    table['P'] = dump.pn[1:-2]
    table['R'] = dump.rn[1:-2]
    return table


def save_times(cycles, run, batch, source, basename='xrb',
               prefix=''):
    dashes()
    print('Extracting cycle times')
    table = pd.DataFrame()
    table['timestep'] = cycles
    table['time (s)'] = extract_times(cycles, run, batch, source=source,
                                      basename=basename, prefix=prefix)

    save_table(table, table_name='timesteps', run=run, batch=batch,
               source=source, basename=basename)
    return table


def extract_times(cycles, run, batch, source, basename='xrb', prefix=''):
    """Returns timestep values (s) for given cycles
    """
    times = np.zeros(len(cycles))
    for i, cycle in enumerate(cycles):
        print_cycle_progress(cycle=cycle, cycles=cycles,
                             i=i, prefix='Getting cycle times: ')
        dump = kepler_tools.load_dump(cycle, run=run, batch=batch, source=source,
                                      basename=basename, prefix=prefix)
        times[i] = dump.time
    sys.stdout.write('\n')
    return times


def save_table(table, table_name, run, batch, source, basename='xrb',
               subdir=''):
    """Save provided table to oscillations project
    """
    source_path = os.path.join(PROJECT_PATH, source)
    batch_str = grid_strings.get_batch_string(batch, source)
    run_str = get_run_string(run, basename)

    run_path = os.path.join(source_path, batch_str, run_str, subdir)
    grid_tools.try_mkdir(run_path, skip=True)

    filename = f'{table_name}.txt'
    filepath = os.path.join(run_path, filename)
    print(f'Saving: {filepath}')

    table_str = table.to_string(index=False, justify='left')
    with open(filepath, 'w') as f:
        f.write(table_str)


def copy_lightcurve(run, batch, source, basename='xrb'):
    """Copies over full model lightcurve
    """
    dashes()
    print('Copying model lightcurve')

    path = grid_strings.get_source_subdir(source, 'burst_analysis')
    batch_str = grid_strings.get_batch_string(batch, source)
    run_str = get_run_string(run, basename)
    model_string = grid_strings.get_model_string(run=run, batch=batch,
                                                 source=source)

    filename = f'{model_string}.txt'
    filepath = os.path.join(path, batch_str, 'input', filename)

    target_filename = f'model_lightcurve_{model_string}'
    target_filepath = os.path.join(PROJECT_PATH, source, batch_str, run_str,
                                   target_filename)

    print(f'from: \n{filepath}'
          + f'to: {target_filepath}')
    subprocess.run(['cp', filepath, target_filepath])


def plot_temp_multi(cycles, runs, batches, sources, basename='xrb', prefix='',
                    fontsize=12, legend=True):
    """Plots Temp profiles of multiple different sources/batches/runs

    cycles,runs,batches,sources are arrays of length N, where the i'th entry
        correspond to a single model to plot
    """
    # TODO: auto use largest cycle common to all models
    runs, batches, sources = expand_lists(cycles, runs, batches, sources)
    fig, ax = plt.subplots()
    dump = None
    for i, cycle in enumerate(cycles):
        dump = kepler_tools.load_dump(cycle, runs[i], batches[i], source=sources[i],
                                      basename=basename, prefix=prefix)
        ax.plot(dump.y[1:-2], dump.tn[1:-2],
                label=f'{sources[i]}_{batches[i]}_{runs[i]}_#{cycle}')

    ax.set_yscale('log')
    ax.set_xscale('log')

    y0 = dump.y[1]   # column depth at inner zone
    y1 = dump.y[-3]  # outer zone
    ax.set_xlim([y1, y0])

    ax.set_xlabel(r'y (g cm$^{-2}$)', fontsize=fontsize)
    ax.set_ylabel(r'T (K)', fontsize=fontsize)
    if legend:
        ax.legend()
    plt.tight_layout()
    plt.show(block=False)


def plot_temp(run, batch, source, cycles=None, basename='xrb', title=None,
              display=True, prefix='', fontsize=14, marker='', relative=False,
              xlims=None, ylims=(5e7, 5e9), legend=True,
              yscale='log', xscale='log'):
    """Plot temperature profile at given cycle (timestep)
    """
    fig, ax = plt.subplots()
    cycles = kepler_tools.check_cycles(cycles, run=run, batch=batch, source=source)

    if relative:
        yscale = 'linear'
        d0 = kepler_tools.load_dump(0, run, batch, source=source, basename=basename,
                                    prefix=prefix)
        t0 = d0.tn[1:-1]
        i_end = len(t0) + 1
        ylabel = r'T - $T_{\#0}$ (K)'
    else:
        ax.set_ylim(ylims)
        t0 = 0.0
        i_end = -1
        ylabel = r'T o(K)'

    for cycle in cycles:
        dump = kepler_tools.load_dump(cycle, run, batch, source=source, basename=basename,
                                      prefix=prefix)
        ax.plot(dump.y[1:i_end], dump.tn[1:i_end]-t0, label=f'#{cycle}', marker=marker)

    if title is None:
        title = f'{source}_{batch}_{run}'
    ax.set_title(title)

    if xlims is not None:
        ax.set_xlim(xlims)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_xlabel(r'y (g cm$^{-2}$)', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if legend:
        ax.legend()
    plt.tight_layout()

    if display:
        plt.show(block=False)
    return fig


def save_temps(run, batch, source, zero_times=True, cycles=None, **kwargs):
    """Iterate through cycles and save temperature profile plots
    """
    batch_str = grid_strings.get_batch_string(batch, source)
    path = os.path.join(grid_strings.plots_path(source), 'temp', batch_str, str(run))
    grid_tools.try_mkdir(path, skip=True)

    cycles = kepler_tools.check_cycles(cycles, run=run, batch=batch, source=source)
    times = extract_times(cycles, run=run, batch=batch, source=source)

    if zero_times:
        times = times - times[0]

    for i, cycle in enumerate(cycles):
        print_cycle_progress(cycle=cycle, cycles=cycles, i=i, prefix='Saving plots: ')
        title = f'cycle={cycle},  t={times[i]:.6f}'
        fig = plot_temp(cycles=[cycle], run=run, batch=batch, source=source, title=title,
                        display=False, **kwargs)

        filename = f'temp_{source}_{batch}_{run}_{i:04}.png'
        filepath = os.path.join(path, filename)
        fig.savefig(filepath)
        plt.close('all')
    sys.stdout.write('\n')


def plot_base_temp_multi(runs, batches, sources, cycles=None, legend=True, linear=False,
                         temp_zone=20):
    fig, ax = plt.subplots()
    n = len(runs)
    if len(sources) == 1:
        sources = np.full(n, sources[0])  # assume all same source

    for i in range(n):
        plot_base_temp(run=runs[i], batch=batches[i], source=sources[i],
                       cycles=cycles, ax=ax, display=False, title=False, linear=linear,
                       temp_zone=temp_zone)
    if legend:
        ax.legend(loc='center left')
    plt.show(block=False)


def plot_base_temp(run, batch, source, cycles=None, basename='xrb', title=True,
                   display=True, ax=None, linear=False, temp_zone=20):
    if ax is None:
        fig, ax = plt.subplots()
    xscale = 3600
    temps = kepler_tools.extract_temps(run, batch, source, cycles=cycles,
                                       basename=basename, temp_zone=temp_zone)
    model_str = grid_strings.get_model_string(run, batch, source)
    ax.plot(temps[:, 0]/xscale, temps[:, 1], marker='o', label=model_str)

    if linear:
        i0 = 1 if len(temps) > 2 else 0  # skip first dump if possible
        linr = linregress(temps[i0:, 0], temps[i0:, 1])
        x = np.array([temps[0, 0], temps[-1, 0]])
        y = linr[0] * x + linr[1]
        ax.plot(x/xscale, y)

    ax.set_ylabel(r'T (K)')
    ax.set_xlabel('time (hr)')

    if title:
        ax.set_title(f'{source}_{batch}_{run}')
    plt.tight_layout()
    if display:
        plt.show(block=False)


def plot_saxj(x_units='time', dumptimes=True, cycles=None):
    """Plotting SAXJ1808 model, to explore dumpfiles
    to try and get temperature profiles"""
    filepath = '/home/zacpetej/archive/kepler/grid_94/xrb2/preload2.txt'
    lc = np.loadtxt(filepath, skiprows=1)
    tscale = 1
    dump_nums = np.arange(len(lc))

    fig, ax = plt.subplots()
    if x_units == 'time':
        ax.plot(lc[:, 0]/tscale, lc[:, 1], marker='o', markersize=2)
    else:
        ax.plot(dump_nums, lc[:, 1], marker='o', markersize=2)

    if dumptimes:
        dumps = np.arange(1, 51) * 1000
        if x_units == 'time':
            ax.plot(lc[dumps, 0]/tscale, lc[dumps, 1], marker='o', ls='none')
        else:
            if x_units == 'time':
                ax.plot(dumps, lc[dumps, 1], marker='o', ls='none')

    if cycles is not None:
        ax.plot(lc[cycles, 0], lc[cycles, 1], marker='o', ls='none')

    plt.show(block=False)


def get_mean_qnuc(cycles, run, batch, source):
    """Return energy generation per mass averaged over model (erg/g)
    cycles: length 2 array
    """
    dumps = []
    for i, cycle in enumerate(cycles):
        dumps += [kepler_tools.load_dump(cycle, run=run, batch=batch, source=source)]

    mass_diff = dumps[1].qparm('xmacc') - dumps[0].qparm('xmacc')
    energy_diff = dumps[1].qparm('epro') - dumps[0].qparm('epro')
    rate = energy_diff / mass_diff
    rate_mev = rate * (units.erg/units.g).to(units.MeV/units.M_p)
    print(f'{rate_mev:.2f}  MeV/nucleon')
    return rate


def expand_lists(cycles, runs, batches, sources):
    lists = []
    n_cyc = len(cycles)
    for i, var in enumerate([runs, batches, sources]):
        if len(var) == 1:
            lists += [np.full(n_cyc, var[0])]
        else:
            lists += [var]
    return lists


def print_cycle_progress(cycle, cycles, i, prefix=''):
    sys.stdout.write(f'\r{prefix}cycle {cycle}/{cycles[-1]} '
                     f'({(i+1) / len(cycles) * 100:.1f}%)')


def get_run_string(run, basename='xrb'):
    return f'{basename}{run}'


def dashes():
    print('=' * 40)

