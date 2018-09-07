import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
from scipy.stats import linregress

# pygrids
from pygrids.grids import grid_analyser, grid_strings, grid_tools
from pygrids.kepler import kepler_tools
from pygrids.qnuc import qnuc_tools

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']
PROJECT_PATH = '/home/zacpetej/projects/oscillations/'


def extract_cycles(cycles, run, batch, source='biggrid2', basename='xrb',
                   prefix=''):
    """Iterates over dump cycles and extracts profiles and tables
    """
    # TODO: is prefix still necessary?
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


def save_times(cycles, run, batch, source='biggrid2', basename='xrb',
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


def extract_times(cycles, run, batch, source='biggrid2', basename='xrb',
                  prefix=''):
    """Returns timestep values (s) for given cycles
    """
    times = np.zeros_like(cycles, dtype=float)

    for i, cycle in enumerate(cycles):
        dump = kepler_tools.load_dump(cycle, run, batch, source=source, basename=basename,
                                      prefix=prefix)
        times[i] = dump.time

    return times


def save_table(table, table_name, run, batch, source='biggrid2', basename='xrb',
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


def copy_lightcurve(run, batch, source='biggrid2', basename='xrb'):
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


def plot_temp(run, batch, source, cycles=None, basename='xrb', title='',
              display=True, prefix='', fontsize=14):
    """Plot temperature profile at given cycle (timestep)
    """
    fig, ax = plt.subplots()
    dump = None
    if cycles is None:
        cycles = kepler_tools.get_cycles(run, batch, source)
    for cycle in cycles:
        dump = kepler_tools.load_dump(cycle, run, batch, source=source, basename=basename,
                                      prefix=prefix)
        ax.plot(dump.y[1:-1], dump.tn[1:-1], label=f'#{cycle}')  # color='C3')

    bookends = cycles[[0, -1]]
    temp = [0, 0]

    for i, cyc in enumerate(bookends):
        dump = kepler_tools.load_dump(cyc, run, batch, source=source, basename=basename,
                                      prefix=prefix)
        temp[i] = dump.tn[1]

    y0 = dump.y[1]   # column depth at inner zone
    y1 = dump.y[-3]  # outer zone

    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlim([y1, y0])
    ax.set_ylim([5e7, 5e9])

    ax.set_xlabel(r'y (g cm$^{-2}$)', fontsize=fontsize)
    ax.set_ylabel(r'T (K)', fontsize=fontsize)
    ax.legend()

    if display:
        plt.show(block=False)
    return fig


def plot_base_temp_multi(runs, batches, sources, cycles=None, legend=True, linear=False):
    fig, ax = plt.subplots()

    n = len(runs)
    if len(sources) == 1:
        sources = np.full(n, sources[0])  # assume all same source

    for i in range(n):
        plot_base_temp(run=runs[i], batch=batches[i], source=sources[i],
                       cycles=cycles, ax=ax, display=False, title=False, linear=linear)
    if legend:
        ax.legend(loc='center left')
    plt.show(block=False)


def plot_base_temp(run, batch, source='biggrid2', cycles=None, basename='xrb', title=True,
                   display=True, ax=None, linear=False):
    if ax is None:
        fig, ax = plt.subplots()
    xscale = 3600
    temps = kepler_tools.extract_base_temps(run, batch, source, cycles=cycles, basename=basename)
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


def plot_slope(source, params, xaxis='qnuc', cycles=None, linear=True, display=True):
    """xaxis : ['accrate', 'qnuc']
    """
    xlabel = {'accrate': '$\dot{M} / \dot{M}_\mathrm{Edd}$',
              'qnuc': '$Q_\mathrm{nuc}$'}.get(xaxis, xaxis)
    kgrid = grid_analyser.Kgrid(source)
    subset = kgrid.get_params(params=params)
    slopes = qnuc_tools.get_slopes(table=subset, source=source, cycles=cycles)

    fig, ax = plt.subplots()
    ax.plot(subset[xaxis], slopes, ls='none', marker='o')
    x = np.array((4, 9))
    ax.plot(x, [0, 0], color='black')
    set_axes(ax, xlabel=xlabel, ylabel='dT/dt (K s$^{-1}$)', title=params)

    if linear:
        linr = linregress(subset[xaxis], slopes)
        ax.plot(x, x * linr[0] + linr[1])
    if display:
        plt.show(block=False)
    else:
        plt.close()


def plot_bprops(source, params, bprop='dt'):
    """Plots burst property versus qnuc
    """
    kgrid = grid_analyser.Kgrid(source)
    sub_p = kgrid.get_params(params=params)
    sub_s = kgrid.get_summ(params=params)

    fig, ax = plt.subplots()
    ax.errorbar(sub_p['qnuc'], sub_s[bprop], yerr=sub_s[f'u_{bprop}'],
                ls='None', marker='o', capsize=3)
    ax.set_xlabel('$Q_\mathrm{nuc}$')
    ax.set_ylabel(bprop)
    plt.show(block=False)


def plot_qnuc(source, mass, linear=True):
    table = qnuc_tools.load_qnuc_table(source)
    table = grid_tools.reduce_table(table, params={'mass': mass})
    acc_unique = np.unique(table['accrate'])
    sub_params = grid_tools.reduce_table(table, params={'accrate': acc_unique[0]})

    fig, ax = plt.subplots()
    for row in sub_params.itertuples():
        models = grid_tools.reduce_table(table, params={'x': row.x, 'z': row.z})
        ax.plot(models['accrate'], models['qnuc'], marker='o',
                label=f'x={row.x:.2f}, z={row.z:.4f}')
    if linear:
        linr_table = qnuc_tools.linregress_qnuc(source)
        row = linr_table[linr_table['mass'] == mass]
        x = np.array([0.1, 0.2])
        y = row.m.values[0] * x + row.y0.values[0]
        ax.plot(x, y, color='black', ls='--')

    ax.set_title(f'mass={mass:.1f}')
    ax.legend()
    plt.show(block=False)


def extract_qnuc_table(source, ref_table, cycles=None):
    """Extracts optimal Qnuc across parameters

    ref_table : pd.DataFrame
        table covering all unique parameters (x, z, accrate, mass)
    """
    qnuc_table = qnuc_tools.iterate_solve_qnuc(source, ref_table, cycles=cycles)
    qnuc_tools.save_qnuc_table(qnuc_table, source)


def save_temps(cycles, run, batch, source, zero_times=True):
    """Iterate through cycles and save temperature profile plots
    """
    batch_str = grid_strings.get_batch_string(batch, source)
    path = grid_strings.get_source_subdir(source, 'plots')
    path = os.path.join(path, 'temp', batch_str, str(run))
    grid_tools.try_mkdir(path, skip=True)

    times = extract_times(cycles, run, batch)

    if zero_times:
        times = times - times[0]

    for i, cycle in enumerate(cycles):
        print(f'Cycle {cycle}')
        title = f'cycle={cycle},  t={times[i]:.6f}'
        fig = plot_temp([cycle], run, batch, source=source, title=title, display=False)

        filename = f'temp_{source}_{batch}_{run}_{i:02}.png'
        filepath = os.path.join(path, filename)
        fig.savefig(filepath)
        plt.close('all')


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


def expand_lists(cycles, runs, batches, sources):
    lists = []
    n_cyc = len(cycles)
    for i, var in enumerate([runs, batches, sources]):
        if len(var) == 1:
            lists += [np.full(n_cyc, var[0])]
        else:
            lists += [var]
    return lists


def get_run_string(run, basename='xrb'):
    return f'{basename}{run}'


def dashes():
    print('=' * 40)


def set_axes(ax, title='', xlabel='', ylabel='', yscale='linear', xscale='linear',
             fontsize=14, yticks=True, xticks=True):
    if not yticks:
        ax.axes.tick_params(axis='both', left='off', labelleft='off')
    if not xticks:
        ax.axes.tick_params(axis='both', bottom='off', labelbottom='off')

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
