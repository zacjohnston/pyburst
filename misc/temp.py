import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess

import kepdump
from pygrids.grids import grid_strings, grid_tools

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']
PROJECT_PATH = '/home/zacpetej/projects/oscillations/'


def extract_cycles(cycles, run, batch, source='biggrid2', basename='xrb',
                   prefix='re_'):
    """Iterates over dump cycles and extracts profiles and tables
    """
    save_times(cycles, run, batch, source=source, basename=basename, prefix=prefix)
    copy_lightcurve(run, batch, source=source, basename=basename)

    dashes()
    print('Extracting profiles')
    for i, cycle in enumerate(cycles):
        pre = get_prefix(i, prefix)
        dump = load_dump(cycle, run, batch, source=source,
                         basename=basename, prefix=pre)

        table = get_profile(dump)
        save_table(table, table_name=f'{cycle}', run=run, batch=batch,
                   source=source, basename=basename, subdir='profiles')


def load_dump(cycle, run, batch, source='biggrid2', basename='xrb',
              prefix='re_'):
    batch_str = get_batch_string(batch, source)
    run_str = get_run_string(run, basename)
    filename = get_dump_filename(cycle, run, basename, prefix=prefix)

    filepath = os.path.join(MODELS_PATH, batch_str, run_str, filename)
    return kepdump.load(filepath)


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
               prefix='re_'):
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
                  prefix='re_'):
    """Returns timestep values (s) for given cycles
    """
    times = np.zeros_like(cycles, dtype=float)

    for i, cycle in enumerate(cycles):
        pre = get_prefix(i, prefix)
        dump = load_dump(cycle, run, batch, source=source, basename=basename,
                         prefix=pre)
        times[i] = dump.time

    return times


def save_table(table, table_name, run, batch, source='biggrid2', basename='xrb',
               subdir=''):
    """Save provided table to oscillations project
    """
    source_path = os.path.join(PROJECT_PATH, source)
    batch_str = get_batch_string(batch, source)
    run_str = get_run_string(run, basename)

    run_path = os.path.join(source_path, batch_str, run_str, subdir)
    grid_tools.try_mkdir(run_path, skip=True)

    filename = f'{table_name}.txt'
    filepath = os.path.join(run_path, filename)
    print(f'Saving: {filepath}')

    table_str = table.to_string(index=False, justify='left', col_space=12)
    with open(filepath, 'w') as f:
        f.write(table_str)


def copy_lightcurve(run, batch, source='biggrid2', basename='xrb'):
    """Copies over full model lightcurve
    """
    dashes()
    print('Copying model lightcurve')

    path = grid_strings.get_source_subdir(source, 'burst_analysis')
    batch_str = get_batch_string(batch, source)
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


def plot_temp(cycle, run, batch, source='biggrid2', basename='xrb', title='',
              display=True):
    """Plot temperature profile at given cycle (timestep)
    """
    dump = load_dump(cycle, run, batch, source=source,
                     basename=basename)
    y0 = dump.y[1]  # column depth at inner zone

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlim([1e5, y0])
    ax.set_ylim([5e7, 5e9])

    ax.set_xlabel(r'y (g cm$^{-2}$)')
    ax.set_ylabel(r'T (K)')

    ax.plot(dump.y, dump.tn, color='C3')

    if display:
        plt.show(block=False)

    return fig


def save_temps(cycles, zero_times=True):
    """Iterate through cycles and save temperature profile plots
    """
    path = grid_strings.get_source_subdir('saxj1808', 'plots')
    times = extract_times(cycles)

    if zero_times:
        times = times - times[0]

    for i, cycle in enumerate(cycles):
        print(f'Cycle {cycle}')
        title = f'cycle={cycle},  t={times[i]:.6f}'
        fig = plot_temp(cycle, title=title, display=False)

        filename = f'temp_{i:02}.png'
        filepath = os.path.join(path, 'temp', filename)
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


def get_prefix(index, prefix):
    """Accounts for first cycle dump having no prefix
    """
    return {-1: ''}.get(index, prefix)


def get_batch_string(batch, source):
    return f'{source}_{batch}'


def get_run_string(run, basename='xrb'):
    return f'{basename}{run}'


def get_dump_filename(cycle, run, basename, prefix='re_'):
    return f'{prefix}{basename}{run}#{cycle}'


def dashes():
    print('=' * 40)
