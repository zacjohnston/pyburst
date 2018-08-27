import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import astropy.units as u
from scipy.stats import linregress

#kepler
import kepdump

#pygrids
from pygrids.grids import grid_analyser, grid_strings, grid_tools

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']
PROJECT_PATH = '/home/zacpetej/projects/oscillations/'


def extract_cycles(cycles, run, batch, source='biggrid2', basename='xrb',
                   prefix=''):
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
              prefix=''):
    batch_str = get_batch_string(batch, source)
    run_str = get_run_string(run, basename)
    filename = get_dump_filename(cycle, run, basename, prefix=prefix)

    filepath = os.path.join(MODELS_PATH, batch_str, run_str, filename)
    return kepdump.load(filepath, graphical=False, silent=True)


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

    table_str = table.to_string(index=False, justify='left')
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


def plot_temp_multi(cycles, runs, batches, sources, basename='xrb', prefix='', legend=True):
    """Plots Temp profiles of multiple different sources/batches/runs

    cycles,runs,batches,sources are arrays of length N, where the i'th entry
        correspond to a single model to plot
    """
    runs, batches, sources = expand_lists(cycles, runs, batches, sources)
    fig, ax = plt.subplots()
    dump = None
    for i, cycle in enumerate(cycles):
        dump = load_dump(cycle, runs[i], batches[i], source=sources[i],
                         basename=basename, prefix=prefix)
        ax.plot(dump.y, dump.tn, label=f'{sources[i]}_{batches[i]}_{runs[i]}_#{cycle}')

    ax.set_yscale('log')
    ax.set_xscale('log')

    y0 = dump.y[1]   # column depth at inner zone
    y1 = dump.y[-3]  # outer zone
    ax.set_xlim([y1, y0])
    # ax.set_ylim([5e7, 5e9])

    ax.set_xlabel(r'y (g cm$^{-2}$)')
    ax.set_ylabel(r'T (K)')
    if legend:
        ax.legend()
    plt.show(block=False)


def plot_temp(cycles, run, batch, source='biggrid2', basename='xrb', title='',
              display=True, prefix=''):
    """Plot temperature profile at given cycle (timestep)
    """
    fig, ax = plt.subplots()

    for cycle in cycles:
        dump = load_dump(cycle, run, batch, source=source, basename=basename, prefix=prefix)
        ax.plot(dump.y, dump.tn, label=f'#{cycle}')  # color='C3')

    bookends = cycles[[0, -1]]
    temp = [0, 0]

    for i, cyc in enumerate(bookends):
        dump = load_dump(cyc, run, batch, source=source, basename=basename, prefix=prefix)
        temp[i] = dump.tn[1]

    print(f'{temp[1]/temp[0]:.3f}')

    y0 = dump.y[1]   # column depth at inner zone
    y1 = dump.y[-3]  # outer zone

    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlim([y1, y0])
    ax.set_ylim([5e7, 5e9])

    ax.set_xlabel(r'y (g cm$^{-2}$)')
    ax.set_ylabel(r'T (K)')
    ax.legend()

    if display:
        plt.show(block=False)

    return fig


def plot_base_temp(cycles, run, batch, source='biggrid2', basename='xrb', title='',
                   display=True, prefix=''):
    fig, ax = plt.subplots()
    temps = np.zeros_like(cycles)
    times = np.zeros_like(cycles)

    for i, cycle in enumerate(cycles):
        dump = load_dump(cycle, run, batch, source=source, basename=basename, prefix=prefix)
        times[i] = dump.time
        temps[i] = dump.tn[1]
        if i == 0:
            t0 = dump.time/3600
            temp0 = dump.tn[1]

    t1 = dump.time/3600
    temp1 = dump.tn[1]
    slope = (temp1 - temp0) / (t1 - t0)
    days = 1e8 / (24 * np.abs(slope))
    print(f'{run}     {slope:.2e} (K/hr)     {days:.1f} (days)')
    ax.plot(times/3600, temps, marker='o')
    # ax.set_yscale('log')
    ax.set_ylabel(r'T (K)')
    ax.set_xlabel('time (hr)')
    # ax.set_ylim([2e8, 4e8])
    ax.set_title(f'{source}_{batch}_{run}')
    plt.tight_layout()
    if display:
        plt.show(block=False)
    else:
        plt.close()

    return slope


def plot_slope(cycles, source, params, linear=True, display=True):
    kgrid = grid_analyser.Kgrid(source)
    subset = kgrid.get_params(params=params)
    slopes = get_slopes(cycles, table=subset, source=source)

    fig, ax = plt.subplots()
    x = subset['accrate'].iloc[[0, -1]]
    ax.plot(subset['accrate'], slopes, ls='none', marker='o')
    ax.plot(x, [0, 0], color='black')

    if linear:
        linr = linregress(subset['accrate'], slopes)
        ax.plot(x, x * linr[0] + linr[1])
        print(f'{linr[0]:.3f}   {linr[1]:.2f}')
    if display:
        plt.show(block=False)
    else:
        plt.close()


def get_slopes(cycles, table, source):
    """Returns slopes of base temperature change (K/s), for given model table
    """
    slopes = []
    for row in table.itertuples():
        d0 = load_dump(cycles[0], run=row.run, batch=row.batch, source=source)
        d1 = load_dump(cycles[1], run=row.run, batch=row.batch, source=source)
        slopes += [(d1.tn[1] - d0.tn[1]) / (d1.time - d0.time)]
    return np.array(slopes)


def get_qnuc(cycles, run, batch, source):
    """Return energy generation per mass averaged over model (erg/g)
    cycles: length 2 array
    """
    dumps = []
    for i, cycle in enumerate(cycles):
        # between two dumps:
        #   get energy produced (q epro)
        #   get mass accreted (q xmacc)
        #   calculate energy per mass accreted (erg/g)
        dumps += [load_dump(cycle, run=run, batch=batch, source=source)]

    mass_diff = dumps[1].qparm('xmacc') - dumps[0].qparm('xmacc')
    energy_diff = dumps[1].qparm('epro') - dumps[0].qparm('epro')
    rate = energy_diff / mass_diff
    rate_mev = rate * (u.erg/u.g).to(u.MeV/u.M_p)
    print(f'{rate_mev:.2f}  MeV/nucleon')
    return rate


def save_temps(cycles, run, batch, source, zero_times=True):
    """Iterate through cycles and save temperature profile plots
    """
    batch_str = get_batch_string(batch, source)
    path = grid_strings.get_source_subdir(source, 'plots')
    path = os.path.join(path, 'temp', batch_str, str(run))
    grid_tools.try_mkdir(path, skip=True)

    times = extract_times(cycles, run, batch)

    if zero_times:
        times = times - times[0]

    for i, cycle in enumerate(cycles):
        print(f'Cycle {cycle}')
        title = f'cycle={cycle},  t={times[i]:.6f}'
        fig = plot_temp([cycle], run, batch, source=source,
                        title=title, display=False)

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
