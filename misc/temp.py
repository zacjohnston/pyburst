import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import kepdump


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


def load_dump(cycle, run=2, basename='xrb'):
    path = '/home/zacpetej/archive/kepler/grid_94'
    run_str = f'{basename}{run}'
    filename = f'{run_str}#{cycle}'
    filepath = os.path.join(path, run_str, filename)

    return kepdump.load(filepath)


def extract_dump(cycle, run=2, basename='xrb'):
    """Extracts key quantities of dump profile and saves as table
    """
    table = pd.DataFrame()

    dump = load_dump(cycle, run=run, basename=basename)
    table['y'] = dump.y[1:-2]
    table['rho'] = dump.dn[1:-2]
    table['T'] = dump.tn[1:-2]
    table['P'] = dump.pn[1:-2]
    table['R'] = dump.rn[1:-2]

    return table


def extract_cycles(cycles):
    """Iterates over dump cycles and saves each as a table
    """
    pass


def plot_temp(cycle, run=2, basename='xrb'):
    """Plot temperature profile at given cycle (timestep)
    """
    dump = load_dump(cycle, run=run, basename=basename)
    y0 = dump.y[1]  # column depth at inner zone

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlim([1e5, y0])
    ax.set_ylim([5e7, 5e9])

    ax.set_xlabel(r'y (g cm$^{-2}$)')
    ax.set_ylabel(r'T (K)')

    ax.plot(dump.y, dump.tn, color='C1')
    plt.show(block=False)
