import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# pygrids
from . import qnuc_tools
from pygrids.grids import grid_analyser, grid_tools
from pygrids.plotting.plotting_tools import set_axes


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