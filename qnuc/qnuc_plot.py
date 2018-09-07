import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# pygrids
from . import qnuc_tools
from pygrids.grids import grid_analyser
from pygrids.plotting.plotting_tools import set_axes


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

