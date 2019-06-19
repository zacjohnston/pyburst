import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

# TODO:

def plot_hist(table, var='feh', bins=100, histtype='step', display=True,
              values=None):
    """Plots histogram of the given table variable
    """
    xlabels = {'feh': '[Fe/H]'}

    if values is None:
        values = table[var]

    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, density=1, histtype=histtype)

    xlabel = xlabels.get(var, var)
    ax.set_xlabel(xlabel)

    if display:
        plt.show(block=False)
    return fig, ax


def fit_gaussian(table, plot=False, xlims=(-3, 3)):
    """Returns Gaussian fit (mean, std) to a given [Fe/H] distribution

    table : pd.DataFrame
        table containing a column of 'feh'
    plot : bool
        show Gaussian fit against distribution histogram
    """
    mean, std = norm.fit(table['feh'])

    if plot:
        x = np.linspace(xlims[0], xlims[1], 200)
        y = norm.pdf(x, mean, std)

        fig, ax = plot_hist(table=table, var='feh')
        ax.plot(x, y)
        plt.show(block=False)

    return mean, std


def fit_beta(table, plot=False, xlims=(-2, 0.5)):
    """Returns fit of Beta Distribution to a given [Fe/H] table
    See: fit_gaussian()
    """
    z_sort = np.sort(table['feh'])
    i_0 = np.searchsorted(z_sort, xlims[0])
    i_1 = np.searchsorted(z_sort, xlims[1])

    loc = xlims[0]
    scale = xlims[1] - xlims[0]

    a, b, loc, scale = beta.fit(z_sort[i_0:i_1], floc=loc, fscale=scale)

    if plot:
        x = np.linspace(xlims[0], xlims[1], 100)
        y = beta.pdf(x, a, b, loc=loc, scale=scale)

        fig, ax = plot_hist(table=table, var='feh', values=z_sort)
        ax.plot(x, y)
        plt.show(block=False)

    return a, b, loc, scale
