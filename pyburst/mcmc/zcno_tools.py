import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

# TODO:

def plot_fit(table, fit='beta', var='feh', bins=100, hisstype='step', display=True,
             xlims=(-3.5, 1.0), func=None):
    """Plots fit of distribution to given table

    fit : str
        one of 'gaussian' or 'beta'
    """
    if func is None:
        if fit == 'gaussian':
            loc, scale = fit_gaussian(table, xlims=xlims)
            func = norm(loc=loc, scale=scale).pdf
        elif fit == 'beta':
            a, b, loc, scale = fit_beta(table, xlims=xlims)
            func = beta(a, b, loc=loc, scale=scale).pdf

    x = np.linspace(xlims[0], xlims[1], 200)
    y = func(x)

    fig, ax = plot_hist(table=table, var=var, bins=bins, histtype=hisstype)
    ax.plot(x, y, label=fit)
    ax.legend()

    if display:
        plt.show(block=False)

    return fig, ax


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


def fit_gaussian(table, xlims=(-3, 3)):
    """Returns Gaussian fit (mean, std) to a given [Fe/H] distribution

    table : pd.DataFrame
        table containing a column of 'feh'
    plot : bool
        show Gaussian fit against distribution histogram
    """
    z_sort = np.sort(table['feh'])
    i_0 = np.searchsorted(z_sort, xlims[0])
    i_1 = np.searchsorted(z_sort, xlims[1])

    mean, std = norm.fit(table['feh'][i_0:i_1])
    return mean, std


def fit_beta(table, xlims=(-2, 0.5)):
    """Returns fit of Beta Distribution to a given [Fe/H] table
    See: fit_gaussian()
    """
    z_sort = np.sort(table['feh'])
    i_0 = np.searchsorted(z_sort, xlims[0])
    i_1 = np.searchsorted(z_sort, xlims[1])

    loc = xlims[0]
    scale = xlims[1] - xlims[0]

    a, b, loc, scale = beta.fit(z_sort[i_0:i_1], floc=loc, fscale=scale)
    return a, b, loc, scale
