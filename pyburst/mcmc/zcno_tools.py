import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

# TODO:p1.5/3.75hhh
#   - plot_distribution(var='feh')

def plot_hist(table, var='feh', bins=100, histtype='step'):
    """Plots histogram of the given table variable
    """
    xlabels = {'feh': '[Fe/H]'}

    fig, ax = plt.subplots()
    ax.hist(table[var], bins=bins, density=1, histtype=histtype)

    xlabel = xlabels.get(var, var)
    ax.set_xlabel(xlabel)

    plt.show(block=False)
    return fig, ax


def fit_z_gaussian(table, plot=False, xlims=(-3, 3)):
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

        fig, ax = plt.subplots()
        ax.hist(table['feh'], bins=100, density=1.0)
        ax.plot(x, y)
        plt.show(block=False)

    return mean, std
