import numpy as np
import matplotlib.pyplot as plt

from pygrids.grids import grid_analyser

# resolution tests

y_factors = {'dt': 3600,
             'fluence': 1e39,
             'peak': 1e38,
             }

y_labels = {'dt': '$\Delta t$',
            'rate': 'Burst rate',
            'fluence': '$E_b$',
            'peak': '$L_{peak}$',
            'length': 'Burst length',
            }

y_units = {'dt': 'hr',
           'rate': 'day$^{-1}$',
           'fluence': '$10^39$ erg',
           'peak': '$10^38$ erg s$^{-1}$',
           'length': 's',
           }

def plot(params, res_param, sources, bprop='rate'):
    """Plot burst properties for given resolution parameter

    parameters
    ----------
    params : {}
    res_param : str
        resolution parameter to plot on x-axis. One of [accmass, accrate]
    sources: [str]
        list of source(s) to get models from
    bprop : str
    """
    check_params(params)
    u_bprop = f'u_{bprop}'
    grids, sub_summ, sub_params = get_multi_subgrids(params=params, sources=sources)
    fig, ax = plt.subplots()

    y_label = f'{y_labels[bprop]} ({y_units[bprop]})'
    y_factor = y_factors.get(bprop, 1)

    for source in sources:
        ax.errorbar(x=sub_params[source][res_param],
                    y=sub_summ[source][bprop] / y_factor,
                    yerr=sub_summ[source][u_bprop] / y_factor,
                    ls='none', marker='o', capsize=3, color='C0')

    set_axes(ax, xlabel=res_param, ylabel=y_label, xscale='log')
    plt.show(block=False)


def get_multi_subgrids(params, sources):
    """Returns subgrids of multiple given sources
    """
    grids = {}
    sub_params = {}
    sub_summ = {}

    for source in sources:
        grids[source] = grid_analyser.Kgrid(source)
        sub_params[source] = grids[source].get_params(params=params)
        sub_summ[source] = grids[source].get_summ(params=params)

    return grids, sub_summ, sub_params


def set_axes(ax, title='', xlabel='', ylabel='', yscale='linear', xscale='linear', fontsize=14):
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def check_params(params, must_specify=('x', 'z', 'accrate', 'mass')):
    for param in must_specify:
        if param not in params:
            raise ValueError(f'{param} not specified in params')

