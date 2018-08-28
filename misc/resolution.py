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

reference_params = {'accmass': 1e16,
                    'accdepth': 1e20}

x_bounds = {'accmass': [1e15, 1e17],
            'accdepth': [1e19, 1e21]}

def plot(params, sources, res_param, bprops=('rate', 'fluence', 'peak', 'length'),
         figsize=(6, 10), shaded=True):
    """Plot burst properties for given resolution parameter

    parameters
    ----------
    params : {}
    res_param : str
        resolution parameter to plot on x-axis. One of [accmass, accrate]
    sources: [str]
        list of source(s) to get models from
    bprops : [str]
    figsize : [int, int]
    shaded : bool
    """
    check_params(params)
    ref = reference_params[res_param]
    grids, sub_summ, sub_params = get_multi_subgrids(params=params, sources=sources)
    n = len(bprops)
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=figsize)

    for i, bprop in enumerate(bprops):
        u_bprop = f'u_{bprop}'
        y_label = f'{y_labels[bprop]} ({y_units[bprop]})'
        y_factor = y_factors.get(bprop, 1)
        set_axes(ax[i], ylabel=y_label, xscale='log',
                 xlabel=res_param if i == n-1 else '')

        for source in sources:
            x = sub_params[source][res_param]
            y = sub_summ[source][bprop] / y_factor
            yerr = sub_summ[source][u_bprop] / y_factor

            if shaded:
                idx = np.where(x == ref)[0]
                if len(idx) == 1:
                    y_ref = y.iloc[idx]
                    yerr_ref = yerr.iloc[idx]
                    ax[i].fill_between(x_bounds[res_param],
                                       np.full(2, y_ref + yerr_ref),
                                       np.full(2, y_ref - yerr_ref), color='0.8')

            ax[i].errorbar(x=x, y=y, yerr=yerr, ls='none',
                           marker='o', capsize=3, color='C0')
    plt.tight_layout()
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

