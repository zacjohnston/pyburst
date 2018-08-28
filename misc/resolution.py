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

other_param = {'accmass': 'accdepth',
               'accdepth': 'accmass'}
x_bounds = {'accmass': [1e15, 1e17],
            'accdepth': [1e19, 1e21]}

colors = {True: 'C1',
          False: 'C0'}

def plot(params, sources, ref_source, bprops=('rate', 'fluence', 'peak', 'length'),
         figsize=(9, 10), shaded=False):
    """Plot burst properties for given resolution parameter

    parameters
    ----------
    params : {}
    ref_source : str
        source from which the reference model comes
    sources: [str]
        list of source(s) to get models from
    bprops : [str]
    figsize : [int, int]
    shaded : bool
        shade between y_values of reference model
    """
    check_params(params)
    n = len(bprops)
    fig, ax = plt.subplots(n, 2, sharex=False, figsize=figsize)
    grids = get_multigrids(sources)

    for i, res_param in enumerate(reference_params):
        ref_value = reference_params[res_param]
        other = other_param[res_param]
        full_params = dict(params)
        full_params[other] = reference_params[other]
        sub_summ, sub_params = get_subgrids(grids, params=full_params)

        for j, bprop in enumerate(bprops):
            u_bprop = f'u_{bprop}'
            y_label = f'{y_labels[bprop]} ({y_units[bprop]})'
            y_factor = y_factors.get(bprop, 1)
            set_axes(ax[j, i], xscale='log',
                     ylabel=y_label if i == 0 else '',
                     xlabel=res_param if j == n-1 else '',
                     # title=params if (i==0 and j==0) else '',
                     yticks=True if i == 0 else False)

            for source in sources:
                ref = source == ref_source
                x = sub_params[source][res_param]
                y = sub_summ[source][bprop] / y_factor
                yerr = sub_summ[source][u_bprop] / y_factor

                if shaded and ref:
                    idx = np.where(x == ref_value)[0]
                    y_ref = y.iloc[idx]
                    yerr_ref = yerr.iloc[idx]
                    ax[j, i].fill_between(x_bounds[res_param],
                                          np.full(2, y_ref + yerr_ref),
                                          np.full(2, y_ref - yerr_ref), color='0.85')

                ax[j, i].errorbar(x=x, y=y, yerr=yerr, ls='none',
                                  marker='o', capsize=3, color=colors[ref])
    plt.tight_layout()
    plt.show(block=False)


def get_multigrids(sources):
    grids = {}
    for source in sources:
        grids[source] = grid_analyser.Kgrid(source)
    return grids

def get_subgrids(grids, params):
    """Returns subgrids of multiple given sources
    """
    sub_params = {}
    sub_summ = {}

    for source in grids:
        sub_params[source] = grids[source].get_params(params=params)
        sub_summ[source] = grids[source].get_summ(params=params)

    return sub_summ, sub_params


def set_axes(ax, title='', xlabel='', ylabel='', yscale='linear', xscale='linear',
             fontsize=14, yticks=True, xticks=True):
    if not yticks:
        # ax.yaxis.set_ticks_position('none')
        # ax.yaxis.set_ticks([])
        ax.axes.tick_params(axis='both', left='off', labelleft='off')
    if not xticks:
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.set_ticks([])

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def check_params(params, must_specify=('x', 'z', 'accrate', 'mass')):
    for param in must_specify:
        if param not in params:
            raise ValueError(f'{param} not specified in params')

