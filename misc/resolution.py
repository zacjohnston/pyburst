import numpy as np
import matplotlib.pyplot as plt
import os

from pygrids.grids import grid_analyser, grid_strings, grid_tools

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

# TODO add save plot, iterate over params

def save_all_plots(sources, ref_source, params=('x', 'z', 'mass', 'accrate'),
                   **kwargs):
    grids = get_multigrids(sources)
    source = get_not(sources, ref_source)
    unique_all = grids[source].unique_params
    unique_subset = {}
    for p in params:
        unique_subset[p] = unique_all[p]

    params_full = grid_tools.enumerate_params(unique_subset)

    n = len(params_full[params[0]])

    for i in range(n):
        params_sub = {}
        for p in params:
            params_sub[p] = params_full[p][i]
        plot(params=params_sub, sources=sources, ref_source=ref_source,
             grids=grids, save=True, display=False, title=False, **kwargs)

def plot(params, sources, ref_source, grid_version,
         bprops=('rate', 'fluence', 'peak', 'length'), figsize=(9, 10), shaded=False,
         display=True, save=False, grids=None, title=True, show_nbursts=True):
    """Plot burst properties for given resolution parameter

    parameters
    ----------
    params : dict
    ref_source : str
        source from which the reference model comes
    sources: set(str)
        list of source(s) to get models from
    bprops : [str]
    figsize : [int, int]
    shaded : bool
        shade between y_values of reference model
    """
    check_params(params)
    n = len(bprops)
    fig, ax = plt.subplots(n, 2, sharex=False, figsize=figsize)

    if grids is None:
        grids = get_multigrids(sources, grid_version=grid_version)

    for i, res_param in enumerate(reference_params):
        ref_value = reference_params[res_param]
        other_res_param = other_param[res_param]
        full_params = dict(params)
        full_params[other_res_param] = reference_params[other_res_param]
        sub_summ, sub_params = get_subgrids(grids, params=full_params)

        for j, bprop in enumerate(bprops):
            u_bprop = f'u_{bprop}'
            y_label = f'{y_labels[bprop]} ({y_units[bprop]})'
            y_factor = y_factors.get(bprop, 1)
            set_axes(ax[j, i], xscale='log',
                     ylabel=y_label if i == 0 else '',
                     xlabel=res_param if j == n-1 else '',
                     yticks=True if i == 0 else False)

            for source in sources:
                ref = source == ref_source
                x = sub_params[source][res_param]
                y = sub_summ[source][bprop] / y_factor
                yerr = sub_summ[source][u_bprop] / y_factor

                if show_nbursts:
                    n_bursts = sub_summ[source]['n_used']
                    for k in range(len(n_bursts)):
                        x_offset = 1.15
                        nb = n_bursts.iloc[k]
                        ax[j, i].text(x.iloc[k] * x_offset, y.iloc[k], f'{nb:.0f}',
                                      verticalalignment='center')

                if shaded and ref:
                    idx = np.where(x == ref_value)[0]
                    y_ref = y.iloc[idx]
                    yerr_ref = yerr.iloc[idx]
                    ax[j, i].fill_between(x_bounds[res_param],
                                          np.full(2, y_ref + yerr_ref),
                                          np.full(2, y_ref - yerr_ref), color='0.85')

                ax[j, i].errorbar(x=x, y=y, yerr=yerr, ls='none',
                                  marker='o', capsize=3, color=colors[ref])
    if title:
        ax[0, 0].set_title(params, fontsize=11)
    plt.tight_layout()

    if save:
        source = get_not(sources, ref_source)
        precisions = {'z': 4, 'x': 2, 'qb': 3, 'mass': 1, 'accrate': 2}
        fixed_str = ''
        for p, v in params.items():
            precision = precisions.get(p, 3)
            fixed_str += f'_{p}={v:.{precision}f}'

        filename = f'resolution_{source}{fixed_str}.png'
        path = os.path.join(grid_strings.plots_path(source), 'resolution')
        filepath = os.path.join(path, filename)
        print(f'Saving {filepath}')
        plt.savefig(filepath)
        plt.close(fig)
    else:
        plt.show(block=False)


def get_not(array, var):
    """Returns value in length-2 'array' that is not 'var'
    """
    copy = list(array)
    copy.remove(var)
    return copy[0]

def get_multigrids(sources, grid_version):
    grids = {}
    for source in sources:
        grids[source] = grid_analyser.Kgrid(source, grid_version=grid_version)
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
        ax.axes.tick_params(axis='both', left='off', labelleft='off')
    if not xticks:
        ax.axes.tick_params(axis='both', bottom='off', labelbottom='off')

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def check_params(params, must_specify=('x', 'z', 'accrate', 'mass')):
    for param in must_specify:
        if param not in params:
            raise ValueError(f'{param} not specified in params')

