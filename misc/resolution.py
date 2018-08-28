import numpy as np
import matplotlib.pyplot as plt

from pygrids.grids import grid_analyser

# resolution tests

def plot(params, res_param, sources, bprop='rate'):
    """Plot burst properties for given resolution parameter

    parameters
    ----------
    params : {}
    res_param : str
        resolution parameter to plot on x-axis. One of [accmass, accrate]
    source : [str]
        list of source(s) to get models from
    """
    check_params(params)
    u_bprop = f'u_{bprop}'
    grids, sub_summ, sub_params = get_multi_subgrids(params=params, sources=sources)
    fig, ax = plt.subplots()

    for source in sources:
        ax.errorbar(x=sub_params[source][res_param], y=sub_summ[source][bprop],
                    yerr=sub_summ[source][u_bprop], ls='none', marker='o',
                    capsize=3, color='C0')

    ax.set_xscale('log')
    plt.show(block=False)

    return sub_params, sub_summ


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


def check_params(params, must_specify=('x', 'z', 'accrate', 'mass')):
    for param in must_specify:
        if param not in params:
            raise ValueError(f'{param} not specified in params')

