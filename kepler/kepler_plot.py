import numpy as np
import matplotlib.pyplot as plt

from pyburst.kepler import kepler_tools


def plot_dump_profile(run, batch, source, y_param, x_param='y', cycles=None,
                      basename='xrb', title=None,
                      display=True, prefix='', fontsize=14, marker='', relative=False,
                      xlims=None, ylims=None, legend=True):
    """Plot profile of given cycle dump, for some radial (zonal) quantity

    relative : bool
        plot y-axis relative to first cycle (y_n-y_0)
    """
    fig, ax = plt.subplots()
    cycles = kepler_tools.check_cycles(cycles, run=run, batch=batch, source=source)
    i0 = 2
    i1 = -3
    interp0 = None

    y_label = {'tn': r'$T$', 'xkn': r'$\kappa$'}.get(y_param, y_param)
    y_units = {'tn': 'K', 'xkn': r'cm$^2$ g$^{-1}$'}.get(y_param, '')
    x_label = {'y': 'y'}.get(x_param, x_param)
    x_units = {'y': r'g cm$^{-2}$'}.get(x_param, '')
    yscale = {'tn': 'log', 'xkn': 'linear'}.get(y_param, 'log')
    xscale = {'y': 'log'}.get(y_param, 'log')

    if relative:
        yscale = 'linear'
        dump0 = kepler_tools.load_dump(cycles[0], run, batch, source=source, basename=basename,
                                       prefix=prefix)
        interp0 = kepler_tools.interp_temp(dump=dump0)
        y_str = r'T - $T_{\#0}$ (K)'
    else:
        y_str = fr'{y_label} ({y_units})'

    for cycle in cycles:
        dump = kepler_tools.load_dump(cycle, run, batch, source=source, basename=basename,
                                      prefix=prefix)
        profile = kepler_tools.dump_dict(dump)
        x = profile[x_param][i0:i1]
        y = profile[y_param][i0:i1]
        if relative:
            y = y - interp0(x)

        ax.plot(x, y, label=f'#{cycle}', marker=marker)

    if title is None:
        title = f'{source}_{batch}_{run}'
    ax.set_title(title)
    x_str = fr'{x_label} ({x_units})'

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_xlabel(x_str, fontsize=fontsize)
    ax.set_ylabel(y_str, fontsize=fontsize)
    if legend:
        ax.legend()
    plt.tight_layout()

    if display:
        plt.show(block=False)
    return fig
