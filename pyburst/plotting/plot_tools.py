import os
import matplotlib.pyplot as plt

def default_plt_options():
    """Initialise default plot parameters"""
    params = {'mathtext.default': 'regular',
              'font.family': 'serif',
              'text.usetex': False}
    plt.rcParams.update(params)


default_plt_options()


def save_plot(fig, label, path, extensions=('.pdf',)):
    """Saves a figure passed to it
    """
    for ext in extensions:
        filename = f'{label}{ext}'
        filepath = os.path.join(path, filename)
        fig.savefig(filepath, bbox_inches="tight")


def set_axes(ax, title='', xlabel='', ylabel='', yscale='linear', xscale='linear',
             fontsize=14, yticks=True, xticks=True):
    """Standardised formatting of labels etc.
    """
    if not yticks:
        ax.axes.tick_params(axis='both', left='off', labelleft='off')
    if not xticks:
        ax.axes.tick_params(axis='both', bottom='off', labelbottom='off')

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def unit_scale(quantity):
    """Returns typical scale factor for given quantity
    """
    scales = {
        'rate': 1.0,
        'dt': 1.0,
        'fluence': 1e39,
        'peak': 1e38,
    }
    return scales.get(quantity, 1.0)


def full_label(quantity):
    """Returns string of formatted label with units
    """
    quant_str = quantity_label(quantity)
    unit_str = unit_label(quantity)

    if unit_str is '':
        label = f'{quant_str}'
    else:
        label = f'{quant_str} ({unit_str})'

    return label


def value_label(quantity, value, precision=None):
    """Returns formatted label including a float value
    """
    quant_str = quantity_label(quantity)
    if precision is None:
        return f'{quant_str}={value}'
    else:
        return f'{quant_str}={value:.{precision}f}'


def unit_label(quantity):
    """Returns units as a string, for given quantity
    """
    labels = {
        'rate': 'day$^{-1}$',
        'dt': 'h',
        'd': 'kpc',
        'i': 'deg',
        'fluence': '$10^{39}$ erg',
        'peak': '$10^{38}$ erg s$^{-1}$',
        'accrate': r'$\dot{M}_\mathrm{Edd}$',
        'mdot': r'$\dot{M}_\mathrm{Edd}$',
        'qb': r'MeV $\mathrm{nucleon}^{-1}$',
        'lum': '$10^{38}$ erg s$^{-1}$',
        'mass': r'$\mathrm{M_\odot}$',
        'radius': 'km',
    }
    return labels.get(quantity, '')


def quantity_label(quantity):
    """Returns formatted string of parameter label
    """
    labels = {
        'x': r'$X_0$',
        'z': r'$Z_\mathrm{CNO}$',
        'redshift': r'$(1+z)$',
        'd_b': r'$d \sqrt{\xi_\mathrm{b}}$',
        'xi_ratio': r'$\xi_\mathrm{p} / \xi_\mathrm{b}$',
        'xi_b': r'$\xi_\mathrm{b}$',
        'xi_p': r'$\xi_\mathrm{p}$',
        'm_gr': r'$M_\mathrm{GR}$',
        'm_nw': r'$M_\mathrm{NW}$',
        'xedd_ratio': r'$X_\mathrm{Edd} / X_0$',
        'xedd': r'$X_\mathrm{Edd}$',
        'qb': r'$Q_\mathrm{b}$',
        'accrate': r'$\dot{M}$',
        'mdot': r'$\dot{M}$',
        'dt': r'$\Delta t$',
        'fluence': r'$E_\mathrm{b}$',
        'peak': r'$L_\mathrm{peak}$',
        'rate': r'Burst rate',
        'alpha': r'$\alpha$',
        'length': 'Burst length',
        'lum': r'$\mathit{L}$',
        'mass': '$M$',
        'radius': '$R$',
    }
    return labels.get(quantity, quantity)


def convert_full_labels(quantities):
    """Returns formatted sequence of labels
    """
    keys = list(quantities)
    for i, key in enumerate(keys):
        keys[i] = full_label(key)
    return keys


def convert_mcmc_labels(param_keys):
    """Returns sequence of formatted MCMC parameter labels
    """
    keys = list(param_keys)

    for i, key in enumerate(keys):
        if 'qb' in key:
            label_str = r'$Q_\mathrm{b' + f'{key[-1]}' + '}$'
        elif 'mdot' in key:
            label_str = rf'$\dot{{M}}_{key[-1]}$'
        else:
            label_str = quantity_label(key)

        keys[i] = label_str

    return keys
