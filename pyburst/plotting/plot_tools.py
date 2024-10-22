import os
import matplotlib.pyplot as plt



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
        'accrate': r'$\dot{m}_\mathrm{Edd}$',
        'dt': 'h',
        'd': 'kpc',
        'd_b': 'kpc',
        'fluence': '$10^{39}$ erg',
        'g': '$10^{14}$ cm s$^{-2}$',
        'i': 'deg',
        'lum': '$10^{38}$ erg s$^{-1}$',
        'm_gr': r'$\mathrm{M_\odot}$',
        'M': r'$\mathrm{M_\odot}$',
        'mdot': r'$\dot{m}_\mathrm{Edd}$',
        'peak': '$10^{38}$ erg s$^{-1}$',
        'qb': r'MeV $\mathrm{nucleon}^{-1}$',
        'rate': 'day$^{-1}$',
        'R': 'km',
    }
    return labels.get(quantity, '')


def quantity_label(quantity):
    """Returns formatted string of parameter label
    """
    labels = {
        'accrate': r'$\dot{m}$',
        'alpha': r'$\alpha$',
        'd_b': r'$d \sqrt{\xi_\mathrm{b}}$',
        'dt': r'$\Delta t$',
        'fluence': r'$E_\mathrm{b}$',
        'length': 'Burst length',
        'lum': r'$\mathit{L}$',
        'm_gr': '$M$',
        'm_nw': r'$M_\mathrm{NW}$',
        'mdot': r'$\dot{m}$',
        'peak': r'$L_\mathrm{peak}$',
        'qb': r'$Q_\mathrm{b}$',
        'rate': r'Burst rate',
        'redshift': '$z$',
        'x': r'$X_0$',
        'xedd': r'$X_\mathrm{Edd}$',
        'xedd_ratio': r'$X_\mathrm{Edd} / X_0$',
        'xi_ratio': r'$\xi_\mathrm{p} / \xi_\mathrm{b}$',
        'xi_b': r'$\xi_\mathrm{b}$',
        'xi_p': r'$\xi_\mathrm{p}$',
        'z': r'$Z_\mathrm{CNO}$',
    }
    return labels.get(quantity, f'${quantity}$')


def convert_full_labels(quantities):
    """Returns formatted sequence of labels
    """
    keys = list(quantities)
    for i, key in enumerate(keys):
        keys[i] = full_label(key)
    return keys


def convert_mcmc_labels(param_keys, unit_labels=False):
    """Returns sequence of formatted MCMC parameter labels
    """
    keys = list(param_keys)

    for i, key in enumerate(keys):
        if 'qb' in key:
            label_str = r'$Q_\mathrm{b,' + f'{key[-1]}' + '}$'
        elif 'mdot' in key:
            label_str = rf'$\dot{{m}}_{key[-1]}$'
        elif 'Mdot' in key:
            label_str = rf'$\dot{{M}}_{key[-1]}$'
        else:
            if unit_labels:
                label_str = full_label(key)
            else:
                label_str = quantity_label(key)

        keys[i] = label_str

    return keys
